"""
Create data products from loop simulations
"""
import os
import warnings
import logging
import itertools
import toolz

import numpy as np
from scipy.interpolate import splev, splprep, interp1d
import scipy.ndimage
import astropy.units as u
from sunpy.sun import constants
from sunpy.coordinates.frames import HeliographicStonyhurst
import h5py
try:
    import distributed
except ImportError:
    warnings.warn('Dask distributed scheduler required for parallel execution')

from synthesizAR.util import get_keys


class Observer(object):
    """
    Class for assembling AR from loops and creating data products from 2D projections.

    Parameters
    ----------
    field : `~synthesizAR.Field`
    instruments : `list`
    parallel : `bool`

    Examples
    --------
    """
    def __new__(cls, *args, parallel=False, **kwargs):
        if parallel:
            return ObserverParallel(*args, **kwargs)
        else:
            return ObserverSerial(*args, **kwargs)


class ObserverSerial(object):

    def __init__(self, field, instruments):
        self.field = field
        self.instruments = instruments
        self._channels_setup()

    def _channels_setup(self):
        """
        Tell each channel of each detector which wavelengths fall in it.
        """
        for instr in self.instruments:
            for channel in instr.channels:
                if channel['wavelength_range'] is not None:
                    channel['model_wavelengths'] = []
                    for wvl in self.field.loops[0].resolved_wavelengths:
                        if channel['wavelength_range'][0] <= wvl <= channel['wavelength_range'][-1]:
                            channel['model_wavelengths'].append(wvl)
                    if channel['model_wavelengths']:
                        channel['model_wavelengths'] = u.Quantity(channel['model_wavelengths'])

    @u.quantity_input
    def _interpolate_loops(self, ds: u.cm):
        """
        Interpolate all loops to a resolution (`ds`) below the minimum bin width
        of all of the instruments. This ensures that the image isn't 'patchy'
        when it is binned.
        """
        # Interpolate all loops in HEEQ coordinates
        total_coordinates = []
        interpolated_loop_coordinates = []
        for loop in self.field.loops:
            n_interp = int(np.ceil((loop.full_length/ds).decompose()))
            interpolated_s = np.linspace(loop.field_aligned_coordinate.value[0],
                                         loop.field_aligned_coordinate.value[-1], n_interp)
            interpolated_loop_coordinates.append(interpolated_s)
            nots, _ = splprep(loop.coordinates.cartesian.xyz.value)
            total_coordinates.append(np.array(splev(np.linspace(0, 1, n_interp), nots)).T)

        total_coordinates = np.vstack(total_coordinates) * loop.coordinates.cartesian.xyz.unit

        return total_coordinates, interpolated_loop_coordinates

    def build_detector_files(self, savedir, ds, **kwargs):
        """
        Create files to store interpolated counts before binning.

        .. note:: After creating the instrument objects and passing them to the observer,
                  it is always necessary to call this method.
        """
        file_template = os.path.join(savedir, '{}_counts.h5')
        total_coordinates, self._interpolated_loop_coordinates = self._interpolate_loops(ds)
        interp_s_shape = (int(np.median([s.shape for s in self._interpolated_loop_coordinates])),)
        for instr in self.instruments:
            chunks = kwargs.get('chunks', instr.observing_time.shape + interp_s_shape)
            dset_shape = instr.observing_time.shape + (len(total_coordinates),)
            instr.build_detector_file(file_template, dset_shape, chunks, self.field, **kwargs)
            with h5py.File(instr.counts_file, 'a') as hf:
                if 'coordinates' not in hf:
                    dset = hf.create_dataset('coordinates', data=total_coordinates.value)
                    dset.attrs['unit'] = total_coordinates.unit.to_string()

    def flatten_detector_counts(self, **kwargs):
        """
        Calculate intensity for each loop, interpolate it to the appropriate spatial and temporal
        resolution, and store it. This is done either in serial or parallel.
        """
        emission_model = kwargs.get('emission_model', None)
        interpolate_hydro_quantities = kwargs.get('interpolate_hydro_quantities', True)
        for instr in self.instruments:
            with h5py.File(instr.counts_file, 'a', driver=kwargs.get('hdf5_driver', None)) as hf:
                start_index = 0
                if interpolate_hydro_quantities:
                    for interp_s, loop in zip(self._interpolated_loop_coordinates, self.field.loops):
                        for q in ['velocity_x', 'velocity_y', 'velocity_z', 'electron_temperature',
                                  'ion_temperature', 'density']:
                            val = instr.interpolate(q, loop, interp_s)
                            instr.commit(val, hf[q], start_index)
                        start_index += interp_s.shape[0]
                instr.flatten_serial(self.field.loops, self._interpolated_loop_coordinates, hf,
                                     emission_model=emission_model)

    @staticmethod
    def assemble_map(observed_map, filename, time):
        observed_map.meta['tunit'] = time.unit.to_string()
        observed_map.meta['t_obs'] = time.value
        observed_map.save(filename)

    def bin_detector_counts(self, savedir, **kwargs):
        """
        Assemble pipelines for building maps at each timestep.

        Build pipeline for computing final synthesized data products. This can be done
        either in serial or parallel.

        Parameters
        ----------
        savedir : `str`
            Top level directory to save data products in
        """
        file_path_template = os.path.join(savedir, '{}', '{}', 'map_t{:06d}.fits')
        for instr in self.instruments:
            bins, bin_range = instr.make_detector_array(self.field)
            with h5py.File(instr.counts_file, 'r') as hf:
                reference_time = u.Quantity(hf['time'],
                                            get_keys(hf['time'].attrs, ('unit', 'units')))
            for channel in instr.channels:
                header = instr.make_fits_header(self.field, channel)
                dirname = os.path.dirname(file_path_template.format(instr.name, channel['name'], 0))
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                for time in instr.observing_time:
                    i_time = np.where(reference_time == time)[0][0]
                    raw_map = instr.detect(channel, i_time, header, bins, bin_range)
                    file_path = file_path_template.format(instr.name, channel['name'], i_time)
                    self.assemble_map(raw_map, file_path, time)


class ObserverParallel(ObserverSerial):

    def flatten_detector_counts(self, **kwargs):
        """
        Build custom Dask graph interpolating quantities for each in loop in time and space.
        """
        client = distributed.get_client()
        emission_model = kwargs.get('emission_model', None)
        interpolate_hydro_quantities = kwargs.get('interpolate_hydro_quantities', True)
        futures = {}
        start_indices = np.insert(np.array(
            [s.shape[0] for s in self._interpolated_loop_coordinates]).cumsum()[:-1], 0, 0)
        for instr in self.instruments:
            interp_futures = []
            if interpolate_hydro_quantities:
                for q in ['velocity_x', 'velocity_y', 'velocity_z', 'electron_temperature',
                          'ion_temperature', 'density']:
                    loop_futures = []
                    for i, loop in enumerate(self.field.loops):
                        y = client.submit(instr.interpolate, q, loop, self._interpolated_loop_coordinates[i])
                        loop_futures.append(client.submit(instr.write_to_hdf5, y, start_indices[i], q))
                    # Block until complete
                    distributed.client.wait(loop_futures)
                    interp_futures += loop_futures

            # Calculate and interpolate channel counts for instrument
            counts_futures = instr.flatten_parallel(self.field.loops,
                                                    self._interpolated_loop_coordinates,
                                                    emission_model=emission_model)
            futures[f'{instr.name}'] = interp_futures + counts_futures

        return futures

    def bin_detector_counts(self, savedir, **kwargs):
        """
        Assemble pipelines for building maps at each timestep.

        Build pipeline for computing final synthesized data products. This can be done
        either in serial or parallel.

        Parameters
        ----------
        savedir : `str`
            Top level directory to save data products in
        """
        futures = {instr.name: {} for instr in self.instruments}
        client = distributed.get_client()
        file_path_template = os.path.join(savedir, '{}', '{}', 'map_t{:06d}.fits')
        for instr in self.instruments:
            bins, bin_range = instr.make_detector_array(self.field)
            with h5py.File(instr.counts_file, 'r') as hf:
                reference_time = u.Quantity(hf['time'], get_keys(hf['time'].attrs, ('unit', 'units')))
            for channel in instr.channels:
                header = instr.make_fits_header(self.field, channel)
                dirname = os.path.dirname(file_path_template.format(instr.name, channel['name'], 0))
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                futures[instr.name][channel['name']] = []
                for time in instr.observing_time:
                    i_time = np.where(reference_time == time)[0][0]
                    m = client.submit(instr.detect, channel, i_time, header, bins, bin_range)
                    file_path = file_path_template.format(instr.name, channel['name'], i_time)
                    futures[instr.name][channel['name']].append(
                        client.submit(self.assemble_map, m, file_path,))
                distributed.client.wait(futures[instr.name][channel['name']])

        return futures
