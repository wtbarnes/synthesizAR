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

from synthesizAR.util import heeq_to_hcc, future_property


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
    Notes
    -----
    """

    def __init__(self, field, instruments, parallel=False):
        self.parallel = parallel
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
        Interpolate loops to common resolution
        
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
            nots, _ = splprep(loop.coordinates.value.T)
            total_coordinates.append(np.array(splev(np.linspace(0, 1, n_interp), nots)).T)

        total_coordinates = np.vstack(total_coordinates) * loop.coordinates.unit

        return total_coordinates, interpolated_loop_coordinates

    def build_detector_files(self, savedir, ds, **kwargs):
        """
        Create files to store interpolated counts before binning.

        Note
        ----
        After creating the instrument objects and passing them to the observer,
        it is always necessary to call this method.
        """
        file_template = os.path.join(savedir, '{}_counts.h5')
        total_coordinates, self._interpolated_loop_coordinates = self._interpolate_loops(ds)
        interp_s_shape = (int(np.median([s.shape for s in self._interpolated_loop_coordinates])),)
        for instr in self.instruments:
            # If no observer coordinate specified, use that of the magnetogram
            if instr.observer_coordinate is None:
                # FIXME: Setting attributes of other classes like this is bad!
                instr.observer_coordinate = (self.field.magnetogram.observer_coordinate
                                             .transform_to(HeliographicStonyhurst))
            chunks = kwargs.get('chunks', instr.observing_time.shape + interp_s_shape)
            dset_shape = instr.observing_time.shape + (len(total_coordinates),)
            instr.build_detector_file(file_template, dset_shape, chunks, self.field,
                                      parallel=self.parallel, **kwargs)
            with h5py.File(instr.counts_file, 'a') as hf:
                if 'coordinates' not in hf:
                    dset = hf.create_dataset('coordinates', data=total_coordinates.value)
                    dset.attrs['units'] = total_coordinates.unit.to_string()

    def flatten_detector_counts(self, **kwargs):
        """
        Calculate intensity for each loop, interpolate it to the appropriate spatial and temporal
        resolution, and store it. This is done either in serial or parallel.
        """
        if self.parallel:
            return self._flatten_detector_counts_parallel(**kwargs)
        else:
            self._flatten_detector_counts_serial(**kwargs)

    def _flatten_detector_counts_serial(self, **kwargs):
        emission_model = kwargs.get('emission_model', None)
        for instr in self.instruments:
            with h5py.File(instr.counts_file, 'a', driver=kwargs.get('hdf5_driver', None)) as hf:
                start_index = 0
                for interp_s, loop in zip(self._interpolated_loop_coordinates, self.field.loops):
                    instr.commit(instr.interpolate_and_store(
                        instr.observing_time.value, loop.velocity_x, loop, interp_s),
                        hf['velocity_x'], start_index)
                    instr.commit(instr.interpolate_and_store(
                        instr.observing_time.value, loop.velocity_y, loop, interp_s),
                        hf['velocity_y'], start_index)
                    instr.commit(instr.interpolate_and_store(
                        instr.observing_time.value, loop.velocity_z, loop, interp_s),
                        hf['velocity_z'], start_index)
                    instr.commit(instr.interpolate_and_store(
                        instr.observing_time.value, loop.electron_temperature, loop, interp_s),
                        hf['electron_temperature'], start_index)
                    instr.commit(instr.interpolate_and_store(
                        instr.observing_time.value, loop.ion_temperature, loop, interp_s),
                        hf['ion_temperature'], start_index)
                    instr.commit(instr.interpolate_and_store(
                        instr.observing_time.value, loop.density, loop, interp_s),
                        hf['density'], start_index)
                    start_index += interp_s.shape[0]
                instr.flatten_serial(self.field.loops, self._interpolated_loop_coordinates, hf,
                                     emission_model=emission_model)

    def _flatten_detector_counts_parallel(self, **kwargs):
        """
        Build custom Dask graph interpolating quantities for each in loop in time and space.
        """
        client = distributed.get_client()
        emission_model = kwargs.get('emission_model', None)
        futures = {}
        start_indices = np.insert(np.array(
            [s.shape[0] for s in self._interpolated_loop_coordinates]).cumsum()[:-1], 0, 0)
        for instr in self.instruments:
            # Need to create lock for writing HDF5 files
            lock = distributed.Lock(name=f'hdf5_{instr.name}')
            # Create temporary files where interpolated results will be written
            tmp_dir = os.path.join(os.path.dirname(instr.counts_file), 'tmp_parallel_files')
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
            interp_futures = []
            for q in ['velocity_x', 'velocity_y', 'velocity_z', 'electron_temperature',
                      'ion_temperature', 'density']:
                paths = [os.path.join(tmp_dir, f'{l.name}_{instr.name}_{q}.npz')
                         for l in self.field.loops]
                partial_interp = toolz.curry(instr.interpolate_and_store)(
                    instr.observing_time.value, q)
                loop_futures = client.map(partial_interp, self.field.loops,
                                          self._interpolated_loop_coordinates, start_indices, paths)
                interp_futures.append(client.submit(
                    instr.assemble_arrays, loop_futures, q, instr.counts_file, lock))

            # Get tasks for instrument-specific calculations
            # counts_futures = instr.flatten_parallel(self.field.loops,
            #                                         self._interpolated_loop_coordinates,
            #                                         tmp_dir, emission_model=emission_model)

            futures[f'{instr.name}'] = client.submit(self._cleanup, interp_futures)

        return futures

    @staticmethod
    def _cleanup(filenames):
        for f in itertools.chain.from_iterable(filenames):
            os.remove(f)
        os.rmdir(os.path.dirname(f))

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
        tasks = {instr.name: [] for instr in self.instruments} if self.parallel else None
        file_path_template = os.path.join(savedir, '{}', '{}', 'map_t{:06d}.fits')
        for instr in self.instruments:
            bins, bin_range = instr.make_detector_array(self.field)
            with h5py.File(instr.counts_file, 'r') as hf:
                reference_time = u.Quantity(hf['time'], hf['time'].attrs['units'])
            for channel in instr.channels:
                header = instr.make_fits_header(self.field, channel)
                for time in instr.observing_time:
                    try:
                        i_time = np.where(reference_time == time)[0][0]
                    except IndexError as err:
                        raise IndexError(f'{time} not a valid observing time for {instr.name}') from err
                    file_path = file_path_template.format(instr.name, channel['name'], i_time)
                    if not os.path.exists(os.path.dirname(file_path)):
                        os.makedirs(os.path.dirname(file_path))
                    if self.parallel:
                        m = dask.delayed(instr.detect)(channel, i_time, header, bins, bin_range)
                        tasks[f'{instr.name}'].append(dask.delayed(self.assemble_map)(
                            m, file_path, time))
                    else:
                        raw_map = instr.detect(channel, i_time, header, bins, bin_range)
                        self.assemble_map(raw_map, file_path, time)

        return tasks
