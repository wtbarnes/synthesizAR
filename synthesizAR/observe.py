"""
Create data products from loop simulations
"""

import os
import warnings
import logging
from itertools import groupby

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

from synthesizAR.util import heeq_to_hcc


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
    def _interpolate_loops(self, instrument, ds: u.cm):
        """
        Interpolate loops to common resolution
        
        Interpolate all loops to a resolution (`ds`) below the minimum bin width 
        of all of the instruments. This ensures that the image isn't 'patchy' 
        when it is binned.
        """
        # If no observer coordinate specified, use that of the magnetogram
        if instrument.observer_coordinate is None:
            # FIXME: Setting attributes of other classes like this is bad!
            instrument.observer_coordinate = (self.field.magnetogram.observer_coordinate
                                              .transform_to(HeliographicStonyhurst))
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

    def build_detector_files(self, savedir, ds=None, **kwargs):
        """
        Create files to store interpolated counts before binning.

        Note
        ----
        After creating the instrument objects and passing them to the observer,
        it is always necessary to call this method.
        """
        file_template = os.path.join(savedir, '{}_counts.h5')
        self._interpolated_loop_coordinates = {}
        for instr in self.instruments:
            ds = 0.5 * min(instr.resolution.x, instr.resolution.y) if ds is None else ds
            total_coordinates, _interpolated_loop_coordinates = self._interpolate_loops(instr, ds)
            self._interpolated_loop_coordinates[instr.name] = _interpolated_loop_coordinates
            interp_s_shape = (int(np.median([s.shape for s in _interpolated_loop_coordinates])),)
            chunks = kwargs.get('chunks', instr.observing_time.shape+interp_s_shape)
            dset_shape = instr.observing_time.shape+(len(total_coordinates),)
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
            if 'client' not in kwargs:
                raise ValueError('Dask scheduler client required for parallel execution.')
            else:
                client = kwargs['client']
                del kwargs['client']
            return self._flatten_detector_counts_parallel(client, **kwargs)
        else:
            self._flatten_detector_counts_serial(**kwargs)

    @staticmethod
    def _get_los_velocity(velocity, observer_coordinate):
        """
        Calculate LOS velocity component for a given observer
        """
        _, _, v_los = heeq_to_hcc(velocity[:, :, 0], velocity[:, :, 1], velocity[:, :, 2],
                                  observer_coordinate)
        return -v_los

    def _flatten_detector_counts_serial(self, **kwargs):
        emission_model = kwargs.get('emission_model', None)
        for instr in self.instruments:
            with h5py.File(instr.counts_file, 'a', driver=kwargs.get('hdf5_driver', None)) as hf:
                start_index = 0
                for interp_s, loop in zip(self._interpolated_loop_coordinates[instr.name],
                                          self.field.loops):
                    params = (loop, instr.observing_time, interp_s)
                    los_velocity = self._get_los_velocity(loop.velocity_xyz,
                                                          instr.observer_coordinate)
                    self.commit(instr.interpolate_and_store(los_velocity, *params),
                                hf['los_velocity'], start_index)
                    self.commit(instr.interpolate_and_store(loop.electron_temperature, *params),
                                hf['electron_temperature'], start_index)
                    self.commit(instr.interpolate_and_store(loop.ion_temperature, *params),
                                hf['ion_temperature'], start_index)
                    self.commit(instr.interpolate_and_store(loop.density, *params), hf['density'],
                                start_index)
                    start_index += interp_s.shape[0]
                instr.flatten_serial(self.field.loops,
                                     self._interpolated_loop_coordinates[instr.name], hf,
                                     emission_model=emission_model)

    @staticmethod
    def commit(y, dset, start_index):
        if 'units' not in dset.attrs:
            dset.attrs['units'] = y.unit.to_string()
        dset[:, start_index:(start_index + y.shape[1])] = y.value

    def _flatten_detector_counts_parallel(self, client, **kwargs):
        """
        Build Dask Futures for interpolating quantities for each in loop in time and space.
        """
        emission_model = kwargs.get('emission_model', None)
        # Build futures for each instrument
        futures = {}
        for instr in self.instruments:
            instr_futures = {'los_velocity': [], 'electron_temperature': [], 'density': [],
                             'ion_temperature': []}
            tmp_file_path = os.path.join(instr.tmp_file_template, '{}.npy')
            # Build futures for each loop
            for interp_s, loop in zip(self._interpolated_loop_coordinates[instr.name],
                                      self.field.loops):
                params = (loop, instr.observing_time, interp_s)
                los_velocity = client.submit(self._get_los_velocity, loop.velocity_xyz,
                                             instr.observer_coordinate)
                instr_futures['los_velocity'].append(client.submit(
                                            instr.interpolate_and_store,
                                            los_velocity, *params,
                                            tmp_file_path.format('los_velocity', loop.name)))
                instr_futures['electron_temperature'].append(client.submit(
                                            instr.interpolate_and_store,
                                            loop.electron_temperature, *params,
                                            tmp_file_path.format('electron_temperature', loop.name)))
                instr_futures['ion_temperature'].append(client.submit(
                                            instr.interpolate_and_store,
                                            loop.ion_temperature, *params,
                                            tmp_file_path.format('electron_temperature', loop.name)))
                instr_futures['density'].append(client.submit(
                                            instr.interpolate_and_store,
                                            loop.density, *params,
                                            tmp_file_path.format('density', loop.name)))

            # Get futures for counts calculation
            counts_futures = instr.flatten_parallel(
                                            self.field.loops,
                                            self._interpolated_loop_coordinates[instr.name],
                                            tmp_file_path, client, emission_model=emission_model)
            # Combine with other futures and add assemble future
            instr_futures.update(counts_futures)
            futures[f'{instr.name}'] = client.submit(self.assemble_arrays, instr_futures,
                                                     instr.counts_file, **kwargs)

        return futures

    @staticmethod
    def assemble_arrays(futures, h5py_filename, **kwargs):
        with h5py.File(h5py_filename, 'a', driver=kwargs.get('hdf5_driver', None)) as hf:
            for key in futures:
                dset = hf[key]
                start_index = 0
                for filename, units in futures[key]:
                    tmp = u.Quantity(np.load(filename), units)
                    Observer.commit(tmp, dset, start_index)
                    os.remove(filename)
                    start_index += tmp.shape[1]

    @staticmethod
    def assemble_map(observed_map, filename, time):
        observed_map.meta['tunit'] = time.unit.to_string()
        observed_map.meta['t_obs'] = time.value
        observed_map.save(filename)

    def bin_detector_counts(self, savedir, **kwargs):
        """
        Assemble pipelines for building maps at each timestep.

        Build pipeline for computing final synthesized data products. This can be done
        either in serial or parallel. The Dask.distributed scheduler is required for the latter
        option.

        Parameters
        ----------
        savedir : `str`
            Top level directory to save data products in

        Other Parameters
        ----------------
        client : `~distributed.Client`
        """
        if self.parallel:
            client = kwargs.get('client', None)
            if client is None:
                raise ValueError('Dask scheduler client required for parallel execution.')
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
                        raw_map = client.submit(instr.detect, channel, i_time, header, bins,
                                                bin_range)
                        distributed.fire_and_forget(client.submit(
                            self.assemble_map, raw_map, file_path, time))
                    else:
                        raw_map = instr.detect(channel, i_time, header, bins, bin_range)
                        self.assemble_map(raw_map, file_path, time)
