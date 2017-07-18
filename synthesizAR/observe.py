"""
Create data products loop simulations
"""

import os
import logging
import pickle

import numpy as np
from scipy.interpolate import splev, splprep, interp1d
import scipy.ndimage
import matplotlib.colors
import astropy.units as u
import sunpy.map
import h5py

from synthesizAR.util import EMCube

class Observer(object):
    """
    Class for assembling AR from loops and creating data products from 2D projections.

    Parameters
    ----------
    field
    instruments
    ds : optional

    Examples
    --------
    Notes
    -----
    """

    def __init__(self, field, instruments, ds=None, line_of_sight=(0,0,-1)):
        self.logger = logging.getLogger(name=type(self).__name__)
        self.field = field
        self.instruments = instruments
        self._channels_setup()
        self.line_of_sight = line_of_sight
        if ds is None:
            ds = 0.1*np.min([min(instr.resolution.x.value, instr.resolution.y.value)
                             for instr in self.instruments])*self.instruments[0].resolution.x.unit
        ds = self.field._convert_angle_to_length(ds)
        self._interpolate_loops(ds)

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

    def _interpolate_loops(self, ds):
        """
        Interpolate all loops to a resolution (`ds`) below the minimum bin width of all of the
        instruments. This ensures that the image isn't 'patchy' when it is binned.
        """
        # FIXME: memory requirements for this list will grow with number of loops, consider saving 
        # it to the instrument files, both the interpolated s and total_coordinates
        self.total_coordinates = []
        self._interpolated_loop_coordinates = []
        for loop in self.field.loops:
            self.logger.debug('Interpolating loop {}'.format(loop.name))
            n_interp = int(np.ceil(loop.full_length/ds))
            interpolated_s = np.linspace(loop.field_aligned_coordinate.value[0],
                                         loop.field_aligned_coordinate.value[-1], n_interp)
            self._interpolated_loop_coordinates.append(interpolated_s)
            nots, _ = splprep(loop.coordinates.value.T)
            _tmp = splev(np.linspace(0, 1, n_interp), nots)
            self.total_coordinates += [(x,y,z) for x, y, z in zip(_tmp[0], _tmp[1], _tmp[2])]

        self.total_coordinates = np.array(self.total_coordinates)*loop.coordinates.unit

    def build_detector_files(self, savedir):
        """
        Create files to store interpolated counts before binning.
        """
        file_template = os.path.join(savedir, '{}_counts.h5')
        for instr in self.instruments:
            instr.total_coordinates = self.total_coordinates
            instr.make_detector_array(self.field)
            instr.build_detector_file(self.field, len(self.total_coordinates), file_template)

    def flatten_detector_counts(self, **kwargs):
        """
        Interpolate and flatten emission data from loop objects.
        """
        for instr in self.instruments:
            self.logger.info('Flattening counts for {}'.format(instr.name))
            with h5py.File(instr.counts_file, 'a', driver=kwargs.get('hdf5_driver',None)) as hf:
                start_index = 0
                for counter, (interp_s, loop) in enumerate(zip(self._interpolated_loop_coordinates, self.field.loops)):
                    self.logger.debug('Flattening counts for {}'.format(loop.name))
                    # LOS velocity
                    los_velocity = np.dot(loop.velocity_xyz, self.line_of_sight)
                    dset = hf['los_velocity']
                    instr.interpolate_and_store(los_velocity, loop, interp_s, dset, start_index)
                    # Electron temperature
                    dset = hf['electron_temperature']
                    instr.interpolate_and_store(loop.electron_temperature, loop, interp_s, dset, start_index)
                    # Ion temperature
                    dset = hf['ion_temperature']
                    instr.interpolate_and_store(loop.ion_temperature, loop, interp_s, dset, start_index)
                    # Average density
                    dset = hf['density']
                    instr.interpolate_and_store(loop.density, loop, interp_s, dset, start_index)
                    # Counts/emission
                    instr.flatten(loop, interp_s, hf, start_index)
                    start_index += len(interp_s)

    def bin_detector_counts(self, savedir):
        """
        Assemble instrument data products and print to FITS file.
        """
        fn_template = os.path.join(savedir, '{instr}', '{channel}', 'map_t{i_time:06d}.fits')
        for instr in self.instruments:
            self.logger.info('Building data products for {}'.format(instr.name))
            # make coordinates histogram for normalization, only need them in 2D
            hist_coordinates, _ = np.histogramdd(self.total_coordinates.value[:,:2],
                                                 bins=[instr.bins.x, instr.bins.y],
                                                 range=[instr.bin_range.x, instr.bin_range.y])
            with h5py.File(instr.counts_file, 'r') as hf:
                reference_time = np.array(hf['time'])*u.Unit(hf['time'].attrs['units'])
                # produce map for each timestep
                for time in instr.observing_time:
                    try:
                        i = np.where(reference_time == time)[0][0]
                    except IndexError:
                        self.logger.exception('{} {} is not a valid observing time for {}'.format(time.value, time.unit.to_string(), instr.name))
                    self.logger.debug('Building data products at time {t:.3f} {u}'.format(t=time.value, u=time.unit))
                    # ion temperature map
                    hist, _ = np.histogramdd(self.total_coordinates.value[:,:2],
                                             bins=[instr.bins.x, instr.bins.y],
                                             range=[instr.bin_range.x, instr.bin_range.y],
                                             weights=np.array(hf['ion_temperature'][i,:]))
                    hist /= np.where(hist_coordinates == 0, 1, hist_coordinates)
                    ion_temperature = hist.T*u.Unit(hf['ion_temperature'].attrs['units'])
                    # LOS velocity map
                    hist, _ = np.histogramdd(self.total_coordinates.value[:,:2],
                                             bins=[instr.bins.x, instr.bins.y],
                                             range=[instr.bin_range.x, instr.bin_range.y],
                                             weights=np.array(hf['los_velocity'][i,:]))
                    hist /= np.where(hist_coordinates == 0, 1, hist_coordinates)
                    los_velocity = hist.T*u.Unit(hf['los_velocity'].attrs['units'])
                    for channel in instr.channels:
                        dummy_dir = os.path.dirname(fn_template.format(instr=instr.name,channel=channel['name'],
                                                                       i_time=0))
                        if not os.path.exists(dummy_dir):
                            os.makedirs(dummy_dir)
                        # setup fits header
                        header = instr.make_fits_header(self.field, channel)
                        header['tunit'] = time.unit.to_string()
                        header['t_obs'] = time.value
                        # combine lines for given channel and return SunPy Map
                        tmp_map = instr.detect(hf, channel, i, header, ion_temperature, los_velocity)
                        # crop to desired region and save
                        if instr.observing_area is not None:
                            tmp_map = tmp_map.crop(instr.observing_area)
                        tmp_map.save(fn_template.format(instr=instr.name, channel=channel['name'],
                                                        i_time=i))

    @u.quantity_input(time=u.s)
    def make_los_velocity_map(self, time, instr, **kwargs):
        """
        Return map of LOS velocity at a given time for a given instrument resolution.
        """
        plot_settings = {
            'cmap': plt.get_cmap('bwr'),
            'norm': matplotlib.colors.SymLogNorm(10, vmin=-1e8, vmax=1e8)
        }
        if 'plot_settings' in kwargs:
            plot_settings.update(kwargs.get('plot_settings'))

        hist_coordinates, _ = np.histogramdd(self.total_coordinates.value[:,:2],
                                             bins=[instr.bins.x, instr.bins.y],
                                             range=[instr.bin_range.x, instr.bin_range.y])
        with h5py.File(instr.counts_file, 'r') as hf:
            try:
                i_time = np.where(np.array(hf['time'])*u.Unit(hf['time'].attrs['units']) == time)[0][0]
            except IndexError:
                self.logger.exception('{} is not a valid time in observing time for {}'.format(time, instr.name))
            tmp = np.array(hf['los_velocity'][i_time,:])
            units = u.Unit(hf['los_velocity'].attrs['units'])
        hist, _ = np.histogramdd(self.total_coordinates.value[:,:2],
                                 bins=[instr.bins.x, instr.bins.y],
                                 range=[instr.bin_range.x, instr.bin_range.y],
                                 weights=tmp)
        hist /= np.where(hist_coordinates == 0, 1, hist_coordinates)
        meta = instr.make_fits_header(self.field, instr.channels[0])
        del meta['wavelnth']
        del meta['waveunit']
        meta['bunit'] = units.to_string()
        meta['detector'] = 'LOS Velocity'
        meta['comment'] = 'LOS velocity calculated by synthesizAR'
        tmp_map = sunpy.map.GenericMap(hist.T, meta)
        tmp_map.plot_settings.update(plot_settings)

        return tmp_map

    @u.quantity_input(time=u.s)
    def make_temperature_map(self, time, instr, **kwargs):
        """
        Return map of column-averaged electron temperature at a given time for a given instrument resolution.
        """
        plot_settings = {'cmap': plt.get_cmap('inferno')}
        if 'plot_settings' in kwargs:
            plot_settings.update(kwargs.get('plot_settings'))

        hist_coordinates, _ = np.histogramdd(self.total_coordinates.value[:,:2],
                                             bins=[instr.bins.x, instr.bins.y],
                                             range=[instr.bin_range.x, instr.bin_range.y])
        with h5py.File(instr.counts_file, 'r') as hf:
            try:
                i_time = np.where(np.array(hf['time'])*u.Unit(hf['time'].attrs['units']) == time)[0][0]
            except IndexError:
                self.logger.exception('{} is not a valid time in observing time for {}'.format(time, instr.name))
            tmp = np.array(hf['electron_temperature'][i_time,:])
            units = u.Unit(hf['electron_temperature'].attrs['units'])
        hist, _ = np.histogramdd(self.total_coordinates.value[:,:2],
                                 bins=[instr.bins.x, instr.bins.y],
                                 range=[instr.bin_range.x, instr.bin_range.y],
                                 weights=tmp)
        hist /= np.where(hist_coordinates == 0, 1, hist_coordinates)
        meta = instr.make_fits_header(self.field, instr.channels[0])
        del meta['wavelnth']
        del meta['waveunit']
        meta['bunit'] = units.to_string()
        meta['detector'] = 'Electron Temperature'
        meta['comment'] = 'Column-averaged electron temperature calculated by synthesizAR'
        tmp_map = sunpy.map.GenericMap(hist.T, meta)
        tmp_map.plot_settings.update(plot_settings)

        return tmp_map

    @u.quantity_input(time=u.s)
    def make_emission_measure_map(self, time, instr, temperature_bin_edges=None, **kwargs):
        """
        Return a cube of maps showing the true emission meausure in each pixel
        as a function of electron temperature.
        """
        plot_settings = {'cmap': matplotlib.cm.get_cmap('magma'),
                         'norm': matplotlib.colors.SymLogNorm(1, vmin=1e25, vmax=1e29)}
        if 'plot_settings' in kwargs:
            plot_settings.update(kwargs.get('plot_settings'))

        # read unbinned temperature and density
        with h5py.File(instr.counts_file, 'r') as hf:
            try:
                i_time = np.where(np.array(hf['time'])*u.Unit(hf['time'].attrs['units']) == time)[0][0]
            except IndexError:
                self.logger.exception('{} is not a valid time in observing time for {}'.format(time, instr.name))
            unbinned_temperature = np.array(hf['electron_temperature'][i_time,:])
            temperature_unit = u.Unit(hf['electron_temperature'].attrs['units'])
            unbinned_density = np.array(hf['density'][i_time,:])
            density_unit = u.Unit(hf['density'].attrs['units'])

        # setup bin edges and weights
        if temperature_bin_edges is None:
            temperature_bin_edges = 10.**(np.arange(5.5, 7.5, 0.1))*u.K
        x_bin_edges = np.diff(instr.bin_range.x)/instr.bins.x*np.arange(instr.bins.x+1) + instr.bin_range.x[0]
        y_bin_edges = np.diff(instr.bin_range.y)/instr.bins.y*np.arange(instr.bins.y+1) + instr.bin_range.y[0]
        z_bin_edges = np.diff(instr.bin_range.z)/instr.bins.z*np.arange(instr.bins.z+1) + instr.bin_range.z[0]
        z_bin_indices = np.digitize(self.total_coordinates.value[:,2], z_bin_edges, right=True)
        dh = np.diff(z_bin_edges)[z_bin_indices - 1]
        emission_measure_weights = (unbinned_density**2)*dh
        # bin in x,y,T space with emission measure weights
        xyT_coordinates = np.append(self.total_coordinates.value[:,:2], 
                                    unbinned_temperature[:,np.newaxis], axis=1)
        hist, _ = np.histogramdd(xyT_coordinates, bins=[x_bin_edges, y_bin_edges, temperature_bin_edges.value], 
                                 weights=emission_measure_weights)

        meta_base = instr.make_fits_header(self.field, instr.channels[0])
        del meta_base['wavelnth']
        del meta_base['waveunit']
        meta_base['detector'] = r'$\mathrm{EM}(T)$'
        meta_base['comment'] = 'LOS Emission Measure distribution'
        data = np.transpose(hist, (1,0,2))*density_unit*density_unit*self.total_coordinates.unit

        return EMCube(data, meta_base, temperature_bin_edges, plot_settings=plot_settings)
