"""
Create data products loop simulations
"""

import os
import logging
import pickle

import numpy as np
from scipy.interpolate import splev,splprep,interp1d
import scipy.ndimage
import matplotlib.colors
import seaborn.apionly as sns
import astropy.units as u
import sunpy.map
import h5py


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


    def __init__(self,field,instruments,ds=None,line_of_sight=(0,0,-1)):
        self.logger = logging.getLogger(name=type(self).__name__)
        self.field = field
        self.instruments = instruments
        self._channels_setup()
        self.line_of_sight = line_of_sight
        if ds is None:
            ds = 0.1*np.min([min(instr.resolution.x.value,instr.resolution.y.value) \
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
                    for wvl in self.field.loops[0].wavelengths:
                        if channel['wavelength_range'][0] <= wvl <= channel['wavelength_range'][-1]:
                            channel['model_wavelengths'].append(wvl)
                    channel['model_wavelengths'] = u.Quantity(channel['model_wavelengths'])

    def _interpolate_loops(self,ds):
        """
        Interpolate all loops to a resolution (`ds`) below the minimum bin width of all of the
        instruments. This ensures that the image isn't 'patchy' when it is binned.
        """
        # FIXME: memory requirements for this list will grow with number of loops, consider saving it to the instrument files, both the interpolated s and total_coordinates
        self.total_coordinates = []
        self._interpolated_loop_coordinates = []
        for loop in self.field.loops:
            self.logger.debug('Interpolating loop {}'.format(loop.name))
            n_interp = int(np.ceil(loop.full_length/ds))
            interpolated_s = np.linspace(loop.field_aligned_coordinate.value[0],
                                        loop.field_aligned_coordinate.value[-1],n_interp)
            self._interpolated_loop_coordinates.append(interpolated_s)
            nots,_ = splprep(loop.coordinates.value.T)
            _tmp = splev(np.linspace(0,1,n_interp),nots)
            self.total_coordinates += [(x,y,z) for x,y,z in zip(_tmp[0],_tmp[1],_tmp[2])]

        self.total_coordinates = np.array(self.total_coordinates)*loop.coordinates.unit

    def build_detector_files(self,savedir):
        """
        Create files to store interpolated counts before binning.
        """
        file_template = os.path.join(savedir,'{}_counts.h5')
        for instr in self.instruments:
            instr.total_coordinates = self.total_coordinates
            instr.make_detector_array(self.field)
            instr.build_detector_file(self.field,len(self.total_coordinates),file_template)

    def flatten_detector_counts(self):
        """
        Interpolate and flatten emission data from loop objects.
        """
        for instr in self.instruments:
            self.logger.info('Flattening counts for {}'.format(instr.name))
            with h5py.File(instr.counts_file,'a') as hf:
                start_index = 0
                for interp_s,loop in zip(self._interpolated_loop_coordinates,self.field.loops):
                    self.logger.debug('Flattening counts for {}'.format(loop.name))
                    # LOS velocity
                    los_velocity = np.dot(loop.velocity_xyz,self.line_of_sight)
                    dset = hf['los_velocity']
                    instr.interpolate_and_store(los_velocity,loop,interp_s,dset,start_index)
                    # Average temperature
                    dset = hf['average_temperature']
                    instr.interpolate_and_store(loop.temperature,loop,interp_s,dset,start_index)
                    # Counts/emission
                    instr.flatten(loop,interp_s,hf,start_index)
                    start_index += len(interp_s)

    def bin_detector_counts(self,savedir):
        """
        Assemble instrument data products and print to FITS file.
        """
        fn_template = os.path.join(savedir,'{instr}','{channel}','map_t{time:06d}.fits')
        for instr in self.instruments:
            self.logger.info('Building data products for {}'.format(instr.name))
            # make coordinates histogram for normalization
            hist_coordinates,_ = np.histogramdd(self.total_coordinates.value,
                                    bins=[instr.bins.x,instr.bins.y,instr.bins.z],
                                    range=[instr.bin_range.x,instr.bin_range.y,instr.bin_range.z])
            with h5py.File(instr.counts_file,'r') as hf:
                #produce map for each timestep
                for i,time in enumerate(instr.observing_time.value):
                    self.logger.debug('Building data products at time {t:.3f} {u}'.format(t=time,u=instr.observing_time.unit))
                    # temperature map
                    hist,edges = np.histogramdd(self.total_coordinates.value,
                                    bins=[instr.bins.x,instr.bins.y,instr.bins.z],
                                    range=[instr.bin_range.x,instr.bin_range.y,instr.bin_range.z],
                                    weights=np.array(hf['average_temperature'][i,:]))
                    hist /= np.where(hist_coordinates==0,1,hist_coordinates)
                    average_temperature = np.dot(hist,np.diff(edges[2])).T/np.sum(np.diff(edges[2]))
                    average_temperature = average_temperature\
                                            *u.Unit(hf['average_temperature'].attrs['units'])
                    # LOS velocity map
                    hist,edges = np.histogramdd(self.total_coordinates.value,
                                    bins=[instr.bins.x,instr.bins.y,instr.bins.z],
                                    range=[instr.bin_range.x,instr.bin_range.y,instr.bin_range.z],
                                    weights=np.array(hf['los_velocity'][i,:]))
                    hist /= np.where(hist_coordinates==0,1,hist_coordinates)
                    los_velocity = np.dot(hist,np.diff(edges[2])).T/np.sum(np.diff(edges[2]))
                    los_velocity = los_velocity*u.Unit(hf['los_velocity'].attrs['units'])
                    for channel in instr.channels:
                        dummy_dir = os.path.dirname(fn_template.format(instr=instr.name,
                                                                        channel=channel['name'],
                                                                        time=0))
                        if not os.path.exists(dummy_dir):
                            os.makedirs(dummy_dir)
                        #setup fits header
                        header = instr.make_fits_header(self.field,channel)
                        header['tunit'] = instr.observing_time.unit.to_string()
                        header['t_obs'] = time
                        #combine lines for given channel and return SunPy Map
                        tmp_map = instr.detect(hf,channel,i,header,average_temperature,los_velocity)
                        #crop to desired region and save
                        if instr.observing_area is not None:
                            tmp_map = tmp_map.crop(instr.observing_area)
                        tmp_map.save(fn_template.format(instr=instr.name,channel=channel['name'],
                                                        time=i))

    def make_los_velocity_map(self,time,instr,**kwargs):
        """
        Return map of LOS velocity at a given time for a given instrument resolution.
        """
        plot_settings = {
            'cmap':matplotlib.colors.ListedColormap(sns.color_palette('coolwarm',n_colors=1000)),
            'norm':matplotlib.colors.SymLogNorm(10,vmin=-1e8,vmax=1e8)
        }
        if 'plot_settings' in kwargs:
            plot_settings.update(kwargs.get('plot_settings'))

        i_time = np.where(instr.observing_time==time)[0]
        if len(i_time)==0:
            raise ValueError('{} is not a valid time in observing time for {}'.format(time,
                                                                                     instr.name))
        else:
            i_time = i_time[0]

        hist_coordinates,_ = np.histogramdd(self.total_coordinates.value[:,:2],
                                bins=[instr.bins.x,instr.bins.y],#,instr.bins.z],
                                range=[instr.bin_range.x,instr.bin_range.y],#instr.bin_range.z]
                                )
        with h5py.File(instr.counts_file,'r') as hf:
            tmp = np.array(hf['los_velocity'][i_time,:])
            units = u.Unit(hf['los_velocity'].attrs['units'])
        hist,edges = np.histogramdd(self.total_coordinates.value[:,:2],
                        bins=[instr.bins.x,instr.bins.y],#,instr.bins.z],
                        range=[instr.bin_range.x,instr.bin_range.y],#,instr.bin_range.z],
                        weights=tmp)
        hist /= np.where(hist_coordinates==0,1,hist_coordinates)
        los_velocity = hist#np.dot(hist,np.diff(edges[2])).T/np.sum(np.diff(edges[2]))
        meta = instr.make_fits_header(self.field,instr.channels[0])
        del meta['wavelnth']
        del meta['waveunit']
        meta['bunit'] = units.to_string()
        meta['detector'] = 'LOS Velocity'
        meta['comment'] = 'LOS velocity calculated by synthesizAR'
        tmp_map = sunpy.map.GenericMap(los_velocity,meta)
        tmp_map.plot_settings.update(plot_settings)

        return tmp_map

    def make_temperature_map(self,time,instr):
        """
        Return map of average temperature at a given time for a given instrument resolution.
        """
        plot_settings = {'cmap':sns.cubehelix_palette(reverse=True,rot=.4,as_cmap=True)}
        if 'plot_settings' in kwargs:
            plot_settings.update(kwargs.get('plot_settings'))

        i_time = np.where(instr.observing_time==time)[0]
        if len(i_time)==0:
            raise ValueError('{} is not a valid time in observing time for {}'.format(time,
                                                                                     instr.name))
        else:
            i_time = i_time[0]

        hist_coordinates,_ = np.histogramdd(self.total_coordinates.value[:,:2],
                                bins=[instr.bins.x,instr.bins.y],#,instr.bins.z],
                                range=[instr.bin_range.x,instr.bin_range.y],#instr.bin_range.z]
                                )
        with h5py.File(instr.counts_file,'r') as hf:
            tmp = np.array(hf['average_temperature'][i_time,:])
            units = u.Unit(hf['average_temperature'].attrs['units'])
        hist,edges = np.histogramdd(self.total_coordinates.value[:,:2],
                        bins=[instr.bins.x,instr.bins.y],#,instr.bins.z],
                        range=[instr.bin_range.x,instr.bin_range.y],#,instr.bin_range.z],
                        weights=tmp)
        hist /= np.where(hist_coordinates==0,1,hist_coordinates)
        temperature = hist#np.dot(hist,np.diff(edges[2])).T/np.sum(np.diff(edges[2]))
        meta = instr.make_fits_header(self.field,instr.channels[0])
        del meta['wavelnth']
        del meta['waveunit']
        meta['bunit'] = units.to_string()
        meta['detector'] = 'Temperature'
        meta['comment'] = 'Average temperature calculated by synthesizAR'
        tmp_map = sunpy.map.GenericMap(temperature,meta)
        tmp_map.plot_settings.update(plot_settings)

        return tmp_map
