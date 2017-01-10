"""
Create data products loop simulations
"""

import os
import logging
import pickle

import numpy as np
from scipy.interpolate import splev,splprep,interp1d
import scipy.ndimage
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
                    dset = hf['los_velocity/flat_counts']
                    instr.interpolate_and_store(los_velocity,loop,interp_s,dset,start_index)
                    # Average temperature
                    dset = hf['average_temperature/flat_counts']
                    instr.interpolate_and_store(loop.temperature,loop,interp_s,dset,start_index)
                    # Counts/emission
                    instr.flatten(loop,interp_s,hf,start_index)
                    start_index += len(interp_s)

    def bin_detector_counts(self):
        """
        Bin all channels or lines into a 3D histogram and project onto x-y plane
        """
        for instr in self.instruments:
            self.logger.info('Binning counts for {}'.format(instr.name))
            bins_z,bin_range_z = self._make_z_bins(instr)
            # make coordinates histogram for normalization
            hist_coordinates,_ = np.histogramdd(self.total_coordinates.value,
                                        bins=[instr.bins.x,instr.bins.y,bins_z],
                                        range=[instr.bin_range.x,instr.bin_range.y,bin_range_z])
            with h5py.File(instr.counts_file,'a') as hf:
                for group in hf:
                    self.logger.info('Binning counts for {}'.format(group))
                    dset_flat = hf['{}/flat_counts'.format(group)]
                    dset_map = hf['{}/map'.format(group)]
                    if group=='los_velocity' or group=='average_temperature':
                        dset_map.attrs['units'] = dset_flat.attrs['units']
                    else:
                        dset_map.attrs['units'] = (u.Unit(dset_flat.attrs['units'])*self.total_coordinates.unit).to_string()
                    for i,time in enumerate(instr.observing_time.value):
                        self.logger.debug('Binning counts for time = {t:.3f} {u}'.format(t=time,instr.observing_time.unit))
                        tmp = np.array(dset_flat[i,:])
                        hist,edges = np.histogramdd(self.total_coordinates.value,
                                            bins=[instr.bins.x,instr.bins.y,bins_z],
                                            range=[instr.bin_range.x,instr.bin_range.y,bin_range_z],
                                            weights=tmp)
                        if group=='los_velocity' or group=='average_temperature':
                            hist /= np.where(hist_coordinates==0,1,hist_coordinates)
                            projection = np.dot(hist,np.diff(edges[2])).T/np.sum(np.diff(edges[2]))
                        else:
                            projection = np.dot(hist,np.diff(edges[2])).T
                        dset_map[:,:,i] = projection

    def __calculate_detector_counts(self):
        """
        Calculate counts for each channel of each detector. Counts are interpolated to the
        desired spatial and temporal resolution.
        """
        #rebuild detector files
        for instr in self.instruments:
            self.logger.info('Calculating counts for {}'.format(instr.name))
            with h5py.File(instr.counts_file,'a') as hf:
                for channel in instr.channels:
                    self.logger.info('Calculating counts for channel {}'.format(channel['name']))
                    dset = hf[channel['name']]
                    start_index = 0
                    for interp_s,loop in zip(self._interpolated_loop_coordinates,self.field.loops):
                        self.logger.debug('Calculating counts for {}'.format(loop.name))
                        counts = instr.detect(loop,channel)
                        #interpolate in s and t
                        f_s = interp1d(loop.field_aligned_coordinate.value,counts.value,
                                        axis=1,kind='linear')
                        interpolated_counts = interp1d(loop.time.value,f_s(interp_s),
                                                        axis=0,kind='linear')(instr.observing_time)
                        dset[:,start_index:(start_index+len(interp_s))] = interpolated_counts
                        if 'units' not in dset.attrs:
                            dset.attrs['units'] = counts.unit.to_string()
                        start_index += len(interp_s)

    def _make_z_bins(self,instr):
        """
        Make z bins and ranges. The bin width isn't all that important since
        the final data product will be integrated along the LOS.
        """
        min_z = min(self.field.extrapolated_3d_field.domain_left_edge[2].value,
                self.total_coordinates[:,2].min().value)
        max_z = max(self.field.extrapolated_3d_field.domain_right_edge[2].value,
                self.total_coordinates[:,2].max().value)
        delta_z = self.field._convert_angle_to_length(
                max(instr.resolution.x,instr.resolution.y)).value
        bins_z = int(np.ceil(np.fabs(max_z-min_z)/delta_z))
        bin_range_z = [min_z,max_z]

        return bins_z,bin_range_z

    def make_data_products(self,savedir):
        """
        Assemble instrument data products and print to FITS file.
        """
        fn_template = os.path.join(savedir,'{instr}','{channel}','map_t{time:06d}.fits')
        for instr in self.instruments:
            self.logger.info('Building data products for {}'.format(instr.name))
            with h5py.File(instr.counts_file,'r') as hf:
                for channel in instr.channels:
                    self.logger.info('Building data products for channel {}'.format(channel['name']))
                    dummy_dir = os.path.dirname(fn_template.format(instr=instr.name,
                                                                    channel=channel['name'],
                                                                    time=0))
                    if not os.path.exists(dummy_dir):
                        os.makedirs(dummy_dir)
                    #setup fits header
                    header = instr.make_fits_header(self.field,channel)
                    header['tunit'] = instr.observing_time.unit.to_string()
                    #produce map for each timestep
                    for i,time in enumerate(instr.observing_time.value):
                        self.logger.debug('Building data products at time {t:.3f} {u}'.format(t=time,u=instr.observing_time.unit))
                        #combine lines for given channel
                        data = instr.detect(hf,channel,i,header)
                        #make SunPy map and save as FITS
                        header['t_obs'] = time
                        tmp_map = sunpy.map.Map(data,header)
                        #crop to desired region and save
                        if instr.observing_area is not None:
                            tmp_map = tmp_map.crop(instr.observing_area)
                        tmp_map.save(fn_template.format(instr=instr.name,channel=channel['name'],
                                                        time=i))

    def __bin_detector_counts(self,savedir,apply_psf=False):
        """
        Bin the counts into the detector array, project it down to 2 dimensions,
        and save it to a FITS file.
        """
        if type(apply_psf) is bool:
            apply_psf = len(self.instruments)*[apply_psf]
        fn_template = os.path.join(savedir,'{instr}','{channel}','map_t{time:06d}.fits')
        for instr in self.instruments:
            self.logger.info('Building maps for {}'.format(instr.name))
            #create instrument array bins
            bins_z,bin_range_z = self._make_z_bins(instr)
            instr.make_detector_array(self.field)
            with h5py.File(instr.counts_file,'r') as hf:
                for channel in instr.channels:
                    self.logger.info('Building maps for channel {}'.format(channel['name']))
                    dummy_dir = os.path.dirname(fn_template.format(instr=instr.name,
                                                                    channel=channel['name'],
                                                                    time=0))
                    if not os.path.exists(dummy_dir):
                        os.makedirs(dummy_dir)
                    dset = hf[channel['name']]
                    #setup fits header
                    header = instr.make_fits_header(self.field,channel)
                    header['tunit'] = instr.observing_time.unit.to_string()
                    header['bunit'] = (u.Unit(dset.attrs['units'])*self.total_coordinates.unit).to_string()
                    for i,time in enumerate(instr.observing_time.value):
                        self.logger.debug('Building map at t={}'.format(time))
                        #slice at particular time
                        _tmp = np.array(dset[i,:])
                        #bin counts into 3D histogram
                        hist,edges = np.histogramdd(
                                        self.total_coordinates.value,
                                        bins=[instr.bins.x,instr.bins.y,bins_z],
                                        range=[instr.bin_range.x,instr.bin_range.y,bin_range_z],
                                        weights=_tmp)
                        #project down to x-y plane
                        projection = np.dot(hist,np.diff(edges[2])).T
                        if apply_psf[self.instruments.index(instr)]:
                            projection = scipy.ndimage.filters.gaussian_filter(projection,
                                                                    channel['gaussian_width'].value)
                        header['t_obs'] = time
                        tmp_map = sunpy.map.Map(projection,header)
                        #crop to desired region and save
                        if instr.observing_area is not None:
                            tmp_map = tmp_map.crop(instr.observing_area)
                        tmp_map.save(fn_template.format(instr=instr.name,channel=channel['name'],
                                                        time=i))
