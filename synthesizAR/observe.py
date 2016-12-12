"""
Create data products loop simulations
"""

import os
import logging

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


    def __init__(self,field,instruments,ds=None):
        self.logger = logging.getLogger(name=type(self).__name__)
        self.field = field
        self.instruments = instruments
        if ds is None:
            ds = 0.1*np.min([min(instr.resolution.x.value,nstr.resolution.y.value) \
            for instr in self.instruments])*self.instruments[0].resolution.x.unit
        self.ds = self.field._convert_angle_to_length(ds)

    def build_detector_files(self,savedir):
        """
        Create files to store interpolated counts before binning.
        """
        file_template = os.path.join(savedir,'{}_counts.h5')
        interpolated_points = sum([int(np.ceil(loop.full_length/self.ds)) \
        for loop in self.field.loops])

        for instr in self.instruments:
            instr.counts_file = file_template.format(instr.name)
            self.logger.info('Creating instrument file {}'.format(instr.counts_file))
            with h5py.File(instr.counts_file,'w') as hf:
                for channel in instr.channels:
                    hf.create_dataset(channel['name'],(len(instr.observing_time),
                                        interpolated_points))

    def calculate_detector_counts(self):
        """
        Calculate counts for each channel of each detector. Counts are interpolated to the
        desired spatial and temporal resolution.
        """
        # initialize offset and list for coordinates
        start_index = 0
        self.total_coordinates = []
        # iterate over all loops in the field
        for loop in self.field.loops:
            self.logger.info('Calculating counts for loop {}'.format(loop.name))
            n_interp = int(np.ceil(loop.full_length/self.ds))
            interpolated_s = np.linspace(loop.field_aligned_coordinate.value[0],
                                        loop.field_aligned_coordinate.value[-1],n_interp)
            nots,_ = splprep(loop.coordinates.value.T)
            _tmp = splev(np.linspace(0,1,n_interp),nots)
            self.total_coordinates += [(x,y,z) for x,y,z in zip(_tmp[0],_tmp[1],_tmp[2])]
            #iterate over detectors
            for instr in self.instruments:
                self.logger.debug('Calculating counts for instrument {}'.format(instr.name))
                with h5py.File(instr.counts_file,'a') as hf:
                    #iterate over channels
                    for channel in instr.channels:
                        self.logger.debug(
                                        'Calculating counts for channel{}'.format(channel['name']))
                        counts = instr.detect(loop,channel)
                        #interpolate in s and t
                        f_s = interp1d(loop.field_aligned_coordinate.value,counts.value,axis=1)
                        interpolated_counts = interp1d(loop.time.value,f_s(interpolated_s),
                                                        axis=0)(instr.observing_time)
                        #save to file
                        dset = hf[channel['name']]
                        if 'units' not in dset.attrs:
                            dset.attrs['units'] = counts.unit.to_string()
                        dset[:,start_index:(start_index+n_interp)] = interpolated_counts
            #increment offset
            start_index += n_interp
        self.total_coordinates = np.array(self.total_coordinates)*loop.coordinates.unit

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

    def bin_detector_counts(self,savedir,apply_psf=False):
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
