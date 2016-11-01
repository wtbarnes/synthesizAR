"""
Create data products loop simulations
"""

import os
import logging

import numpy as np
import scipy.interpolate
import astropy.units as u
import h5py


class Observer(object):
    """
    Class for assembling AR from loops and creating data products from 2D
    projections.

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
        """
        Constructor
        """
        self.logger = logging.getLogger(name=type(self).__name__)
        self.field = field
        self.instrument = instruments
        if ds is None:
            ds = 0.1*np.min([min(instr.resolution_x.value,
                    instr.resolution_y.value) for instr in self.instruments])*self.instruments[0].resolution.x.unit
        self.ds = self.field._convert_angle_to_length(ds)


    def build_detector_files(self,savedir):
        """
        Create files to store interpolated emissivity results before binning.
        """
        file_template = os.path.join(savedir,'{detector}_counts.h5')
        interpolated_points = sum([int(np.ceil(loop.full_length/self.ds)) for loop in self.field.loops])
        for instr in self.instruments:
            instr.counts_file = file_template.format(instr.name)
            with h5py.File(file_template.format(instr.name),'w') as hf:
                for c in instr.channels:
                    hf.create_dataset(c['wavelength'].value,
                                (len(instr.observing_time),interpolated_points))


    def calculate_detector_counts(self):
        """
        Calculate counts for each channel in the detector using the emissivity
        for every wavelength that we computed previously.
        """
        # initialize offset and list for coordinates
        start_index = 0
        total_coordinates = []
        # iterate over all loops in the field
        for loop in self.field.loops:
            self.logger.debug(
                            'Calculating counts for loop {}'.format(loop.name))
            n_interp = int(np.ceil(loop.full_length/self.ds))
            interpolated_s = np.linspace(loop.field_aligned_coordinate.value[0],
                                        loop.field_aligned_coordinate.value[-1],
                                        n_interp)
            nots,_ = interpolate.splprep(loop.coordinates.value.T)
            _tmp = interpolate.splev(np.linspace(0,1,n_interp),nots)
            total_coordinates += [(x,y,z) for x,y,z in zip(_tmp[0],
                                                           _tmp[1],
                                                           _tmp[2])]
            #iterate over detectors
            for instr in self.instruments:
                self.logger.debug(
                    'Calculating counts for instrument {}'.format(instr.name))
                with h5py.File(instr.counts_file,'a') as hf:
                    #iterate over channels
                    for channel in instr.channels:
                        self.logger.debug('Calculating counts for channel {}'.format(channel['wavelength']))
                        counts = instr.detect(loop,channel)
                        #interpolate in s and t
                        f_s = interpolate.interp1d(
                                        loop.field_aligned_coordinate.value,
                                        counts.value,axis=1)
                        interpolated_counts = interpolate.interp1d(loop.time.value, f_s(interpolated_s), axis=0)(instr.observing_time)
                        #save to file
                        dset = hf[channel['wavelength'].value]
                        dset[:,start_index:(start_index+n_interp)] = interpolated_counts

            #increment offset
            start_index += n_interp
