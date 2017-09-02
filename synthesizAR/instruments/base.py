"""
Base class for instrument objects.
"""

import os
import logging
from collections import namedtuple

import numpy as np
import dask
from scipy.interpolate import interp1d
import astropy.units as u
import h5py
import dask.delayed

Pair = namedtuple('Pair', 'x y z')


class InstrumentBase(object):
    """
    Base class for instruments. Need to at least implement a detect() method that is used by the
    `Observer` class to get the detector counts.
    """

    @u.quantity_input(observing_time=u.s)
    def __init__(self, observing_time, observing_area=None):
        self.logger = logging.getLogger(name=type(self).__name__)
        self.observing_time = np.arange(observing_time[0].to(u.s).value,
                                        observing_time[1].to(u.s).value,
                                        self.cadence.value)*u.s
        self.observing_area = observing_area

    def detect(self, *args, **kwargs):
        """
        Converts emissivity for a particular transition to counts per detector channel. When writing
        a new instrument class, this method should be overridden.
        """
        raise NotImplementedError('No detect method implemented.')

    def build_detector_file(self, file_template, chunks, *args, **kwargs):
        """
        Allocate space for counts data.
        """
        dset_shape = self.observing_time.shape+(len(self.total_coordinates),)
        dset_names = ['density', 'electron_temperature', 'ion_temperature', 'los_velocity']
        dset_names += kwargs.get('additional_fields',[])
        self.counts_file = file_template.format(self.name)
        self.tmp_file_template = os.path.join(os.path.dirname(self.counts_file),'tmp_parallel_files',self.name,'{}')
        self.logger.info('Creating instrument file {}'.format(self.counts_file))
        with h5py.File(self.counts_file, 'a') as hf:
            if 'time' not in hf:
                dset = hf.create_dataset('time', data=self.observing_time.value)
                dset.attrs['units'] = self.observing_time.unit.to_string()
            for dn in dset_names:
                if dn not in hf:
                    hf.create_dataset(dn, dset_shape, chunks=chunks)
                if not os.path.exists(self.tmp_file_template.format(dn)) and kwargs.get('make_tmp_files',True):
                    os.makedirs(self.tmp_file_template.format(dn))

    @staticmethod
    @dask.delayed
    def interpolate_and_store(y, loop, interp_t, interp_s, save_path):
        """
        Interpolate in time and space and write to HDF5 file.
        """
        f_s = interp1d(loop.field_aligned_coordinate.value, y.value, axis=1, kind='linear')
        interpolated_y = interp1d(loop.time.value, f_s(interp_s), axis=0, kind='linear',
                                  fill_value='extrapolate')(interp_t.value)
        np.save(save_path, interpolated_y)
        return save_path, y.unit.to_string()

    def make_fits_header(self, field, channel):
        """
        Build up FITS header with relevant instrument information.
        """
        update_entries = ['crval1', 'crval2', 'cunit1',
                          'cunit2', 'crlt_obs', 'ctype1', 'ctype2', 'date-obs',
                          'dsun_obs', 'rsun_obs']
        fits_header = self.fits_template.copy()
        for entry in update_entries:
            fits_header[entry] = field.clipped_hmi_map.meta[entry]
        fits_header['cdelt1'] = self.resolution.x.value
        fits_header['cdelt2'] = self.resolution.y.value
        fits_header['crpix1'] = (self.bins.x + 1.0)/2.0
        fits_header['crpix2'] = (self.bins.y + 1.0)/2.0
        if 'instrume' not in fits_header:
            fits_header['instrume'] = channel['instrument_label']
        fits_header['wavelnth'] = int(channel['wavelength'].value)

        return fits_header

    def make_detector_array(self, field):
        """
        Construct bins based on desired observing area.
        """
        delta_x = np.fabs(field.clipped_hmi_map.xrange[1] -
                          field.clipped_hmi_map.xrange[0])
        delta_y = np.fabs(field.clipped_hmi_map.yrange[1] -
                          field.clipped_hmi_map.yrange[0])
        min_z = min(field.extrapolated_3d_field.domain_left_edge[2].value,
                    self.total_coordinates[:,2].min().value)
        max_z = max(field.extrapolated_3d_field.domain_right_edge[2].value,
                    self.total_coordinates[:,2].max().value)
        delta_z = field._convert_angle_to_length(
                max(self.resolution.x, self.resolution.y)).value
        self.bins = Pair(int(np.ceil(delta_x/self.resolution.x).value),
                         int(np.ceil(delta_y/self.resolution.y).value),
                         int(np.ceil(np.fabs(max_z-min_z)/delta_z)))
        self.bin_range = Pair(field._convert_angle_to_length(field.clipped_hmi_map.xrange).value,
                              field._convert_angle_to_length(field.clipped_hmi_map.yrange).value,
                              np.array([min_z, max_z]))
