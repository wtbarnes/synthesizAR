"""
Base class for instrument objects.
"""

from collections import namedtuple

import numpy as np

Pair = namedtuple('Pair','x y')


class InstrumentBase(object):
    """
    Base class for instruments. Need to at least implement a detect() method that is used by the
    `Observer` class to get the detector counts.
    """


    def detect(self,loop,channel):
        """
        Converts emissivity for a particular transition to counts per detector channel. When writing
        a new instrument class, this method should be overridden.
        """
        raise NotImplementedError('No detect method implemented.')


    def make_fits_header(self,field,channel):
        """
        Build up FITS header with relevant instrument information.
        """
        update_entries = ['crval1','crval2','cunit1',
                          'cunit2','crlt_obs','ctype1','ctype2','date-obs',
                          'dsun_obs','rsun_obs']
        fits_header = self.fits_template.copy()
        for entry in update_entries:
            fits_header[entry] = field.clipped_hmi_map.meta[entry]
        fits_header['cdelt1'] = self.resolution.x.value
        fits_header['cdelt2'] = self.resolution.y.value
        fits_header['crpix1'] = (self.bins.x + 1.0)/2.0
        fits_header['crpix2'] = (self.bins.y + 1.0)/2.0
        fits_header['instrume'] = channel['instrument_label']
        fits_header['wavelnth'] = int(channel['wavelength'].value)

        return fits_header

    def make_detector_array(self,field):
        """
        Construct bins based on desired observing area.
        """
        delta_x = np.fabs(field.clipped_hmi_map.xrange[1] -
                          field.clipped_hmi_map.xrange[0])
        delta_y = np.fabs(field.clipped_hmi_map.yrange[1] -
                          field.clipped_hmi_map.yrange[0])
        self.bins = Pair(int(np.ceil(delta_x/self.resolution.x).value),
                         int(np.ceil(delta_y/self.resolution.y).value))
        self.bin_range = Pair(
            field._convert_angle_to_length(field.clipped_hmi_map.xrange).value,
            field._convert_angle_to_length(field.clipped_hmi_map.yrange).value)
