"""
Class for the SDO/AIA instrument. Holds information about the cadence and
spatial and spectroscopic resolution.
"""

import os
import sys
import logging
from collections import namedtuple

import numpy as np
import scipy.interpolate
import astropy.units as u
import sunpy.map

Pair = namedtuple('Pair','x y')


class InstrumentSDOAIA(object):
    """
    Atmospheric Imaging Assembly object for observing synthesized active
    region emission.

    Parameters
    ----------
    observing_time : `tuple`
        start and end of observing time
    observing_area : `tuple`
        x and y range of observation
    use_temperature_response_functions : `bool`
        if True, do simple counts calculation
    response_function_file : `str`
        filename containing AIA response functions

    Examples
    --------
    Notes
    -----
    """

    fits_template = sunpy.map.header.MapMeta()
    fits_template['telescop'] = 'SDO/AIA'
    fits_template['detector'] = 'AIA'
    fits_template['waveunit'] = 'angstrom'

    name = 'SDO_AIA'
    channels = [{'wavelength':94*u.angstrom,'telescope_number':4},
                {'wavelength':131*u.angstrom,'telescope_number':1},
                {'wavelength':171*u.angstrom,'telescope_number':3},
                {'wavelength':193*u.angstrom,'telescope_number':2},
                {'wavelength':211*u.angstrom,'telescope_number':2},
                {'wavelength':335*u.angstrom,'telescope_number':1}]
    for channel in channels:
        channel['name'] = str(channel['wavelength'].value).strip('.0')

    cadence = 10.0*u.s
    resolution = Pair(0.600698*u.arcsec/u.pixel,0.600698*u.arcsec/u.pixel)


    def __init__(self, observing_time, observing_area=None,
    use_temperature_response_functions=True,response_function_file=''):
        """
        Constructor
        """
        self.logger = logging.getLogger(name=type(self).__name__)
        self.observing_time = np.arange(
                            observing_time[0].to(u.s).value,
                            observing_time[1].to(u.s).value,
                            self.cadence.value)*u.s
        self.observing_area = observing_area
        self.use_temperature_response_functions = use_temperature_response_functions
        if self.use_temperature_response_functions and response_function_file:
            self._setup_response_functions(response_function_file)


    def detect(self,loop,channel):
        """
        For a given loop object, calculate the counts detected by AIA in a
        particular channel.

        Parameters
        ----------
        loop : loop object
        channel : `dict`

        Returns
        -------
        counts : array-like
        """
        if self.use_temperature_response_functions:
            counts = self._detect_simple(loop,channel)
        else:
            counts = self._detect_full(loop,channel)
        return counts


    def _detect_simple(self,loop,channel):
        """
        Calculate counts using the density and temperature response functions.
        No emissivity model needed.
        """
        response_function = channel['response_interpolator'](np.ravel(loop.temperature))*u.count*u.cm**5/u.s/u.pixel

        return np.reshape(np.ravel(loop.density**2)*response_function,
                          np.shape(loop.density))


    def _detect_full(self,loop,channel):
        """
        Calculate counts use emissivity for a large number of transitions.
        Requires emissivity model.

        Notes
        -----
        This is necessary when taking into account non-equilibrium ionization.
        """
        pass


    def _setup_response_functions(self,filename):
        """
        Setup interpolators from the AIA temperature response functions.

        Notes
        -----
        This should be replaced once the response functions are available in
        SunPy.
        Probably should configure wavelength response function
        interpolators also.
        """
        _tmp = np.loadtxt(filename)
        channel_order = {c:i for c,i in zip([94,131,171,
                                             193,211,335]*u.angstrom,range(6))}
        _tmp_temperature = 10**(_tmp[:,0])
        for i,channel in enumerate(self.channels):
            _tmp_response = _tmp[:,channel_order[channel['wavelength']]+1]
            self.channels[i]['response_interpolator'] = scipy.interpolate.interp1d(_tmp_temperature,_tmp_response)


    def make_fits_header(self,field,channel):
        """
        Build up FITS header with relevant instrument information.
        """
        update_entries = ['crpix1','crpix2','crval1','crval2','cunit1',
                          'cunit2','crlt_obs','ctype1','ctype2','date-obs',
                          'dsun_obs','rsun_obs']
        fits_header = self.fits_template.copy()
        for entry in update_entries:
            fits_header[entry] = field.clipped_hmi_map.meta[entry]
        fits_header['cdelt1'] = self.resolution.x.value
        fits_header['cdelt2'] = self.resolution.y.value
        fits_header['instrume'] = 'AIA_' + channel['telescope_number']
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
        self.bins = [int(np.ceil(delta_x/self.resolution.x)).value,
                     int(np.ceil(delta_y/self.resolution.y)).value]
        self.bin_ranges = [
            field._convert_angle_to_length(field.clipped_hmi_map.xrange).value,
            field._convert_angle_to_length(field.clipped_hmi_map.yrange).value]
