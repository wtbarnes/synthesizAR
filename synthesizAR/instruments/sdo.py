"""
Class for the SDO/AIA instrument. Holds information about the cadence and
spatial and spectroscopic resolution.
"""

import os
import sys
import logging

import numpy as np
from scipy.interpolate import splrep,splev,interp1d
import scipy.ndimage
import astropy.units as u
import sunpy.map

from synthesizAR.instruments import InstrumentBase,Pair


class InstrumentSDOAIA(InstrumentBase):
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
    channels = [
        {'wavelength':94*u.angstrom,'telescope_number':4,
            'gaussian_width':0.951*u.pixel},
        {'wavelength':131*u.angstrom,'telescope_number':1,
            'gaussian_width':1.033*u.pixel},
        {'wavelength':171*u.angstrom,'telescope_number':3,
            'gaussian_width':0.962*u.pixel},
        {'wavelength':193*u.angstrom,'telescope_number':2,
            'gaussian_width':1.512*u.pixel},
        {'wavelength':211*u.angstrom,'telescope_number':2,
            'gaussian_width':1.199*u.pixel},
        {'wavelength':335*u.angstrom,'telescope_number':1,
            'gaussian_width':0.962*u.pixel}]
    for channel in channels:
        channel['name'] = str(channel['wavelength'].value).strip('.0')
        channel['instrument_label'] = '{}_{}'.format(fits_template['detector'],
                                                    channel['telescope_number'])
        #TODO: this should be set once we use the wavelength response function for AIA
        channel['wavelength_range'] = None

    cadence = 10.0*u.s
    resolution = Pair(0.600698*u.arcsec/u.pixel,0.600698*u.arcsec/u.pixel)

    def __init__(self, observing_time, observing_area=None,
    use_temperature_response_functions=True,response_function_file='',apply_psf=True):
        super().__init__(observing_time,observing_area)
        self.apply_psf = apply_psf
        self.use_temperature_response_functions = use_temperature_response_functions
        if self.use_temperature_response_functions and response_function_file:
            self._setup_response_functions(response_function_file)

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
            self.channels[i]['response_spline_nots'] = splrep(_tmp_temperature,_tmp_response)

    def build_detector_file(self,field,num_loop_coordinates,file_format):
        """
        Allocate space for counts data.
        """
        super().build_detector_file(num_loop_coordinates,file_format)
        if self.use_temperature_response_functions:
            with h5py.File(self.counts_file,'a') as hf:
                for channel in self.channels:
                    hf.create_dataset('{}/flat_counts'.format(channel['name']),
                                        (len(self.observing_time),num_loop_coordinates))
                    hf.create_dataset('{}/maps'.format(channel['name']),
                                        (self.bins.y,self.bins.x,len(self.observing_time)))

    def flatten(self,loop,interp_s,hf,start_index):
        """
        Flatten channel counts to HDF5 file
        """
        if self.use_temperature_response_functions:
            for channel in self.channels:
                response_function = splev(np.ravel(loop.temperature),
                                        channel['response_spline_nots'])*u.count*u.cm**5/u.s/u.pixel
                counts = np.reshape(np.ravel(loop.density**2)*response_function,
                                    np.shape(loop.density))
                dset = hf['{}/flat_counts'.format(channel['name'])]
                self.interpolate_and_store(counts,loop,interp_s,dset)
        else:
            raise NotImplementedError('''Full detect function not yet implemented. Set
                                        use_temperature_response_functions to True to use the
                                        _detect_simple() method.''')

    def detect(self,hf,channel,i_time,header):
        """
        For a given channel and timestep, return the observed intensity in each pixel.

        Parameters
        ----------
        channel : `dict`

        Returns
        -------
        counts : array-like
        """
        if self.use_temperature_response_functions:
            counts = self._detect_simple(hf,channel,i_time,header)
        else:
            counts = self._detect_full(hf,channel,i_time,header)
        if self.apply_psf:
            counts = scipy.ndimage.filters.gaussian_filter(counts,
                                                    channel['gaussian_width'].value)
        return counts

    def _detect_simple(self,hf,channel,i_time,header):
        """
        Calculate counts using the density and temperature response functions.
        No emissivity model needed.
        """
        dset = hf['{}/map'.format(channel['name'])]
        header['bunit'] = dset.attrs['units']
        return np.array(dset[i_time,:,:])

    def _detect_full(self,hf,channel,i_time,header):
        """
        Calculate counts use emissivity for a large number of transitions.
        Requires emissivity model.

        Notes
        -----
        This is necessary when taking into account non-equilibrium ionization.
        """
        raise NotImplementedError('''Full detect function not yet implemented. Set
                                    use_temperature_response_functions to True to use the
                                    _detect_simple() method.''')
