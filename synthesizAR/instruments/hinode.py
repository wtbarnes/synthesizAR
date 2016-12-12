"""
Class for Hinode/EIS instrument. Holds information about spectral, temporal, and spatial resolution
and other instrument-specific information.
"""

import os
import glob
import sys
import logging

import numpy as np
from scipy.interpolate import splrep,splev
import sunpy.map
import astropy.units as u

from synthesizAR.instruments import InstrumentBase,Pair


class InstrumentHinodeEIS(InstrumentBase):
    """
    Class for Extreme-ultraviolet Imaging Spectrometer (EIS) instrument on the Hinode spacecraft.
    Converts emissivity calculations for each loop into detector units based on the spectral,
    spatial, and temporal resolution along with the instrument response functions.
    """

    name = 'Hinode_EIS'
    cadence = 10.0*u.s
    resolution = Pair(1.0*u.arcsec/u.pixel,2.0*u.arcsec/u.pixel)
    fits_template = sunpy.map.header.MapMeta()
    fits_template['telescop'] = 'Hinode'
    fits_template['instrume'] = 'EIS'
    fits_template['detector'] = 'EIS'
    fits_template['waveunit'] = 'angstrom'

    def __init__(self,detector_file_dir,observing_time,observing_area=None):
        super().__init__(observing_time,observing_area)
        self._setup_from_file(detector_file_dir)

    def _setup_from_file(self,detector_file_dir):
        """
        Read instrument properties from files. This is a temporary solution and requires that the
        detector files all be collected into the same directory and be formatted in a specific way.

        .. warning:: This method will be modified once EIS response functions become
                    available in a different format.
        """
        eis_instr_files = glob.glob(os.path.join(detector_file_dir,'EIS_*_*.*.ins'))
        self.channels = []
        for eif in eis_instr_files:
            #extract some metadata from the filename
            base = os.path.splitext(os.path.basename(eif))[0].split('_')[1:]
            wave = float(base[-1])*u.angstrom
            if base[0][1].islower():
                el = base[0][:2]
                ion = base[0][2:]
            else:
                el = base[0][0]
                ion = base[0][1:]
            name = '{}_{}_{}'.format(el,ion,wave.value)
            #read the response function from the file
            with open(eif,'r') as f:
                lines = f.readlines()
            resp_x,resp_y = np.empty(int(lines[0])),np.empty(int(lines[0]))
            for i in range(1,int(lines[0])+1):
                resp_x[i-1],resp_y[i-1] = list(filter(None,lines[i].split(' ')))
            self.channels.append({'wavelength':wave,'name':name,
                    'response':{'x':resp_x*u.angstrom,
                                'y':resp_y*u.count/u.pixel/u.photon*u.steradian*u.cm**2}})

        self.channels = sorted(self.channels,key=lambda x:x['wavelength'])


    def detect(self,loop,channel):
        """
        Calculate response of Hinode/EIS detector for given loop object.
        """
        if 'model_wavelengths' not in channel:
            # only interpolate once
            channel_wvls = [w for w in loop.wavelengths \
                            if channel['response']['x'][0] <= w <= channel['response']['x'][-1]]
            channel_wvls *= loop.wavelengths.unit
            self.logger.debug('Interpolating emission wavelengths into response array for channel {}'.format(channel['name']))
            nots = splrep(channel['response']['x'].value,channel['response']['y'].value)
            channel['model_response'] = splev(channel_wvls.value,nots)*channel['response']['y'].unit
            channel['model_wavelengths'] = channel_wvls

        counts = np.zeros(loop.density.shape)
        for i,wavelength in enumerate(channel['model_wavelengths']):
                counts += channel['model_response'][i].value*loop.get_emission(wavelength).value

        return counts*loop.get_emission(wavelength).unit*channel['response']['y'].unit
