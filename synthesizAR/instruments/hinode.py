"""
Class for Hinode/EIS instrument. Holds information about spectral, temporal, and spatial resolution
and other instrument-specific information.
"""

import os
import glob
import sys
import logging

import numpy as np
from scipy.interpolate import splrep,splev,interp1d
import sunpy.map
import astropy.units as u
import astropy.constants as const
import h5py
import periodictable

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
                                'y':resp_y*u.count/u.pixel/u.photon*u.steradian*u.cm**2},
                    'spectral_resolution':float(lines[int(lines[0])+1])*u.angstrom,
                    'instrument_width':float(lines[int(lines[0])+2])*u.angstrom,
                    'wavelength_range':[resp_x[0],resp_x[-1]]*u.angstrom})

        self.channels = sorted(self.channels,key=lambda x:x['wavelength'])

    def make_fits_header(self,field,channel):
        """
        Extend base method to include extra wavelength dimension.
        """
        header = super().make_fits_header(field,channel)
        header['naxis3'] = len(channel['response']['x'])
        header['ctype3'] = 'wavelength'
        header['cunit3'] = 'angstrom'
        header['cdelt3'] = np.fabs(np.diff(channel['response']['x']).value[0])
        return header

    def build_detector_file(self,field,num_loop_coordinates,file_format):
        """
        Build HDF5 files to store detector counts
        """
        super().build_detector_file(num_loop_coordinates,file_format)
        if not os.path.exists(self.counts_file):
            with h5py.File(self.counts_file,'a') as hf:
                for line in field.loops[0].wavelengths:
                    hf.create_dataset('{}'.format(str(line.value)),
                                        (len(self.observing_time),num_loop_coordinates))

    def flatten(self,loop,interp_s,hf,start_index):
        """
        Flatten loop emission to HDF5 file for given number of wavelengths
        """
        for wavelength in loop.wavelengths:
            emiss,ion_name = loop.get_emission(wavelength,return_ion_name=True)
            dset = hf['{}'.format(str(wavelength.value))]
            hf['{}'.format(str(wavelength.value))].attrs['ion_name'] = ion_name
            self.interpolate_and_store(emiss,loop,interp_s,dset,start_index)

    def detect(self,hf,channel,i_time,header,temperature,los_velocity):
        """
        Calculate response of Hinode/EIS detector for given loop object.
        """
        counts = np.zeros(temperature.shape+channel['response']['x'].shape)
        for wavelength in channel['model_wavelengths']:
            #thermal width + instrument width
            ion_name = hf['{}'.format(str(wavelength.value))].attrs['ion_name']
            ion_mass = periodictable.elements.symbol(ion_name.split(' ')[0]).mass*const.u.cgs
            thermal_velocity = 2.*const.k_B.cgs*temperature/ion_mass
            thermal_velocity = np.expand_dims(thermal_velocity,axis=2)*thermal_velocity.unit
            line_width = 2./3.*wavelength**2/(const.c.cgs**2)*thermal_velocity \
                        + 0.36*channel['instrument_width']**2
            #doppler shift due to LOS velocity
            doppler_shift = wavelength*los_velocity/const.c.cgs
            doppler_shift = np.expand_dims(doppler_shift,axis=2)*doppler_shift.unit
            #combine emissivity with instrument response function
            dset = hf['{}'.format(str(wavelength.value))]
            hist,edges = np.histogramdd(self.total_coordinates.value,
                                        bins=[self.bins.x,self.bins.y,self.bins.z],
                                        range=[self.bin_range.x,self.bin_range.y,self.bin_range.z],
                                        weights=np.array(dset[i_time,:]))
            emiss = np.dot(hist,np.diff(edges[2])).T
            emiss = np.expand_dims(emiss,axis=2)\
                    *u.Unit(dset.attrs['units'])*self.total_coordinates.unit
            intensity = emiss*channel['response']['y']/np.sqrt(np.pi*line_width)
            intensity *= np.exp(-(channel['response']['x'] - wavelength - doppler_shift)**2\
                        /line_width)
            if not hasattr(counts,'unit'):
                counts = counts*intensity.unit
            counts += intensity

        header['bunit'] = counts.unit.to_string()

        return counts
