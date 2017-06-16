"""
Class for Hinode/EIS instrument. Holds information about spectral, temporal, and spatial resolution
and other instrument-specific information.
"""

import os
import glob
import sys
import logging
import json
import pkg_resources

import numpy as np
from scipy.interpolate import splrep, splev, interp1d
from scipy.ndimage.filters import gaussian_filter
try:
    from sunpy.map import MapMeta
except ImportError:
    # This has been renamed in the newest SunPy release, can eventually be removed
    # But necessary for the time being with current dev release
    from sunpy.util.metadata import MetaDict as MapMeta
import astropy.units as u
import astropy.constants as const
import h5py
import periodictable

from synthesizAR.instruments import InstrumentBase, Pair
from synthesizAR.util import EISCube


class InstrumentHinodeEIS(InstrumentBase):
    """
    Class for Extreme-ultraviolet Imaging Spectrometer (EIS) instrument on the Hinode spacecraft.
    Converts emissivity calculations for each loop into detector units based on the spectral,
    spatial, and temporal resolution along with the instrument response functions.
    """

    name = 'Hinode_EIS'
    cadence = 10.0*u.s
    resolution = Pair(1.0*u.arcsec/u.pixel, 2.0*u.arcsec/u.pixel, None)
    fits_template = MapMeta()
    fits_template['telescop'] = 'Hinode'
    fits_template['instrume'] = 'EIS'
    fits_template['detector'] = 'EIS'
    fits_template['waveunit'] = 'angstrom'

    @u.quantity_input(window=u.angstrom)
    def __init__(self, observing_time, observing_area=None, window=0.5*u.angstrom, apply_psf=True):
        super().__init__(observing_time, observing_area)
        self._setup_channels()
        self.apply_psf = apply_psf
        self.window = window

    def _setup_channels(self):
        """
        Read instrument properties from files. This is a temporary solution and requires that the
        detector files all be collected into the same directory and be formatted in a specific way.

        .. warning:: This method will be modified once EIS response functions become
                    available in a different format.
        """
        hinode_fn = pkg_resources.resource_filename('synthesizAR',
                                                    'instruments/data/hinode_eis.json')
        with open(hinode_fn, 'r') as f:
            eis_info = json.load(f)

        self.channels = []
        for key in eis_info:
            if key != 'name' and key != 'description':
                self.channels.append({
                    'wavelength': eis_info[key]['wavelength']*u.Unit(eis_info[key]['wavelength_units']),
                    'name': key,
                    'response': {
                        'x': eis_info[key]['response_x']*u.Unit(eis_info[key]['response_x_units']),
                        'y': eis_info[key]['response_y']*u.Unit(eis_info[key]['response_y_units'])},
                    'spectral_resolution': eis_info[key]['spectral_resolution']*u.Unit(eis_info[key]['spectral_resolution_units']),
                    'gaussian_width': {'x': (3.*u.arcsec)/self.resolution.x,
                                       'y': (3.*u.arcsec)/self.resolution.y}
                    'instrument_width': eis_info[key]['instrument_width']*u.Unit(eis_info[key]['instrument_width_units']),
                    'wavelength_range': [eis_info[key]['response_x'][0],
                                         eis_info[key]['response_x'][-1]]*u.Unit(eis_info[key]['response_x_units'])})

        self.channels = sorted(self.channels, key=lambda x: x['wavelength'])

    def make_fits_header(self, field, channel):
        """
        Extend base method to include extra wavelength dimension.
        """
        header = super().make_fits_header(field, channel)
        header['wavelnth'] = channel['wavelength'].value
        header['naxis3'] = len(channel['response']['x'])
        header['ctype3'] = 'wavelength'
        header['cunit3'] = 'angstrom'
        header['cdelt3'] = np.fabs(np.diff(channel['response']['x']).value[0])
        return header

    def build_detector_file(self, field, num_loop_coordinates, file_format):
        """
        Build HDF5 files to store detector counts
        """
        super().build_detector_file(num_loop_coordinates, file_format)
        with h5py.File(self.counts_file, 'a') as hf:
            for line in field.loops[0].resolved_wavelengths:
                if str(line.value) not in hf:
                    hf.create_dataset('{}'.format(str(line.value)),
                                      (len(self.observing_time), num_loop_coordinates),
                                      chunks=True)

    def flatten(self, loop, interp_s, hf, start_index):
        """
        Flatten loop emission to HDF5 file for given number of wavelengths
        """
        for wavelength in loop.resolved_wavelengths:
            emiss, ion_name = loop.get_emission(wavelength, return_ion_name=True)
            dset = hf['{}'.format(str(wavelength.value))]
            dset.attrs['ion_name'] = ion_name
            self.interpolate_and_store(emiss, loop, interp_s, dset, start_index)

    def detect(self, hf, channel, i_time, header, temperature, los_velocity):
        """
        Calculate response of Hinode/EIS detector for given loop object.
        """
        # trim the instrument response to the appropriate wavelengths
        trimmed_indices = []
        for w in channel['model_wavelengths']:
            indices = np.where(np.logical_and(channel['response']['x'] >= w-self.window,
                                              channel['response']['x'] <= w+self.window))
            trimmed_indices += indices[0].tolist()
        trimmed_indices = list(sorted(set(trimmed_indices+[0, len(channel['response']['x'])-1])))
        response_x = channel['response']['x'][trimmed_indices]
        response_y = channel['response']['y'][trimmed_indices]

        # compute the response
        counts = np.zeros(temperature.shape+response_x.shape)
        for wavelength in channel['model_wavelengths']:
            # thermal width + instrument width
            ion_name = hf['{}'.format(str(wavelength.value))].attrs['ion_name']
            ion_mass = periodictable.elements.symbol(ion_name.split(' ')[0]).mass*const.u.cgs
            thermal_velocity = 2.*const.k_B.cgs*temperature/ion_mass
            thermal_velocity = np.expand_dims(thermal_velocity, axis=2)*thermal_velocity.unit
            line_width = ((wavelength**2)/(2.*const.c.cgs**2)*thermal_velocity
                          + (channel['instrument_width']/(2.*np.sqrt(2.*np.log(2.))))**2)
            # doppler shift due to LOS velocity
            doppler_shift = wavelength*los_velocity/const.c.cgs
            doppler_shift = np.expand_dims(doppler_shift, axis=2)*doppler_shift.unit
            # combine emissivity with instrument response function
            dset = hf['{}'.format(str(wavelength.value))]
            hist, edges = np.histogramdd(self.total_coordinates.value,
                                         bins=[self.bins.x, self.bins.y, self.bins.z],
                                         range=[self.bin_range.x, self.bin_range.y, self.bin_range.z],
                                         weights=np.array(dset[i_time, :]))
            emiss = np.dot(hist, np.diff(edges[2])).T
            emiss = np.expand_dims(emiss, axis=2)*u.Unit(dset.attrs['units'])*self.total_coordinates.unit
            intensity = emiss*response_y/np.sqrt(2.*np.pi*line_width)
            intensity *= np.exp(-((response_x - wavelength - doppler_shift)**2)/(2.*line_width))
            if not hasattr(counts, 'unit'):
                counts = counts*intensity.unit
            counts += intensity

        header['bunit'] = counts.unit.to_string()
        if self.apply_psf:
            counts = (gaussian_filter(counts.value, (channel['gaussian_width']['y'].value,
                                                     channel['gaussian_width']['x'].value, 0))
                      * counts.unit)

        return EISCube(data=counts, header=header, wavelength=response_x)
