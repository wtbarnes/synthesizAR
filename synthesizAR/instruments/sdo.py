"""
Class for the SDO/AIA instrument. Holds information about the cadence and
spatial and spectroscopic resolution.
"""

import os
import sys
import logging
import json
import pkg_resources

import numpy as np
import dask
from scipy.interpolate import splrep, splev, interp1d
from scipy.ndimage import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import astropy.units as u
from sunpy.map import Map
try:
    from sunpy.map import MapMeta
except ImportError:
    # This has been renamed in the newest SunPy release, can eventually be removed
    # But necessary for the time being with current dev release
    from sunpy.util.metadata import MetaDict as MapMeta
import h5py

from synthesizAR.instruments import InstrumentBase, Pair


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

    fits_template = MapMeta()
    fits_template['telescop'] = 'SDO/AIA'
    fits_template['detector'] = 'AIA'
    fits_template['waveunit'] = 'angstrom'

    name = 'SDO_AIA'
    channels = [
        {'wavelength': 94*u.angstrom, 'telescope_number': 4,
            'gaussian_width': {'x': 0.951*u.pixel, 'y': 0.951*u.pixel}},
        {'wavelength': 131*u.angstrom, 'telescope_number': 1,
            'gaussian_width': {'x': 1.033*u.pixel, 'y': 1.033*u.pixel}},
        {'wavelength': 171*u.angstrom, 'telescope_number': 3,
            'gaussian_width': {'x': 0.962*u.pixel, 'y': 0.962*u.pixel}},
        {'wavelength': 193*u.angstrom, 'telescope_number': 2,
            'gaussian_width': {'x': 1.512*u.pixel, 'y': 1.512*u.pixel}},
        {'wavelength': 211*u.angstrom, 'telescope_number': 2,
            'gaussian_width': {'x': 1.199*u.pixel, 'y': 1.199*u.pixel}},
        {'wavelength': 335*u.angstrom, 'telescope_number': 1,
            'gaussian_width': {'x': 0.962*u.pixel, 'y': 0.962*u.pixel}}]

    cadence = 10.0*u.s
    resolution = Pair(0.600698*u.arcsec/u.pixel, 0.600698*u.arcsec/u.pixel, None)

    def __init__(self, observing_time, observing_area=None,
                 use_temperature_response_functions=True, apply_psf=True, emission_model=None):
        super().__init__(observing_time, observing_area)
        self.apply_psf = apply_psf
        self.use_temperature_response_functions = use_temperature_response_functions
        self.emission_model = emission_model
        if not self.use_temperature_response_functions and not self.emission_model:
            raise ValueError('Must supply an emission model if not using temperature response functions.')
        self._setup_channels()

    def _setup_channels(self):
        """
        Setup channel, specifically the wavelength or temperature response functions.

        Notes
        -----
        This should be replaced once the response functions are available in SunPy. Probably should
        configure wavelength response function interpolators also.
        """
        aia_fn = pkg_resources.resource_filename('synthesizAR', 'instruments/data/sdo_aia.json')
        with open(aia_fn, 'r') as f:
            aia_info = json.load(f)

        for channel in self.channels:
            channel['name'] = str(channel['wavelength'].value).strip('.0')
            channel['instrument_label'] = '{}_{}'.format(self.fits_template['detector'],
                                                         channel['telescope_number'])
            channel['wavelength_range'] = None

            if self.use_temperature_response_functions:
                x = aia_info[channel['name']]['temperature_response_x']
                y = aia_info[channel['name']]['temperature_response_y']
                channel['temperature_response_spline'] = splrep(x, y)
            else:
                x = aia_info[channel['name']]['response_x']
                y = aia_info[channel['name']]['response_y']
                channel['wavelength_response_spline'] = splrep(x, y)

    def build_detector_file(self, file_template, dset_shape, chunks, *args):
        """
        Allocate space for counts data.
        """
        additional_fields = ['{}'.format(channel['name']) for channel in self.channels]
        super().build_detector_file(file_template, dset_shape, chunks, *args, additional_fields=additional_fields)
        
    @staticmethod
    @dask.delayed
    def calculate_counts_simple(channel, loop):
        response_function = (splev(np.ravel(loop.electron_temperature),channel['temperature_response_spline'])
                             *u.count*u.cm**5/u.s/u.pixel)
        counts = np.reshape(np.ravel(loop.density**2)*response_function, np.shape(loop.density))
        return counts

    @staticmethod
    @dask.delayed
    def calculate_counts_full(channel, electron_temperature, density):
        counts = np.zeros(loop.electron_temperature.shape)
        for ion in self.emission_model.ions:
            fractional_ionization = loop.get_fractional_ionization(ion.chianti_ion.meta['Element'],
                                                                    ion.chianti_ion.meta['Ion'])
            if ion.emissivity is None:
                self.emission_model.calculate_emissivity()
            emiss = ion.emissivity
            interpolated_response = splev(ion.wavelength.value,
                                            channel['wavelength_response_spline'], ext=1)
            em_summed = np.dot(emiss.value, interpolated_response)
            tmp = np.reshape(map_coordinates(em_summed, np.vstack([itemperature, idensity])),
                                loop.electron_temperature.shape)
            tmp = (np.where(tmp > 0.0, tmp, 0.0)*emiss.unit*u.count/u.photon
                    * u.steradian/u.pixel*u.cm**2)
            counts_tmp = (fractional_ionization*loop.density*ion.chianti_ion.abundance*0.83
                            / (4*np.pi*u.steradian)*tmp)
            if not hasattr(counts, 'unit'):
                counts = counts*counts_tmp.unit
            counts += counts_tmp

        return counts
    
    def flatten_delayed_factory(self, loop, interp_s, save_path):
        """
        Create a list of dask.delayed procedures for each channel for a given loop
        """
        if self.use_temperature_response_functions:
            delayed_procedures = []
            for channel in self.channels:
                tmp_path = save_path.format(channel['name'],loop.name)
                y = self.calculate_counts_simple(channel, loop)
                delayed_procedures.append((channel['name'], self.interpolate_and_store(y, loop, self.observing_time, interp_s, tmp_path)))
            return delayed_procedures
        else:
            itemperature, idensity = self.emission_model.interpolate_to_mesh_indices(loop)
            raise NotImplementedError('No parallelized version of full counts calculation.')

    @staticmethod
    @dask.delayed
    def detect(counts_filename, channel, i_time, header, bins, bin_range, apply_psf):
        """
        For a given channel and timestep, map the intensity along the loop to the 3D field and
        return the AIA data product.

        Parameters
        ----------
        hf : `~h5py.File`
        channel : `dict`
        i_time : `int`
        header : `~sunpy.MapMeta`

        Returns
        -------
        AIA data product : `~sunpy.Map`
        """
        with h5py.File(counts_filename,'r') as hf:
            weights = np.array(hf[channel['name']][i_time,:])
            units = u.Unit(hf[channel['name']].attrs['units'])
            coordinates = u.Quantity(hf['coordinates'],hf['coordinates'].attrs['units'])

        hist, edges = np.histogramdd(coordinates.value, bins=bins, range=bin_range, weights=weights)
        header['bunit'] = (units*coordinates.unit).to_string()
        counts = np.dot(hist, np.diff(edges[2])).T

        if apply_psf:
            counts = gaussian_filter(counts, (channel['gaussian_width']['y'].value,
                                              channel['gaussian_width']['x'].value))
        return Map(counts, header)

    def detect_delayed_factory(self, i_time, field):
        """
        Create delayed procedures for binning the AIA intensities.
        """
        delayed_procedures = []
        for channel in self.channels:
            header = self.make_fits_header(field,channel)
            parameters = [self.counts_file,channel,i_time,header,self.bins,self.bin_range,self.apply_psf]
            delayed_procedures.append(self.detect(*parameters))

        return delayed_procedures

