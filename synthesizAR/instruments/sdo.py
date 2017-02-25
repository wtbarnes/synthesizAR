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
from scipy.interpolate import splrep,splev,interp1d
import scipy.ndimage
import astropy.units as u
from sunpy.map import Map,MapMeta
import h5py

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


    fits_template = MapMeta()
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

    cadence = 10.0*u.s
    resolution = Pair(0.600698*u.arcsec/u.pixel,0.600698*u.arcsec/u.pixel,None)

    def __init__(self, observing_time, observing_area=None,
    use_temperature_response_functions=True,apply_psf=True):
        super().__init__(observing_time,observing_area)
        self.apply_psf = apply_psf
        self.use_temperature_response_functions = use_temperature_response_functions
        self._setup_channels()

    def _setup_channels(self):
        """
        Setup channel, specifically the wavelength or temperature response functions.

        Notes
        -----
        This should be replaced once the response functions are available in SunPy. Probably should configure wavelength response function interpolators also.
        """
        aia_fn = pkg_resources.resource_filename('synthesizAR','instruments/data/sdo_aia.json')
        with open(aia_fn,'r') as f:
            aia_info = json.load(f)

        for channel in self.channels:
            channel['name'] = str(channel['wavelength'].value).strip('.0')
            channel['instrument_label'] = '{}_{}'.format(self.fits_template['detector'],
                                                        channel['telescope_number'])
            if self.use_temperature_response_functions:
                x = aia_info[channel['name']]['temperature_response_x']
                y = aia_info[channel['name']]['temperature_response_y']
                channel['temperature_response_spline'] = splrep(x,y)
            else:
                x = aia_info[channel['name']]['response_x']
                y = aia_info[channel['name']]['response_y']
                channel['wavelength_response_spline'] = splrep(x,y)
                channel['wavelength_response_units'] = u.Unit(aia_info[channel['name']]['response_y_units'])
                channel['wavelength_range'] = [x[0],x[-1]]\
                                    *u.Unit(aia_info[channel['name']]['response_x_units'])

    def build_detector_file(self,field,num_loop_coordinates,file_format):
        """
        Allocate space for counts data.
        """
        super().build_detector_file(num_loop_coordinates,file_format)
        if self.use_temperature_response_functions:
            with h5py.File(self.counts_file,'a') as hf:
                for channel in self.channels:
                    if channel['name'] not in hf:
                        hf.create_dataset('{}'.format(channel['name']),
                                            (len(self.observing_time),num_loop_coordinates))

    def flatten(self,loop,interp_s,hf,start_index):
        """
        Flatten channel counts to HDF5 file
        """
        for channel in self.channels:
            dset = hf['{}'.format(channel['name'])]
            if self.use_temperature_response_functions:
                response_function = splev(np.ravel(loop.temperature),
                                            channel['temperature_response_spline']
                                         )*u.count*u.cm**5/u.s/u.pixel
                counts = np.reshape(np.ravel(loop.density**2)*response_function,
                                    np.shape(loop.density))
            else:
                key = '{}_{}'.format(self.name,channel['name'])
                counts = loop.get_emission(key)

            self.interpolate_and_store(counts,loop,interp_s,dset,start_index)

    def detect(self,hf,channel,i_time,header,*args):
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
        dset = hf['{}'.format(channel['name'])]
        hist,edges = np.histogramdd(self.total_coordinates.value,
                                    bins=[self.bins.x,self.bins.y,self.bins.z],
                                    range=[self.bin_range.x,self.bin_range.y,self.bin_range.z],
                                    weights=np.array(dset[i_time,:]))
        header['bunit'] = (u.Unit(dset.attrs['units'])*self.total_coordinates.unit).to_string()
        counts = np.dot(hist,np.diff(edges[2])).T

        if self.apply_psf:
            counts = scipy.ndimage.filters.gaussian_filter(counts,
                                                    channel['gaussian_width'].value)
        return Map(counts,header)
