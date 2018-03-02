"""
Class for the SDO/AIA instrument. Holds information about the cadence and
spatial and spectroscopic resolution.
"""

import os
import json
import pkg_resources
import warnings
import toolz

import numpy as np
from scipy.interpolate import splrep, splev, interp1d
from scipy.ndimage import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import astropy.units as u
from sunpy.map import Map
from sunpy.util.metadata import MetaDict
from sunpy.coordinates.frames import Helioprojective
import h5py
try:
    import distributed
except ImportError:
    warnings.warn('Dask distributed scheduler required for parallel execution')

import synthesizAR
from synthesizAR.util import SpatialPair, is_visible
from synthesizAR.instruments import InstrumentBase


class InstrumentSDOAIA(InstrumentBase):
    """
    Instrument object for the Atmospheric Imaging Assembly on the Solar Dynamics Observatory

    Parameters
    ----------
    observing_time : `tuple`
        start and end of observing time
    observer_coordinate : `~astropy.coordinates.SkyCoord`, optional
    apply_psf : `bool`

    Examples
    --------
    """

    def __init__(self, observing_time, observer_coordinate=None, apply_psf=True):
        self.fits_template['telescop'] = 'SDO/AIA'
        self.fits_template['detector'] = 'AIA'
        self.fits_template['waveunit'] = 'angstrom'
        self.name = 'SDO_AIA'
        self.channels = [
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
        self.cadence = 10.0*u.s
        self.resolution = SpatialPair(x=0.600698*u.arcsec/u.pixel, y=0.600698*u.arcsec/u.pixel,
                                      z=None)
        self.apply_psf = apply_psf
        super().__init__(observing_time, observer_coordinate=observer_coordinate)
        self._setup_channels()

    def _setup_channels(self):
        """
        Setup channel, specifically the wavelength or temperature response functions.

        .. note:: This should be replaced once the response functions are available in SunPy.
        """
        aia_fn = pkg_resources.resource_filename('synthesizAR', 'instruments/data/sdo_aia.json')
        with open(aia_fn, 'r') as f:
            aia_info = json.load(f)

        for channel in self.channels:
            channel['name'] = str(channel['wavelength'].value).strip('.0')
            channel['instrument_label'] = '{}_{}'.format(self.fits_template['detector'],
                                                         channel['telescope_number'])
            channel['wavelength_range'] = None
            x = aia_info[channel['name']]['temperature_response_x']
            y = aia_info[channel['name']]['temperature_response_y']
            channel['temperature_response_spline'] = splrep(x, y)
            x = aia_info[channel['name']]['response_x']
            y = aia_info[channel['name']]['response_y']
            channel['wavelength_response_spline'] = splrep(x, y)

    def build_detector_file(self, file_template, dset_shape, chunks, *args, parallel=False):
        """
        Allocate space for counts data.
        """
        additional_fields = ['{}'.format(channel['name']) for channel in self.channels]
        super().build_detector_file(file_template, dset_shape, chunks, *args,
                                    additional_fields=additional_fields, parallel=parallel)
        
    @staticmethod
    def calculate_counts_simple(channel, loop, *args):
        """
        Calculate the AIA intensity using only the temperature response functions.
        """
        response_function = (splev(np.ravel(loop.electron_temperature),
                                   channel['temperature_response_spline'])
                             * u.count * u.cm**5 / u.s / u.pixel)
        counts = np.reshape(np.ravel(loop.density**2)*response_function, loop.density.shape)
        return counts

    @staticmethod
    def flatten_emissivities(channel, emission_model):
        """
        Compute product between wavelength response and emissivity for all ions
        """
        flattened_emissivities = []
        for ion in emission_model:
            wavelength, emissivity = emission_model.get_emissivity(ion)
            if wavelength is None or emissivity is None:
                flattened_emissivities.append(None)
                continue
            interpolated_response = splev(wavelength.value, channel['wavelength_response_spline'],
                                          ext=1)
            em_summed = np.dot(emissivity.value, interpolated_response)
            unit = emissivity.unit*u.count/u.photon*u.steradian/u.pixel*u.cm**2
            flattened_emissivities.append(u.Quantity(em_summed, unit))

        return flattened_emissivities

    @staticmethod
    def calculate_counts_full(channel, loop, emission_model, flattened_emissivities):
        """
        Calculate the AIA intensity using the wavelength response functions and a
        full emission model.
        """
        density = loop.density
        electron_temperature = loop.electron_temperature
        counts = np.zeros(electron_temperature.shape)
        itemperature, idensity = emission_model.interpolate_to_mesh_indices(loop)
        for ion, flat_emiss in zip(emission_model, flattened_emissivities):
            if flat_emiss is None:
                continue
            ionization_fraction = emission_model.get_ionization_fraction(loop, ion)
            tmp = np.reshape(map_coordinates(flat_emiss.value, np.vstack([itemperature, idensity])),
                             electron_temperature.shape)
            tmp = u.Quantity(np.where(tmp < 0., 0., tmp), flat_emiss.unit)
            counts_tmp = ion.abundance*0.83/(4*np.pi*u.steradian)*ionization_fraction*density*tmp
            if not hasattr(counts, 'unit'):
                counts = counts*counts_tmp.unit
            counts += counts_tmp

        return counts
    
    def flatten_serial(self, loops, interpolated_loop_coordinates, hf, emission_model=None):
        """
        Interpolate intensity in each channel to temporal resolution of the instrument
        and appropriate spatial scale.
        """
        if emission_model is None:
            calculate_counts = self.calculate_counts_simple
            flattened_emissivities = None
        else:
            calculate_counts = self.calculate_counts_full

        for channel in self.channels:
            start_index = 0
            dset = hf[channel['name']]
            if emission_model is not None:
                flattened_emissivities = self.flatten_emissivities(channel, emission_model)
            for loop, interp_s in zip(loops, interpolated_loop_coordinates):
                c = calculate_counts(channel, loop, emission_model, flattened_emissivities)
                y = self.interpolate_and_store(self.observing_time.value, c, loop, interp_s)
                self.commit(y, dset, start_index)
                start_index += interp_s.shape[0]

    def flatten_parallel(self, loops, interpolated_loop_coordinates, tmp_dir, emission_model=None):
        """
        Interpolate intensity in each channel to temporal resolution of the instrument
        and appropriate spatial scale. Returns a dask task.
        """
        # Setup scheduler
        client = distributed.get_client()
        lock = distributed.Lock(name=f'hdf5_{self.name}')
        start_indices = np.insert(np.array(
            [s.shape[0] for s in interpolated_loop_coordinates]).cumsum()[:-1], 0, 0)
        if emission_model is None:
            calculate_counts = self.calculate_counts_simple
            flat_emiss = None
        else:
            calculate_counts = self.calculate_counts_full

        futures = []
        for channel in self.channels:
            paths = [os.path.join(tmp_dir, f"{l.name}_{self.name}_{channel['name']}.npz")
                     for l in loops]
            # Flatten emissivities for appropriate channel
            if emission_model is not None:
                flat_emiss = self.flatten_emissivities(channel, emission_model)
            # Create partial functions
            partial_counts = toolz.curry(calculate_counts)(
                channel, emission_model=emission_model, flattened_emissivities=flat_emiss)
            partial_interp = toolz.curry(self.interpolate_and_store)(self.observing_time.value)
            # Map functions to iterables
            y_futures = client.map(partial_counts, loops)
            interp_futures = client.map(partial_interp, y_futures, loops,
                                        interpolated_loop_coordinates, start_indices, paths)
            # Assemble into array
            assemble_future = client.submit(
                self.assemble_arrays, interp_futures, channel['name'], self.counts_file, lock)
            # Block until complete
            distributed.client.wait([assemble_future])
            futures.append(assemble_future)

        return futures

    def detect(self, channel, i_time, header, bins, bin_range):
        """
        For a given channel and timestep, map the intensity along the loop to the 3D field and
        return the AIA data product.

        Parameters
        ----------
        channel : `dict`
        i_time : `int`
        header : `~sunpy.util.metadata.MetaDict`
        bins : `~synthesizAR.util.SpatialPair`
        bin_range : `~synthesizAR.util.SpatialPair`

        Returns
        -------
        AIA data product : `~sunpy.map.Map`
        """
        with h5py.File(self.counts_file, 'r') as hf:
            weights = np.array(hf[channel['name']][i_time, :])
            units = u.Unit(hf[channel['name']].attrs['units'])

        hpc_coordinates = self.total_coordinates
        dz = np.diff(bin_range.z)[0] / bins.z * (1. * u.pixel)
        visible = is_visible(hpc_coordinates, self.observer_coordinate)
        hist, _, _ = np.histogram2d(hpc_coordinates.Tx.value, hpc_coordinates.Ty.value,
                                    bins=(bins.x.value, bins.y.value),
                                    range=(bin_range.x.value, bin_range.y.value),
                                    weights=visible * weights * dz.value)
        header['bunit'] = (units * dz.unit).to_string()

        if self.apply_psf:
            counts = gaussian_filter(hist.T, (channel['gaussian_width']['y'].value,
                                              channel['gaussian_width']['x'].value))
        return Map(counts, header)
