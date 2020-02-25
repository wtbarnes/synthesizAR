"""
Class for the SDO/AIA instrument. Holds information about the cadence and
spatial and spectroscopic resolution.
"""

import pkg_resources

import numpy as np
import asdf
import zarr
import astropy.units as u
from aiapy.response import Channel
from scipy.interpolate import interp1d, interpn

from synthesizAR.instruments import InstrumentBase

_TEMPERATURE_RESPONSE_FILE = pkg_resources.resource_filename(
    'synthesizAR', 'instruments/data/aia_temperature_response.asdf')
with asdf.open(_TEMPERATURE_RESPONSE_FILE, 'r') as af:
    _TEMPERATURE_RESPONSE = af.tree


class InstrumentSDOAIA(InstrumentBase):
    """
    Instrument object for the Atmospheric Imaging Assembly on the Solar Dynamics Observatory

    Parameters
    ----------
    observing_time : `tuple`
        start and end of observing time
    observer : `~astropy.coordinates.SkyCoord`
    apply_psf : `bool`, optional
        If True (default), apply AIA point-spread function to images

    Examples
    --------
    """

    def __init__(self, observing_time, observer, pad_fov=None):
        super().__init__(observing_time, observer)
        self.telescope = 'SDO/AIA'
        self.detector = 'AIA'
        self.name = 'SDO_AIA'
        self.channels = [
            Channel(94*u.angstrom),
            Channel(131*u.angstrom),
            Channel(171*u.angstrom),
            Channel(193*u.angstrom),
            Channel(211*u.angstrom),
            Channel(335*u.angstrom),
        ]
        self.cadence = 12.0*u.s
        self.resolution = [0.600698, 0.600698]*u.arcsec/u.pixel
        self.pad_fov = pad_fov

    @staticmethod
    def calculate_intensity_kernel(loop, channel, **kwargs):
        em_model = kwargs.get('emission_model', None)
        if em_model:
            # Full intensity calculation using CHIANTI and the
            # wavelength response functions
            n = loop.density
            T = loop.electron_temperature
            Tn_flat = np.vstack((T.value.flatten(), n.value.flatten()))
            kernel = np.zeros(T.shape)
            # Get the group for this channel
            root = zarr.open(em_model.emissivity_table_filename, mode='r')
            grp = root[f'SDO_AIA/{channel.name}']
            for ion in em_model:
                ds = grp[ion.ion_name]
                if ds is None:
                    continue
                em_ion = u.Quantity(ds, ds.attrs['unit'])
                # Interpolate wavelength-convolved emissivity to loop n,T
                em_flat = interpn(
                    (em_model.temperature.to(T.unit).value, em_model.density.to(n.unit).value),
                    em_ion.value,
                    Tn_flat,
                    method='linear',
                    fill_value=None,
                    bounds_error=False,
                )
                em_ion_interp = np.reshape(em_flat, T.shape)
                em_ion_interp = u.Quantity(np.where(em_ion_interp < 0., 0., em_ion_interp),
                                           em_ion.unit)
                ionization_fraction = loop.get_ionization_fraction(ion)
                tmp = ion.abundance*0.83/(4*np.pi*u.steradian)*ionization_fraction*n*em_ion_interp
                if not hasattr(kernel, 'unit'):
                    kernel = kernel*tmp.unit
                kernel += tmp
        else:
            # Use tabulated temperature respone functions
            T, K = _TEMPERATURE_RESPONSE['temperature'], _TEMPERATURE_RESPONSE[channel.name]
            K_interp = np.interp(loop.electron_temperature, T, K)
            kernel = K_interp * loop.density**2
        return kernel

    def convolve_emissivities(self, channel, emission_model):
        """
        Compute product between wavelength response for `channel` and emissivity for all ions
        in an emission model.
        """
        em_convolved = {}
        r = channel.wavelength_response()
        f_interp = interp1d(channel.wavelength, r)
        for ion in emission_model:
            wavelength, emissivity = emission_model.get_emissivity(ion)
            if wavelength is None or emissivity is None:
                em_convolved[ion.ion_name] = None
            else:
                em_convolved[ion.ion_name] = np.dot(emissivity, f_interp(wavelength)) * r.unit

        return em_convolved

    def observe(self, skeleton, save_directory, channels=None, **kwargs):
        em_model = kwargs.get('emission_model')
        if em_model:
            # If using an emission model, we want to first convolve the wavelength-dependent
            # emissivities with the wavelength response functions and store them in the
            # emissivity table
            channels = self.channels if channels is None else channels
            root = zarr.open(em_model.emissivity_table_filename)
            grp = root.create_group(self.name)
            for channel in channels:
                em_convolved = self.convolve_emissivities(channel, em_model)
                chan_grp = grp.create_group(channel.name)
                for k in em_convolved:
                    ds = chan_grp.create_dataset(k, data=em_convolved[k].value)
                    ds.attrs['unit'] = em_convolved[k].unit.to_string()

        super().observe(skeleton, save_directory, channels=channels, **kwargs)
