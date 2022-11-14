"""
Class for the SDO/AIA instrument. Holds information about the cadence and
spatial and spectroscopic resolution.
"""
import warnings
import pkg_resources

import numpy as np
import asdf
import zarr
import astropy.units as u
from aiapy.response import Channel
from aiapy.psf import filter_mesh_parameters
from scipy.interpolate import interp1d, interpn

from synthesizAR.instruments import InstrumentBase

__all__ = ['InstrumentSDOAIA', 'aia_kernel_quick']

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

    Examples
    --------
    """
    name = 'SDO_AIA'

    def __init__(self, observing_time, observer, **kwargs):
        self.channels = [
            Channel(94*u.angstrom),
            Channel(131*u.angstrom),
            Channel(171*u.angstrom),
            Channel(193*u.angstrom),
            Channel(211*u.angstrom),
            Channel(335*u.angstrom),
        ]
        cadence = kwargs.pop('cadence', 12.0 * u.s)
        resolution = kwargs.pop('resolution', [0.600698, 0.600698] * u.arcsec/u.pixel)
        # Add the Gaussian width for the PSF convolution
        psf_params = filter_mesh_parameters(use_preflightcore=True)
        for c in self.channels:
            psf_width = psf_params[c.channel]['width']
            c.psf_width = u.Quantity([psf_width, psf_width])
        super().__init__(observing_time, observer, resolution, cadence=cadence, **kwargs)

    @property
    def observatory(self):
        return 'SDO'

    @property
    def detector(self):
        return 'AIA'

    @property
    def telescope(self):
        return 'SDO/AIA'

    def get_instrument_name(self, channel):
        return f'{self.detector}_{channel.telescope_number}'

    @staticmethod
    def calculate_intensity_kernel(loop, channel, **kwargs):
        em_model = kwargs.get('emission_model', None)
        if em_model:
            # Full intensity calculation using CHIANTI and the
            # wavelength response functions
            n = loop.density
            T = loop.electron_temperature
            Tn_flat = np.stack((T.value.flatten(), n.value.flatten()), axis=1)
            kernel = np.zeros(T.shape)
            # Get the group for this channel
            root = zarr.open(em_model.emissivity_table_filename, mode='r')
            grp = root[f'SDO_AIA/{channel.name}']
            for ion in em_model:
                if ion.ion_name not in grp:
                    warnings.warn(f'Not including contribution from {ion.ion_name}')
                    continue
                ds = grp[ion.ion_name]
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
            kernel = aia_kernel_quick(channel.name, loop.electron_temperature, loop.density)
        return kernel

    def convolve_emissivities(self, channel, emission_model, **kwargs):
        """
        Compute product between wavelength response for `channel` and emissivity for all ions
        in an emission model.
        """
        em_convolved = {}
        r = channel.wavelength_response(**kwargs) * channel.plate_scale
        f_interp = interp1d(channel.wavelength, r, bounds_error=False, fill_value=0.0)
        for ion in emission_model:
            wavelength, emissivity = emission_model.get_emissivity(ion)
            # TODO: need to figure out a better way to propagate missing emissivities
            if wavelength is None or emissivity is None:
                em_convolved[ion.ion_name] = None
            else:
                em_convolved[ion.ion_name] = np.dot(emissivity, f_interp(wavelength)) * r.unit

        return em_convolved

    def observe(self, skeleton, save_directory=None, channels=None, **kwargs):
        em_model = kwargs.get('emission_model')
        if em_model:
            # TODO: skip if the file already exists?
            # If using an emission model, we want to first convolve the wavelength-dependent
            # emissivities with the wavelength response functions and store them in the
            # emissivity table
            channels = self.channels if channels is None else channels
            # NOTE: Don't open with 'w' because we want to preserve the emissivity table
            root = zarr.open(store=em_model.emissivity_table_filename, mode='a')
            if self.name not in root:
                grp = root.create_group(self.name)
            else:
                grp = root[self.name]
            # Get options for wavelength response
            include_crosstalk = kwargs.pop('include_crosstalk', True)
            obstime = self.observer.obstime if kwargs.pop('include_degradation', False) else None
            include_eve_correction = kwargs.pop('include_eve_correction', False)
            for channel in channels:
                em_convolved = self.convolve_emissivities(
                    channel,
                    em_model,
                    include_crosstalk=include_crosstalk,
                    obstime=obstime,
                    include_eve_correction=include_eve_correction,
                )
                if channel.name in grp:
                    chan_grp = grp[channel.name]
                else:
                    chan_grp = grp.create_group(channel.name)
                for k in em_convolved:
                    # NOTE: update dataset even when it already exists
                    if k in chan_grp:
                        ds = chan_grp[k]
                        ds[:, :] = em_convolved[k].value
                    else:
                        ds = chan_grp.create_dataset(k, data=em_convolved[k].value)
                    ds.attrs['unit'] = em_convolved[k].unit.to_string()

        return super().observe(skeleton, save_directory=save_directory, channels=channels, **kwargs)


@u.quantity_input
def aia_kernel_quick(channel,
                     temperature: u.K,
                     density: u.cm**(-3)) -> u.Unit('ct pix-1 s-1 cm-1'):
    r"""
    Calculate AIA intensity kernel for a given channel

    Compute the integrand of the AIA intensity integral,

    .. math::

        p_c = \int\mathrm{d}h\,K_c(T_e)n_e^2

    by interpolating the tabulated response curve to ``temperature``
    and multiplying by the square of ``density``.

    Parameters
    ----------
    channel : `str`
        Name of the AIA channel
    temperature : `~astropy.units.Quantity`
    density : `astropy.units.Quantity`
    """
    T, K = _TEMPERATURE_RESPONSE['temperature'], _TEMPERATURE_RESPONSE[channel]
    return np.interp(temperature, T, K) * density**2
