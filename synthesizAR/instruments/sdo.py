"""
Class for the SDO/AIA instrument. Holds information about the cadence and
spatial and spectroscopic resolution.
"""
import asdf
import astropy.units as u
import numpy as np

from aiapy.psf import filter_mesh_parameters
from aiapy.response import Channel
from astropy.utils.data import get_pkg_data_filename
from functools import cached_property
from scipy.interpolate import interpn

from synthesizAR.instruments import InstrumentBase
from synthesizAR.util.decorators import return_quantity_as_tuple

__all__ = ['InstrumentSDOAIA', 'aia_kernel_quick']

_TEMPERATURE_RESPONSE_FILE = get_pkg_data_filename('data/aia_temperature_response.asdf',
                                                   package='synthesizAR.instruments')
with asdf.open(_TEMPERATURE_RESPONSE_FILE, 'r', memmap=False) as af:
    _TEMPERATURE_RESPONSE = af.tree


class AIAChannel(Channel):

    @u.quantity_input
    def wavelength_response(self) -> u.Unit('cm2 DN ph-1 sr pix-1'):
        response = super().wavelength_response(
            include_crosstalk=True,
            obstime=None,
            include_eve_correction=False,
            correction_table=None,  # Can remove this after aiapy v0.11
        )
        # NOTE: Can remove this once Channel refactor is done
        response *= self.plate_scale
        return response

    @property
    def psf_width(self):
        # Add the Gaussian width for the PSF convolution
        psf_width = filter_mesh_parameters(
            use_preflightcore=True
        )[self.channel]['width']
        return u.Quantity([psf_width, psf_width])


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
        resolution = kwargs.pop('resolution', [0.600698, 0.600698] * u.arcsec/u.pixel)
        cadence = kwargs.pop('cadence', 12.0 * u.s)
        super().__init__(
            observing_time=observing_time,
            observer=observer,
            resolution=resolution,
            cadence=cadence,
            **kwargs,
        )

    @property
    def observatory(self):
        return 'SDO'

    @property
    def detector(self):
        return 'AIA'

    @property
    def telescope(self):
        return 'SDO/AIA'

    @cached_property
    def channels(self):
        return [
            AIAChannel(94*u.angstrom),
            AIAChannel(131*u.angstrom),
            AIAChannel(171*u.angstrom),
            AIAChannel(193*u.angstrom),
            AIAChannel(211*u.angstrom),
            AIAChannel(335*u.angstrom),
        ]

    @property
    def _expected_unit(self):
        return u.DN / (u.pix * u.s)

    def get_instrument_name(self, channel):
        return f'{self.detector}_{channel.telescope_number}'

    @staticmethod
    @return_quantity_as_tuple
    def calculate_intensity_kernel(loop, channel, **kwargs):
        em_model = kwargs.get('emission_model', None)
        if em_model:
            # Full intensity calculation using CHIANTI and the
            # wavelength response functions
            n = loop.density
            T = loop.electron_temperature
            Tn_flat = np.stack((T.value.flatten(), n.value.flatten()), axis=1)
            kernel = u.Quantity(np.zeros(T.shape), 'DN pix-1 s-1 cm-1')
            for ion in em_model:
                # NOTE: This is cached such that it is only calculated once
                # for a given ion/channel pair
                em_ion = em_model.calculate_narrowband_emissivity(ion, channel)
                # Interpolate wavelength-convolved emissivity to loop n,T
                em_flat = interpn(
                    (em_model.temperature.to_value(T.unit), em_model.density.to_value(n.unit)),
                    em_ion.value,
                    Tn_flat,
                    method='linear',
                    fill_value=None,
                    bounds_error=False,
                )
                em_ion_interp = np.reshape(em_flat, T.shape)
                em_ion_interp = u.Quantity(np.where(em_ion_interp < 0., 0., em_ion_interp), em_ion.unit)
                ionization_fraction = loop.get_ionization_fraction(ion)
                kernel += n**2*ionization_fraction*em_ion_interp/(4*np.pi*u.steradian)
        else:
            # Use tabulated temperature response functions
            kernel = aia_kernel_quick(channel.name, loop.electron_temperature, loop.density)
        return kernel


@u.quantity_input
def aia_kernel_quick(channel,
                     temperature: u.K,
                     density: u.cm**(-3)) -> u.Unit('DN pix-1 s-1 cm-1'):
    r"""
    Calculate AIA intensity kernel for a given channel.

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
