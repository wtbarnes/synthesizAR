"""
Interface between loop object and scaling law calculations by Martens
"""
import numpy as np
from scipy.interpolate import interp1d
import astropy.units as u
import astropy.constants as const

from synthesizAR.models import MartensScalingLaws

__all__ = ['MartensInterface']


class MartensInterface(object):
    """
    Interface to the Martens scaling laws.

    The maximum temperature is determined using the RTV scaling laws.

    Parameters
    ----------
    heating_constant : `~astropy.units.Quantity`
        Constant of proportionality for heating rate scaling.
    temperature_cutoff : `~astropy.units.Quantity`, optional
        Lowest possible temperature in the loop. The Martens scaling laws
        permit temperatures of 0 K at the base of the loop which are unphysical.
    model_parameters : `dict`, optional
        Keyword arguments to `synthesizAR.models.MartensScalingLaws`

    See Also
    --------
    synthesizAR.models.MartensScalingLaws :
    """
    name = 'Martens2010'

    @u.quantity_input
    def __init__(self, heating_constant, temperature_cutoff=1e4*u.K, **model_parameters):
        self.temperature_cutoff = temperature_cutoff
        self.model_parameters = model_parameters
        self.heating_constant = heating_constant

    def load_results(self, loop):
        time = u.Quantity([0, ], 's')
        s_half = np.linspace(0, 1, 1000)*loop.length/2
        H = self.get_heating_constant(loop)
        msl = MartensScalingLaws(s_half, loop.length/2, H, **self.model_parameters)
        # Make sure there are no temperatures below specified cutoff
        msl_temperature = np.where(msl.temperature < self.temperature_cutoff,
                                   self.temperature_cutoff,
                                   msl.temperature)
        # Assume symmetric, reflect across apex
        s_full = np.concatenate((msl.s, msl.s[1:]+msl.s[-1])).value * msl.s.unit
        T_full = np.concatenate((msl_temperature,
                                 msl_temperature[::-1][1:])).value * msl_temperature.unit
        # Interpolate to centers of extrapolated fieldline grid cells
        s_center = loop.field_aligned_coordinate_center.to(s_full.unit).value
        temperature = interp1d(
            s_full.value,
            T_full.value,
            kind='slinear',
            bounds_error=False,
            fill_value=T_full[-1].value
        )(s_center)
        temperature = temperature[np.newaxis, :] * T_full.unit
        # Martens loops are isobaric, use ideal gas law to get density
        density = msl.pressure[0] / (2*const.k_B*temperature)
        # NOTE: Scaling laws do not provide any velocity information
        velocity = np.ones(time.shape+s_center.shape) * np.nan * u.cm/u.s

        return time, temperature, temperature, density, velocity

    def get_heating_constant(self, loop):
        """
        Override this to get a more complicated relationship between the loop
        and the heating rate
        """
        return self.heating_constant
