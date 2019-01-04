"""
Interface between loop object and scaling law calculations by Martens
"""
import numpy as np
from scipy.interpolate import interp1d
import astropy.units as u

from synthesizAR.physics import MartensScalingLaws, RTVScalingLaws

__all__ = ['MartensInterface']


class MartensInterface(object):
    """
    Interface to the Martens scaling laws.

    The maximum temperature is determined using the RTV scaling laws.

    Parameters
    ----------
    plasma_beta : `float`, optional
        Ratio of plasma to magnetic pressure
    martens_kwargs : `dict`, optional
        Keyword arguments to `synthesizAR.physics.MartensScalingLaws`
    rtv_kwargs : `dict`, optional
        Keyword arguments to `synthesizAR.physics.RTVScalingLaws`

    See Also
    --------
    synthesizAR.physics.MartensScalingLaws :
    synthesizAR.physics.RTVScalingLaws :
    """

    def __init__(self, plasma_beta=0.001, martens_kwargs=None, rtv_kwargs=None):
        self.plasma_beta = plasma_beta
        self.martens_kwargs = {} if martens_kwargs is None else martens_kwargs
        self.rtv_kwargs = {} if rtv_kwargs is None else rtv_kwargs

    def load_results(self, loop):
        time = u.Quantity([0,], 's')
        # Get maximum temperature from RTV scaling laws
        pressure = loop.field_strength.mean().to(u.G).value**2 / 8 / np.pi * self.plasma_beta
        pressure = pressure * u.dyne / (u.cm**2)
        rtv = RTVScalingLaws(loop.full_length/2, pressure=pressure, **self.rtv_kwargs)
        s_half = np.arange(0, loop.full_length.to(u.Mm).value/2, 0.1) * u.Mm
        martens = MartensScalingLaws(s_half, rtv.max_temperature, **self.martens_kwargs)
        # Assume symmetric, reflect across apex
        s_full = np.concatenate((martens.s, martens.s[1:]+martens.s[-1])).value * martens.s.unit
        T_full = (np.concatenate((martens.temperature, martens.temperature[::-1][1:])).value
                  * martens.temperature.unit)
        n_full = (np.concatenate((martens.density, martens.density[::-1][1:])).value
                  * martens.density.unit)
        # Interpolate to extrapolated field
        temperature = interp1d(s_full.value, T_full.value, kind='slinear', bounds_error=False,
                               fill_value=T_full[-1].value)(
                                  loop.field_aligned_coordinate.to(s_full.unit).value)
        temperature = temperature[np.newaxis, :] * T_full.unit
        density = interp1d(s_full.value, n_full.value, kind='slinear', bounds_error=False,
                           fill_value=n_full[-1].value)(
                               loop.field_aligned_coordinate.to(s_full.unit).value)
        density = density[np.newaxis, :] * n_full.unit
        # Scaling laws do not provide any velocity information
        velocity = np.ones(time.shape+loop.field_aligned_coordinate.shape) * np.nan * u.cm/u.s

        return time, temperature, temperature, density, velocity
