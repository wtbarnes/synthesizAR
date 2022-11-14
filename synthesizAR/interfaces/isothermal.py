"""
Interface to an isothermal loop model
"""
import astropy.units as u
import numpy as np

from synthesizAR.models import Isothermal, RTVScalingLaws
from synthesizAR.models.heating import b_over_l_scaling

__all__ = ['IsothermalInterface', 'IsothermalRTVInterface']


class IsothermalInterface:
    """
    Interface to isothermal loop model
    """
    name = 'isothermal'

    def __init__(self,
                 temperature: u.Unit('K') = None,
                 base_pressure: u.Unit('dyne cm-2') = None):
        self._temperature = temperature
        self._base_pressure = base_pressure

    def load_results(self, loop):
        T_0 = self.get_temperature(loop)
        p_0 = self.get_base_pressure(loop)
        iso = Isothermal(loop.field_aligned_coordinate_center,
                         loop.coordinate_center.spherical.distance,
                         T_0,
                         p_0)
        time = u.Quantity([0, ], 's')
        ones = np.ones(time.shape + iso.pressure.shape)
        temperature = T_0 * ones
        velocity = np.nan * ones * (u.cm / u.s)
        density = iso.density.reshape(ones.shape)
        return time, temperature, temperature, density, velocity

    def get_temperature(self, loop):
        return self._temperature

    def get_base_pressure(self, loop):
        return self._base_pressure


class IsothermalRTVInterface(IsothermalInterface):
    """
    This is the same as `~synthesizAR.interfaces.IsothermalInterface`
    except that the temperature for each loop is computed using the
    RTV scaling laws, with the heating rate calculated using
    `~synthesizAR.models.heating.b_over_l_scaling`.
    """
    name = 'isothermal_rtv'

    def __init__(self, heating_params=None, rtv_params=None):
        self.heating_params = {} if heating_params is None else heating_params
        self.rtv_params = {} if rtv_params is None else rtv_params

    def get_temperature(self, loop):
        heating_rate = self.get_heating_rate(loop)
        rtv = RTVScalingLaws(loop.length/2, heating_rate=heating_rate, **self.rtv_params)
        return rtv.max_temperature

    def get_heating_rate(self, loop):
        return b_over_l_scaling(loop, **self.heating_params)
