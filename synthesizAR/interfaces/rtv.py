"""
Interface between loop object and RTV scaling laws
"""
import astropy.units as u
import numpy as np

from synthesizAR.models import RTVScalingLaws

__all__ = ['RTVInterface']


class RTVInterface:
    """
    Interface to the RTV scaling laws.

    Parameters
    ----------
    heating_rate : `~astropy.units.Quantity`, optional
        Uniform heating rate. If None, you must override the `get_heating_rate`
        method to return a loop-dependent heating rate.
    rtv_kwargs : `dict`, optional
        Keyword arguments to `synthesizAR.models.RTVScalingLaws`

    See Also
    --------
    synthesizAR.models.MartensScalingLaws :
    synthesizAR.models.RTVScalingLaws :
    """
    name = 'RTV'

    @u.quantity_input
    def __init__(self, heating_rate: (u.Unit('erg cm-3 s-1'), None)=None, rtv_kwargs=None):
        self._heating_rate = heating_rate
        self.rtv_kwargs = {} if rtv_kwargs is None else rtv_kwargs

    def load_results(self, loop, **kwargs):
        heating_rate = self.get_heating_rate(loop)
        rtv = RTVScalingLaws(loop.length/2, heating_rate=heating_rate, **self.rtv_kwargs)
        time = u.Quantity([0, ], 's')
        shape = time.shape+loop.field_aligned_coordinate_center.shape
        temperature = np.ones(shape) * rtv.max_temperature
        density = np.ones(shape) * rtv.density
        # Scaling laws do not provide any velocity information
        velocity = np.ones(shape) * np.nan * u.cm/u.s

        return {'time': time,
                'electron_temperature': temperature,
                'ion_temperature': temperature,
                'density': density,
                'velocity': velocity}

    @u.quantity_input
    def get_heating_rate(self, loop) -> u.Unit('erg cm-3 s-1'):
        return self._heating_rate
