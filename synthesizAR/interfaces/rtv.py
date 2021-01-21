"""
Interface between loop object and RTV scaling laws
"""
import numpy as np
import astropy.units as u

from synthesizAR.models import RTVScalingLaws

__all__ = ['RTVInterface']


class RTVInterface(object):
    """
    Interface to the RTV scaling laws.

    Parameters
    ----------
    plasma_beta : `float`, optional
        Ratio of plasma to magnetic pressure
    rtv_kwargs : `dict`, optional
        Keyword arguments to `synthesizAR.models.RTVScalingLaws`

    See Also
    --------
    synthesizAR.models.MartensScalingLaws :
    synthesizAR.models.RTVScalingLaws :
    """
    name = 'RTV'

    def __init__(self, heating_model, rtv_kwargs=None):
        self.heating_model = heating_model
        self.rtv_kwargs = {} if rtv_kwargs is None else rtv_kwargs

    def load_results(self, loop):
        heating_rate = self.heating_model.get_heating_rate(loop)
        rtv = RTVScalingLaws(loop.length/2, heating_rate=heating_rate, **self.rtv_kwargs)
        time = u.Quantity([0, ], 's')
        shape = time.shape+loop.field_aligned_coordinate_center.shape
        temperature = np.ones(shape) * rtv.max_temperature
        density = np.ones(shape) * rtv.density
        # Scaling laws do not provide any velocity information
        velocity = np.ones(shape) * np.nan * u.cm/u.s

        return time, temperature, temperature, density, velocity
