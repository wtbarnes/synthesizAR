"""
Interface between loop object and RTV scaling laws
"""
import numpy as np
import astropy.units as u

from synthesizAR.physics import RTVScalingLaws

__all__ = ['RTVInterface']


class RTVInterface(object):
    """
    Interface to the RTV scaling laws.

    Parameters
    ----------
    plasma_beta : `float`, optional
        Ratio of plasma to magnetic pressure
    rtv_kwargs : `dict`, optional
        Keyword arguments to `synthesizAR.physics.RTVScalingLaws`

    See Also
    --------
    synthesizAR.physics.MartensScalingLaws :
    synthesizAR.physics.RTVScalingLaws :
    """

    def __init__(self, plasma_beta=0.001, rtv_kwargs=None):
        self.plasma_beta = plasma_beta
        self.rtv_kwargs = {} if rtv_kwargs is None else rtv_kwargs

    def load_results(self, loop):
        pressure = loop.field_strength.mean().to(u.G).value**2 / 8 / np.pi * self.plasma_beta
        pressure = pressure * u.dyne / (u.cm**2)
        rtv = RTVScalingLaws(loop.length/2, pressure=pressure, **self.rtv_kwargs)
        time = u.Quantity([0, ], 's')
        shape = time.shape+loop.field_aligned_coordinate.shape
        temperature = np.ones(shape) * rtv.max_temperature
        density = np.ones(shape) * rtv.density
        # Scaling laws do not provide any velocity information
        velocity = np.ones(shape) * np.nan * u.cm/u.s

        return time, temperature, temperature, density, velocity
