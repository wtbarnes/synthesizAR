"""
Some basic tools/utilities needed for active region construction. These functions are generally
peripheral to the actual physics.
"""
from collections import namedtuple

import numpy as np
import dask.delayed
import astropy.units as u
import sunpy.coordinates

__all__ = ['SpatialPair', 'delay_property', 'heeq_to_hcc', 'to_heeq']


SpatialPair = namedtuple('SpatialPair', 'x y z')


def delay_property(instance, attr):
    """
    Lazily evaluate a class property using Dask delayed
    """
    for obj in [instance] + instance.__class__.mro():
        if attr in obj.__dict__:
            prop = obj.__dict__[attr]
            return dask.delayed(prop.fget)(instance)
    raise AttributeError


@u.quantity_input
def heeq_to_hcc(x_heeq: u.cm, y_heeq: u.cm, z_heeq: u.cm, observer_coordinate):
    """
    Convert Heliocentric Earth Equatorial (HEEQ) coordinates to Heliocentric
    Cartesian Coordinates (HCC) for a given observer. See Eqs. 2 and 11 of [1]_.

    References
    ----------
    .. [1] Thompson, W. T., 2006, A&A, `449, 791 <http://adsabs.harvard.edu/abs/2006A%26A...449..791T>`_
    """
    Phi_0 = observer_coordinate.lon.to(u.radian)
    B_0 = observer_coordinate.lat.to(u.radian)

    x_hcc = y_heeq*np.cos(Phi_0) - x_heeq*np.sin(Phi_0)
    y_hcc = z_heeq*np.cos(B_0) - x_heeq*np.sin(B_0)*np.cos(Phi_0) - y_heeq*np.sin(Phi_0)*np.sin(B_0)
    z_hcc = z_heeq*np.sin(B_0) + x_heeq*np.cos(B_0)*np.cos(Phi_0) + y_heeq*np.cos(B_0)*np.sin(Phi_0)

    return x_hcc, y_hcc, z_hcc


def to_heeq(coord):
    """
    Transform a coordinate to HEEQ
    """
    coord = coord.transform_to(sunpy.coordinates.frames.HeliographicStonyhurst)
    phi = coord.lon.to(u.radian)
    theta = coord.lat.to(u.radian)
    radius = coord.radius
    x_heeq = radius * np.cos(theta) * np.cos(phi)
    y_heeq = radius * np.cos(theta) * np.sin(phi)
    z_heeq = radius * np.sin(theta)
    return x_heeq, y_heeq, z_heeq
