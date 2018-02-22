"""
Some basic tools/utilities needed for active region construction. These functions are generally
peripheral to the actual physics.
"""
from collections import namedtuple

import numpy as np
import dask.delayed
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicStonyhurst, Heliocentric
from sunpy.sun import constants

__all__ = ['SpatialPair', 'future_property', 'delay_property', 'heeq_to_hcc', 'heeq_to_hcc_coord',
           'to_heeq', 'is_visible']


SpatialPair = namedtuple('SpatialPair', 'x y z')


def future_property(instance, attr):
    """
    Return function and args to be submitted as a future
    """
    for obj in [instance] + instance.__class__.mro():
        if attr in obj.__dict__:
            prop = obj.__dict__[attr]
            return (prop.fget, instance)
    raise AttributeError


def delay_property(instance, attr):
    """
    Lazily evaluate a class property using Dask delayed
    """
    for obj in [instance] + instance.__class__.mro():
        if attr in obj.__dict__:
            prop = obj.__dict__[attr]
            return dask.delayed(prop.fget)(instance)
    raise AttributeError


def heeq_to_hcc(x_heeq, y_heeq, z_heeq, observer_coordinate):
    """
    Convert Heliocentric Earth Equatorial (HEEQ) coordinates to Heliocentric
    Cartesian Coordinates (HCC) for a given observer. See Eqs. 2 and 11 of [1]_.

    References
    ----------
    .. [1] Thompson, W. T., 2006, A&A, `449, 791 <http://adsabs.harvard.edu/abs/2006A%26A...449..791T>`_
    """
    observer_coordinate = observer_coordinate.transform_to(HeliographicStonyhurst)
    Phi_0 = observer_coordinate.lon.to(u.radian)
    B_0 = observer_coordinate.lat.to(u.radian)

    x_hcc = y_heeq*np.cos(Phi_0) - x_heeq*np.sin(Phi_0)
    y_hcc = z_heeq*np.cos(B_0) - x_heeq*np.sin(B_0)*np.cos(Phi_0) - y_heeq*np.sin(Phi_0)*np.sin(B_0)
    z_hcc = z_heeq*np.sin(B_0) + x_heeq*np.cos(B_0)*np.cos(Phi_0) + y_heeq*np.cos(B_0)*np.sin(Phi_0)

    return x_hcc, y_hcc, z_hcc


@u.quantity_input
def heeq_to_hcc_coord(x_heeq: u.cm, y_heeq: u.cm, z_heeq: u.cm, observer_coordinate):
    """
    Return an HCC `~astropy.coordinates.SkyCoord` object from a set of HEEQ positions.
    This is a wrapper around `~heeq_to_hcc`.
    """
    x, y, z = heeq_to_hcc(x_heeq, y_heeq, z_heeq, observer_coordinate)
    return SkyCoord(x=x, y=y, z=z, frame=Heliocentric(observer=observer_coordinate))


def to_heeq(coord):
    """
    Transform a coordinate to HEEQ
    """
    coord = coord.transform_to(HeliographicStonyhurst)
    phi = coord.lon.to(u.radian)
    theta = coord.lat.to(u.radian)
    radius = coord.radius
    x_heeq = radius * np.cos(theta) * np.cos(phi)
    y_heeq = radius * np.cos(theta) * np.sin(phi)
    z_heeq = radius * np.sin(theta)
    return x_heeq, y_heeq, z_heeq


def is_visible(coords, observer):
    """
    Create mask of coordinates not blocked by the solar disk.

    Parameters
    ----------
    coords : `~astropy.coordinates.SkyCoord`
        Helioprojective oordinates of the object(s) of interest
    observer : `~astropy.coordinates.SkyCoord`
        Heliographic-Stonyhurst Location of the observer
    """
    theta_x = coords.Tx
    theta_y = coords.Ty
    distance = coords.distance
    rsun_obs = ((constants.radius / (observer.radius - constants.radius)).decompose()
                * u.radian).to(u.arcsec)
    off_disk = np.sqrt(theta_x**2 + theta_y**2) > rsun_obs
    in_front_of_disk = distance - observer.radius < 0.

    return np.any(np.stack([off_disk, in_front_of_disk], axis=1), axis=1)
