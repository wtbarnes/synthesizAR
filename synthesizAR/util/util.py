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

__all__ = ['SpatialPair', 'is_visible']


SpatialPair = namedtuple('SpatialPair', 'x y z')


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
