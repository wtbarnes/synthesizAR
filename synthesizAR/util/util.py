"""
Some basic tools/utilities needed for active region construction. These functions are generally
peripheral to the actual physics.
"""
from collections import namedtuple
import warnings

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import astropy.units as u
from astropy.coordinates import SkyCoord
import astropy.constants as const
import sunpy.coordinates
from sunpy.sun import sun_const

__all__ = ['SpatialPair', 'is_visible', 'from_pfsspack']


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
    rsun_obs = ((sun_const.radius / (observer.radius - sun_const.radius)).decompose()
                * u.radian).to(u.arcsec)
    off_disk = np.sqrt(theta_x**2 + theta_y**2) > rsun_obs
    in_front_of_disk = distance - observer.radius < 0.

    return np.any(np.stack([off_disk, in_front_of_disk], axis=1), axis=1)


def from_pfsspack(pfss_fieldlines):
    """
    Convert fieldline coordinates output from the SSW package `pfss <http://www.lmsal.com/~derosa/pfsspack/>`_
    into `~astropy.coordinates.SkyCoord` objects.

    Parameters
    ----------
    pfss_fieldlines : `~numpy.recarray`
        Structure produced by reading pfss output with `~scipy.io.readsav`

    Returns
    -------
    fieldlines : `list`
        Each entry is a `tuple` containing a `~astropy.coordinates.SkyCoord` object and a
        `~astropy.units.Quantity` object listing the coordinates and field strength along the loop.
    """
    # Fieldline coordinates
    num_fieldlines = pfss_fieldlines['ptr'].shape[0]
    # Use HGC frame if possible
    try:
        frame = sunpy.coordinates.HeliographicCarrington(
            obstime=sunpy.time.parse_time(pfss_fieldlines['now'].decode('utf-8')))
    except ValueError:
        warnings.warn('Assuming HGS frame because no date available for HGC frame')
        frame = sunpy.coordinates.HeliographicStonyhurst()
    fieldlines = []
    for i in range(num_fieldlines):
        # NOTE: For an unknown reason, there are a number of invalid points for each line output
        # by pfss
        n_valid = pfss_fieldlines['nstep'][i]
        lon = (pfss_fieldlines['ptph'][i, :] * u.radian).to(u.deg)[:n_valid]
        lat = 90 * u.deg - (pfss_fieldlines['ptth'][i, :] * u.radian).to(u.deg)[:n_valid]
        radius = ((pfss_fieldlines['ptr'][i, :]) * const.R_sun.to(u.cm))[:n_valid]
        coord = SkyCoord(lon=lon, lat=lat, radius=radius, frame=frame)
        fieldlines.append(coord)

    # Magnetic field strengths
    lon_grid = (pfss_fieldlines['phi'] * u.radian - np.pi * u.radian).to(u.deg).value
    lat_grid = (np.pi / 2. * u.radian - pfss_fieldlines['theta'] * u.radian).to(u.deg).value
    radius_grid = pfss_fieldlines['rix'] * const.R_sun.to(u.cm).value
    B_radius = pfss_fieldlines['br']
    B_lat = pfss_fieldlines['bth']
    B_lon = pfss_fieldlines['bph']
    # Create interpolators
    B_radius_interpolator = RegularGridInterpolator((radius_grid, lat_grid, lon_grid), B_radius,
                                                    bounds_error=False, fill_value=None)
    B_lat_interpolator = RegularGridInterpolator((radius_grid, lat_grid, lon_grid), B_lat,
                                                 bounds_error=False, fill_value=None)
    B_lon_interpolator = RegularGridInterpolator((radius_grid, lat_grid, lon_grid), B_lon,
                                                 bounds_error=False, fill_value=None)
    # Interpolate values through each line
    field_strengths = []
    for f in fieldlines:
        points = np.stack([f.spherical.distance.to(u.cm).value,
                           f.spherical.lat.to(u.deg).value,
                           f.spherical.lon.to(u.deg).value], axis=1)
        b_r = B_radius_interpolator(points)
        b_lat = B_lat_interpolator(points)
        b_lon = B_lon_interpolator(points)
        field_strengths.append(np.sqrt(b_r**2 + b_lat**2 + b_lon**2) * u.Gauss)

    return [(l, b) for l, b in zip(fieldlines, field_strengths)]
