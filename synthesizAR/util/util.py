"""
Some basic tools/utilities needed for active region construction. These functions are generally
peripheral to the actual physics.
"""
import astropy.constants as const
import astropy.units as u
import numpy as np
import sunpy.coordinates
import sunpy.sun.constants as sun_const
import warnings

from astropy.coordinates import SkyCoord
from collections import namedtuple
from scipy.interpolate import RegularGridInterpolator

import synthesizAR

__all__ = [
    'SpatialPair',
    'los_velocity',
    'coord_in_fov',
    'find_minimum_fov',
    'is_visible',
    'from_pfsspack',
    'from_pfsspy',
    'change_obstime',
    'change_obstime_frame',
    'power_law_transform',
]


SpatialPair = namedtuple('SpatialPair', 'x y z')


@u.quantity_input
def los_velocity(v_xyz: u.cm/u.s, observer):
    """
    Compute the LOS velocity for some observing angle. The sign of the result
    is consistent with the convention that LOS velocity is :math:`>0` away from
    the observer, i.e. red shifts are positive, blue shifts are negative

    Parameters
    ----------
    v_xyz : `~astropy.units.Quantity`
        Cartesian velocity components in the Heliographic Stonyhurst coordinate
        system, with shape ``(3,...)``
    observer : `~astropy.coordinates.SkyCoord`
        Heliographic Stonyhurst observer coordinate
    """
    # NOTE: transform from HEEQ to HCC with respect to the instrument observer
    Phi_0 = observer.lon.to(u.radian)
    B_0 = observer.lat.to(u.radian)
    v_los = v_xyz[2]*np.sin(B_0) + v_xyz[0]*np.cos(B_0)*np.cos(Phi_0) + v_xyz[1]*np.cos(B_0)*np.sin(Phi_0)
    return -v_los


def coord_in_fov(coord, width, height, center=None, bottom_left_corner=None):
    # NOTE: this does not work for frames other than HPC
    if center is None and bottom_left_corner is None:
        raise ValueError('Must specify either center or bottom left corner')
    if bottom_left_corner is None:
        bottom_left_corner = SkyCoord(Tx=center.Tx-width/2,
                                      Ty=center.Ty-height/2,
                                      frame=center.frame)
    coord = coord.transform_to(bottom_left_corner.frame)
    top_right_corner = SkyCoord(Tx=bottom_left_corner.Tx+width,
                                Ty=bottom_left_corner.Ty+height,
                                frame=bottom_left_corner.frame)
    in_x = np.logical_and(coord.Tx > bottom_left_corner.Tx, coord.Tx < top_right_corner.Tx)
    in_y = np.logical_and(coord.Ty > bottom_left_corner.Ty, coord.Ty < top_right_corner.Ty)
    return np.logical_and(in_x, in_y)


def find_minimum_fov(coordinates, padding=None):
    """
    Given an HPC coordinate, find the coordinates of the corners of the
    FOV that includes all of the coordinates.
    """
    if padding is None:
        padding = [0, 0] * u.arcsec
    Tx = coordinates.Tx
    Ty = coordinates.Ty
    bottom_left_corner = SkyCoord(
        Tx=Tx.min() - padding[0],
        Ty=Ty.min() - padding[1],
        frame=coordinates.frame
    )
    delta_x = Tx.max() + padding[0] - bottom_left_corner.Tx
    delta_y = Ty.max() + padding[1] - bottom_left_corner.Ty
    # Compute right corner after the fact to account for rounding in bin numbers
    # NOTE: this is the coordinate of the top right corner of the top right corner pixel, NOT
    # the coordinate at the center of the pixel!
    top_right_corner = SkyCoord(
        Tx=bottom_left_corner.Tx + delta_x,
        Ty=bottom_left_corner.Ty + delta_y,
        frame=coordinates.frame
    )
    return bottom_left_corner, top_right_corner


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


def change_obstime(coord, obstime):
    """
    Change the obstime of a coordinate, including its observer.
    """
    new_observer = coord.observer.replicate(obstime=obstime)
    return SkyCoord(coord.replicate(observer=new_observer, obstime=obstime))


def change_obstime_frame(frame, obstime):
    """
    Change the obstime of a coordinate frame, including its observer.
    """
    new_observer = frame.observer.replicate(obstime=obstime)
    return frame.replicate_without_data(observer=new_observer, obstime=obstime)


@u.quantity_input
def from_pfsspy(fieldlines,
                n_min=0,
                obstime=None,
                length_min=20*u.Mm,
                length_max=3e3*u.Mm,
                name_template=None,
                cross_sectional_area=None):
    """
    Convert a `pfsspy.fieldline.FieldLines` structure into a list of `~synthesizAR.Strand` objects.

    Parameters
    ----------
    fieldlines: `pfsspy.fieldline.FieldLines`
    n_min: `int`, optional
        The minimum number of points required to keep a traced fieldline. This is often useful when
        trying to filter out very small or erroneous fieldlines.
    obstime: `astropy.time.Time`, optional
        The desired obstime of the coordinates. Use this if the coordinates need to be at a
        different obstime than that of the Carrington map they were traced from.
    length_min : `astropy.units.Quantity`, optional
        Minimum allowed loop length. All strands with length below this are excluded.
    length_max : `astropy.units.Quantity`, optional
        Maximum allowed loop length. All strands with length above this are excluded.
    name_template: `str`, optional
        Name template to use when building strands. Defaults to 'strand_{:06d}'
    cross_sectional_area: `astropy.units.Quantity`, optional
        The cross-sectional area to assign to each loop.
    """
    from synthesizAR import log
    if name_template is None:
        name_template = 'strand_{:06d}'
    strands = []
    for i, f in enumerate(fieldlines):
        if f.coords.shape[0] <= n_min:
            log.debug(f'Dropping {f} as it has less than {n_min} points.')
            continue
        # NOTE: There are some lines along which we cannot find
        # the field strength.
        try:
            if np.isnan(f.b_along_fline).all():
                log.debug(f'Dropping {f} as field strength along strand is all NaN.')
                continue
        except IndexError:
            # TODO: remember why this exception exists.
            continue
        b = np.sqrt((f.b_along_fline**2).sum(axis=1)) * u.G
        # NOTE: redefine the coordinate at a new obstime. This is useful because the
        # Carrington map that the coordinates were derived from has a single time for
        # the entire Carrington rotation, but this is often not the time at which we
        # need the coordinates defined. If deriving coordinates for a relatively small
        # (e.g. AR-sized) patch on the Sun, this time should roughly correspond to the time
        # at which the center of that patch crossed the central meridian.
        if obstime is not None:
            coord = change_obstime(f.coords, obstime)
        else:
            coord = f.coords
        # Construct the loop here to easily filter on loop length and interpolate
        # NaNs from the field strength.
        strand = synthesizAR.Strand(name_template.format(i),
                                coord,
                                field_strength=b,
                                cross_sectional_area=cross_sectional_area)
        if strand.length < length_min or strand.length > length_max:
            log.debug(f'Dropping {strand} as it has length {strand.length} outside of the allowed range.')
            continue
        if np.any(np.isnan(strand.field_strength)):
            # There are often NaN values that show up in the interpolated field strengths.
            # Interpolate over these.
            b = strand.field_strength
            s = strand.field_aligned_coordinate
            nans = np.isnan(b)
            b[nans] = np.interp(s[nans].value, s[~nans].value, b[~nans].value) * b.unit
            strand.field_strength = b
        strands.append(strand)

    return strands


def power_law_transform(x, a0, a1, alpha):
    """
    Transform uniform distribution to a power-law distribution
    with an upper and lower bound.

    Parameters
    ----------
    x : array-like
        Uniform distribution
    a0 : `float`
        Lower bound on power-law distribution
    a1 : `float`
        Upper bound on power-law distribution
    alpha : `float`
        Index of the power-law distribution
    """
    return ((a1**(alpha + 1.) - a0**(alpha + 1.))*x + a0**(alpha + 1.))**(1./(alpha + 1.))
