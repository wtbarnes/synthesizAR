"""
Helper routines for field extrapolation routines and dealing with vector field data
"""
import astropy.constants as const
import astropy.units as u
import numpy as np
import sunpy.coordinates

from astropy.coordinates import SkyCoord

__all__ = ['semi_circular_loop', 'semi_circular_bundle', 'semi_circular_arcade']


@u.quantity_input
def semi_circular_loop(length: u.cm=None,
                       s: u.cm=None,
                       observer=None,
                       obstime=None,
                       n_points=1000,
                       offset: u.cm = None,
                       gamma: u.deg = 0*u.deg,
                       inclination: u.deg = 0*u.deg,
                       ellipticity=0,
                       correct_z=True):
    """
    Generate coordinates for a semi-circular loop

    Parameters
    ----------
    length : `~astropy.units.Quantity`
        Full length of the loop
    s : `~astropy.units.Quantity`
        Field-aligned coordinate. If specifying `s` directly, do not specify `length`
    observer : `~astropy.coordinates.SkyCoord`, optional
        Observer that defines te HCC coordinate system. Effectively, this is the
        coordinate of the midpoint of the loop.
    obstime : parsable by `~astropy.time.Time`, optional
        Observation time of the HCC frame. If `None`, will default to the `obstime`
        of the `observer`.
    n_points : `int`, optional
        Number of points in the coordinate. Only used if `s` is not specified.
    offset : `~astropy.units.Quantity`, optional
        Offset in the direction perpendicular and parallel to loop, convenient for simulating
        arcade geometries. This should always be of length 2.
    gamma : `~astropy.units.Quantity`, optional
        Angle between the loop axis and the HCC x-axis. This defines the orientation
        of the loop in the HCC x-y plane. `gamma=0` corresponds to a loop who's
        axis is perpendicular to the HCC y-axis.
    inclination : `~astropy.units.Quantity`, optional
        Angle between the HCC z-axis and the loop plane. An inclination of 0 corresponds
        to a loop that extends vertically only in the z-direction while an inclination
        of 90 degrees corresponds to a loop that lies entirely in the HCC x-y plane.
    ellipticity : `float`, optional
        Must be between -1 and +1. If > 0, the loop will be "tall and skinny" and if
        < 0, the loop will be "short and fat". Note that if this value is nonzero,
        ``length`` is no longer the actual loop length because the loop is no longer
        semi-circular.
    correct_z : `bool`, optional
        If True, correct the z-coordinate such that the footpoint lies on the surface of the
        Sun. This does not correct the coordinate such that the strand is normal to the surface
        at the footpoint.
    """
    if s is None and length is None:
        raise ValueError('Must specify field-aligned coordinate or loop length')
    if s is None:
        angles = np.linspace(0, 1, n_points) * np.pi * u.rad
    elif length is None:
        length = s[-1]
        angles = (s / length).decompose() * np.pi * u.rad
    else:
        raise ValueError('Specify either length or field-aligned coordinate but not both.')
    z = length / np.pi * np.sin(angles)
    x = -length / np.pi * np.cos(angles)  # add negative sign so that s=0 is the left foot point
    # Account for possible ellipticity. Note that the distance between the footpoints and the height
    # will never be less than the semi-circular case, but it may be greater
    elliptical_factor = 1.0 / np.sqrt(1 - ellipticity**2)
    if ellipticity >= 0:
        z *= elliptical_factor
    else:
        x *= elliptical_factor
    # Define origin in HCC coordinates such that the midpoint of the loop
    # is centered on the origin at the solar surface
    if observer is None:
        observer = SkyCoord(lon=0*u.deg,
                            lat=0*u.deg,
                            frame=sunpy.coordinates.HeliographicStonyhurst)
    hcc_frame = sunpy.coordinates.Heliocentric(
        observer=observer,
        obstime=observer.obstime if obstime is None else obstime,
    )
    if offset is None:
        offset = [0, 0] * u.cm
    x_hcc = -(offset[1] + z * np.sin(inclination)) * np.sin(gamma) + (x + offset[0]) * np.cos(gamma)
    y_hcc = (offset[1] + z * np.sin(inclination)) * np.cos(gamma) + (x + offset[0]) * np.sin(gamma)
    z_hcc = z * np.cos(inclination) + const.R_sun
    if correct_z:
        z_hcc -= const.R_sun - np.sqrt(const.R_sun**2 - x_hcc**2 - y_hcc**2)
    return SkyCoord(x=x_hcc, y=y_hcc, z=z_hcc, frame=hcc_frame)


@u.quantity_input
def semi_circular_bundle(length: u.cm, radius: u.cm, num_strands, **kwargs):
    """
    Generate a cylindrical bundle of semi-circular strands.

    Parameters
    ----------
    length : `~astropy.units.Quantity`
        Nominal full loop length
    radius : `~astropy.units.Quantity`
        Cross-sectional radius
    num_strands : `int`
        Number of strands in the bundle

    See Also
    ---------
    semi_circular_loop
    """
    # Randomly sample points around the right footpoint within radius
    # See https://stackoverflow.com/a/50746409/4717663
    r = np.sqrt(np.random.random_sample(size=num_strands)) * radius
    theta = np.random.random_sample(size=num_strands) * 2 * np.pi * u.radian
    # The resulting X Cartesian coordinate is the difference between the length
    # and nominal length
    lengths = length + np.pi * r * np.cos(theta)
    # The resulting Y Cartesian coordinate is the offset from the HCC origin
    offset = u.Quantity([np.zeros(r.shape)*r.unit, r*np.sin(theta)]).T
    return [semi_circular_loop(length=l, offset=o, **kwargs) for l, o in zip(lengths, offset)]


@u.quantity_input
def semi_circular_arcade(length: u.cm, width: u.deg, num_strands, observer, **kwargs):
    """
    Generate an arcade of `num_loops` of full length `length` evenly spaced in
    over an HGS angular width of `width` centered on a location specified by
    `observer`

    Parameters
    ----------
    length : `~astropy.units.Quantity`
        Full-length of each strand in the arcade
    width : `~astropy.units.Quantity`
        Angular width of the arcade. This can either be in the latitudinal or
        longitudinal direction depending on the orientation of the loops as
        specified by `gamma`
    num_strands : `int`
        Number of strands in the arcade
    observer : `~astropy.coordinates.SkyCoord`
        Observer that defines the "center" loop. Offsets equally-spaced between
        `-width/2` and `width/2` are calculated relative to this observer.

    See Also
    --------
    semi_circular_loop
    """
    hcc_frame = sunpy.coordinates.Heliocentric(observer=observer, obstime=observer.obstime)
    gamma = kwargs.pop('gamma', 90*u.deg)
    inclination = kwargs.pop('inclination', 0*u.deg)
    offsets = np.linspace(-width/2, width/2, num_strands)
    inclinations = np.linspace(-inclination, inclination, num_strands)
    strands = []
    for o, i in zip(offsets, inclinations):
        obs = SkyCoord(lon=observer.lon - o*np.sin(gamma),
                       lat=observer.lat + o*np.cos(gamma),
                       radius=observer.radius,
                       frame=observer.frame)
        s = semi_circular_loop(length=length, observer=obs, gamma=gamma, inclination=i, **kwargs)
        strands.append(s.transform_to(hcc_frame))
    return strands
