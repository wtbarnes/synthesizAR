"""
Helper routines for field extrapolation routines and dealing with vector field data
"""
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import astropy.constants as const
import sunpy.coordinates

__all__ = ['semi_circular_loop', 'semi_circular_bundle']


@u.quantity_input
def semi_circular_loop(length: u.cm=None,
                       s: u.cm=None,
                       observer=None,
                       obstime=None,
                       n_points=1000,
                       offset: u.cm = 0*u.cm,
                       gamma: u.deg = 0*u.deg):
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
    offset : `~astropy.units.Quantity`
        Offset in the direction perpendicular to the arcade
    gamma : `~astropy.units.Quantity`
        Orientation of the arcade relative to the HCC y-axis
    """
    if s is None and length is None:
        raise ValueError('Must specify field-aligned coordinate or loop length')
    if s is None:
        s = np.linspace(0, length, n_points)
    elif length is None:
        length = s[-1]
    else:
        raise ValueError('Specify either length or field-aligned coordinate but not both.')
    z = length / np.pi * np.sin(np.pi * u.rad * s/length)
    x = np.sqrt((length / np.pi)**2 - z**2)
    x = np.where(s < length/2, -x, x)
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
    # Offset along the y-axis, convenient for creating loop arcades
    return SkyCoord(x=-offset * np.sin(gamma) + x * np.cos(gamma),
                    y=offset * np.cos(gamma) + x * np.sin(gamma),
                    z=z + const.R_sun,
                    frame=hcc_frame)


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
    length_max = length + np.pi*radius
    length_min = length - np.pi*radius
    lengths = np.random.random_sample(size=num_strands) * (length_max - length_min) + length_min
    max_offset = np.sqrt(radius**2 - (1/np.pi * (lengths - length))**2)
    offset = np.random.random_sample(size=num_strands)*2*max_offset - max_offset
    return [semi_circular_loop(length=l, offset=o, **kwargs) for l,o in zip(lengths, offset)]
