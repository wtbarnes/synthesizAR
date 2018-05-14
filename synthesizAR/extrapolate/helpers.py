"""
Helper routines for field extrapolation routines and dealing with vector field data
"""
import numpy as np
import astropy.time
import astropy.units as u
import astropy.constants as const
import yt
import sunpy.coordinates
from sunpy.util.metadata import MetaDict
from sunpy.map import GenericMap

from synthesizAR.util import to_heeq

__all__ = ['synthetic_magnetogram', 'magnetic_field_to_yt_dataset', 'local_to_heeq',
           'heeq_to_local']


@u.quantity_input
def synthetic_magnetogram(bottom_left_coord, top_right_coord, shape: u.pixel, centers,
                          sigmas: u.arcsec, amplitudes: u.Gauss, observer=None):
    """
    Compute synthetic magnetogram using 2D guassian "sunspots"
    
    Parameters
    ----------
    bottom_left_coord : `~astropy.coordinates.SkyCoord`
        Bottom left corner
    top_right_coord : `~astropy.coordinates.SkyCoord`
        Top right corner
    shape : `~astropy.units.Quantity`
        Dimensionality of the magnetogram
    centers : `~astropy.coordinates.SkyCoord`
        Center coordinates of flux concentration
    sigmas : `~astropy.units.Quantity`
        Standard deviation of flux concentration with shape `(N,2)`, with `N` the
        number of flux concentrations
    amplitudes : `~astropy.units.Quantity`
        Amplitude of flux concentration with shape `(N,)`
    observer : `~astropy.coordinates.SkyCoord`, optional
        Defaults to Earth observer at current time
    """
    time_now = astropy.time.Time.now()
    if observer is None:
        observer = sunpy.coordinates.ephemeris.get_earth(time=time_now)
    # Transform to HPC frame
    bottom_left_coord = bottom_left_coord.transform_to(
        sunpy.coordinates.Helioprojective(observer=observer))
    top_right_coord = top_right_coord.transform_to(
        sunpy.coordinates.Helioprojective(observer=observer))
    # Setup array
    delta_x = (top_right_coord.Tx - bottom_left_coord.Tx).to(u.arcsec)
    delta_y = (top_right_coord.Ty - bottom_left_coord.Ty).to(u.arcsec)
    dx = delta_x / shape[0]
    dy = delta_y / shape[1]
    data = np.zeros((int(shape[1].value), int(shape[0].value)))
    xphysical, yphysical = np.meshgrid(np.arange(shape[0].value)*shape.unit*dx,
                                       np.arange(shape[1].value)*shape.unit*dy)
    # Add sunspots
    centers = centers.transform_to(sunpy.coordinates.Helioprojective(observer=observer))
    for c, s, a in zip(centers, sigmas, amplitudes):
        xc_2 = (xphysical - (c.Tx - bottom_left_coord.Tx)).to(u.arcsec).value**2.0
        yc_2 = (yphysical - (c.Ty - bottom_left_coord.Ty)).to(u.arcsec).value**2.0
        data += a.to(u.Gauss).value * np.exp(
            - xc_2 / (2 * s[0].to(u.arcsec).value**2) - yc_2 / (2 * s[1].to(u.arcsec).value**2))
    # Build metadata
    meta = MetaDict({
        'telescop': 'synthetic_magnetic_imager',
        'instrume': 'synthetic_magnetic_imager',
        'detector': 'synthetic_magnetic_imager',
        'bunit': 'Gauss',
        'ctype1': 'HPLN-TAN',
        'ctype2': 'HPLT-TAN',
        'hgln_obs': observer.transform_to('heliographic_stonyhurst').lon.to(u.deg).value,
        'hglt_obs': observer.transform_to('heliographic_stonyhurst').lat.to(u.deg).value,
        'cunit1': 'arcsec',
        'cunit2': 'arcsec',
        'crpix1': (shape[0].value + 1)/2.,
        'crpix2': (shape[1].value + 1)/2.,
        'cdelt1': dx.value,
        'cdelt2': dy.value,
        'crval1': ((bottom_left_coord.Tx + top_right_coord.Tx)/2.).to(u.arcsec).value,
        'crval2': ((bottom_left_coord.Ty + top_right_coord.Ty)/2.).to(u.arcsec).value,
        'dsun_obs': observer.transform_to('heliographic_stonyhurst').radius.to(u.m).value,
        'dsun_ref': observer.transform_to('heliographic_stonyhurst').radius.to(u.m).value,
        'rsun_ref': const.R_sun.to(u.m).value,
        'rsun_obs': ((const.R_sun / observer.transform_to(
            'heliographic_stonyhurst').radius).decompose() * u.radian).to(u.arcsec).value,
        't_obs': time_now.iso,
        'date-obs': time_now.iso,
    })
    return GenericMap(data, meta)


@u.quantity_input
def magnetic_field_to_yt_dataset(Bx: u.gauss, By: u.gauss, Bz: u.gauss, range_x: u.cm,
                                 range_y: u.cm, range_z: u.cm):
    """
    Reshape vector magnetic field data into a yt dataset

    Parameters
    ----------
    Bx,By,Bz : `~astropy.units.Quantity`
        3D arrays holding the x,y,z components of the extrapolated field
    range_x, range_y, range_z : `~astropy.units.Quantity`
        Spatial range in the x,y,z dimensions of the grid
    """
    Bx = Bx.to(u.gauss)
    By = By.to(u.gauss)
    Bz = Bz.to(u.gauss)
    data = dict(
                Bx=(np.swapaxes(Bx.value, 0, 1), Bx.unit.to_string()),
                By=(np.swapaxes(By.value, 0, 1), By.unit.to_string()),
                Bz=(np.swapaxes(Bz.value, 0, 1), Bz.unit.to_string()))
    # Uniform, rectangular grid
    bbox = np.array([range_x.to(u.cm).value,
                     range_y.to(u.cm).value,
                     range_z.to(u.cm).value])
    return yt.load_uniform_grid(data, data['Bx'][0].shape, bbox=bbox,
                                length_unit=yt.units.cm,
                                geometry=('cartesian', ('x', 'y', 'z')))


@u.quantity_input
def local_to_heeq(x_local: u.cm, y_local: u.cm, z_local: u.cm, center):
    """
    Transform from a cartesian frame centered on the active region (with the z-axis parallel
    to the surface normal) to a HEEQ frame.
    """
    x_center, y_center, z_center = to_heeq(center)
    center = center.transform_to(sunpy.coordinates.frames.HeliographicStonyhurst)
    # NOTE: the coordinates are permuted because the local z-axis is parallel to the surface normal
    coord_heeq = (rotate_z(center.lon) @ rotate_y(-center.lat)
                  @ u.Quantity([z_local, x_local, y_local]))

    return coord_heeq[0, :] + x_center, coord_heeq[1, :] + y_center, coord_heeq[2, :] + z_center


@u.quantity_input
def heeq_to_local(x_heeq: u.cm, y_heeq: u.cm, z_heeq: u.cm, center):
    """
    Transform from HEEQ frame to a cartesian frame centered on the active region
    (with the z-axis normal to the surface).
    """
    x_center, y_center, z_center = to_heeq(center)
    center = center.transform_to(sunpy.coordinates.frames.HeliographicStonyhurst)
    x_heeq -= x_center
    y_heeq -= y_center
    z_heeq -= z_center
    coord_local = (rotate_y(center.lat) @ rotate_z(-center.lon)
                   @ u.Quantity([x_heeq, y_heeq, z_heeq]))
    # NOTE: the coordinates are permuted because the local z-axis is parallel to the surface normal
    return coord_local[1, :], coord_local[2, :], coord_local[0, :]


@u.quantity_input
def rotate_z(angle: u.radian):
    angle = angle.to(u.radian)
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])


@u.quantity_input
def rotate_y(angle: u.radian):
    angle = angle.to(u.radian)
    return np.array([[np.cos(angle), 0, np.sin(angle)],
                     [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])
