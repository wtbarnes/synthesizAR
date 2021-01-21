"""
Helper routines for field extrapolation routines and dealing with vector field data
"""
import numpy as np
import astropy.time
import astropy.units as u
from astropy.coordinates import SkyCoord
import yt
import sunpy.coordinates
from sunpy.map import GenericMap, make_fitswcs_header

__all__ = [
    'synthetic_magnetogram',
    'magnetic_field_to_yt_dataset',
    'from_local',
    'to_local',
]


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
    hpc_frame = sunpy.coordinates.Helioprojective(observer=observer, obstime=observer.obstime)
    bottom_left_coord = bottom_left_coord.transform_to(hpc_frame)
    top_right_coord = top_right_coord.transform_to(hpc_frame)
    # Setup array
    delta_x = (top_right_coord.Tx - bottom_left_coord.Tx).to(u.arcsec)
    delta_y = (top_right_coord.Ty - bottom_left_coord.Ty).to(u.arcsec)
    dx = delta_x / shape[0]
    dy = delta_y / shape[1]
    data = np.zeros((int(shape[1].value), int(shape[0].value)))
    xphysical, yphysical = np.meshgrid(np.arange(shape[0].value)*shape.unit*dx,
                                       np.arange(shape[1].value)*shape.unit*dy)
    # Add sunspots
    centers = centers.transform_to(hpc_frame)
    for c, s, a in zip(centers, sigmas, amplitudes):
        xc_2 = (xphysical - (c.Tx - bottom_left_coord.Tx)).to(u.arcsec).value**2.0
        yc_2 = (yphysical - (c.Ty - bottom_left_coord.Ty)).to(u.arcsec).value**2.0
        data += a.to(u.Gauss).value * np.exp(
            - xc_2 / (2 * s[0].to(u.arcsec).value**2)
            - yc_2 / (2 * s[1].to(u.arcsec).value**2)
        )
    # Build metadata
    meta = make_fitswcs_header(
        data,
        bottom_left_coord,
        reference_pixel=(0, 0) * u.pixel,
        scale=u.Quantity((dx, dy)),
        instrument='synthetic_magnetic_imager',
        telescope='synthetic_magnetic_imager',
    )
    meta['bunit'] = 'gauss'
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
    data = dict(Bx=(np.swapaxes(Bx.value, 0, 1), Bx.unit.to_string()),
                By=(np.swapaxes(By.value, 0, 1), By.unit.to_string()),
                Bz=(np.swapaxes(Bz.value, 0, 1), Bz.unit.to_string()))
    # Uniform, rectangular grid
    bbox = np.array([range_x.to(u.cm).value,
                     range_y.to(u.cm).value,
                     range_z.to(u.cm).value])
    return yt.load_uniform_grid(data, data['Bx'][0].shape,
                                bbox=bbox,
                                length_unit=yt.units.cm,
                                geometry=('cartesian', ('x', 'y', 'z')))


@u.quantity_input
def from_local(x_local: u.cm, y_local: u.cm, z_local: u.cm, center):
    """
    Transform from a Cartesian frame centered on the active region (with the z-axis parallel
    to the surface normal).

    Parameters
    ----------
    x_local : `~astropy.units.Quantity`
    y_local : `~astropy.units.Quantity`
    z_local : `~astropy.units.Quantity`
    center : `~astropy.coordinates.SkyCoord`
        Center of the active region

    Returns
    -------
    coord : `~astropy.coordinates.SkyCoord`
    """
    center = center.transform_to(sunpy.coordinates.frames.HeliographicStonyhurst)
    x_center, y_center, z_center = center.cartesian.xyz
    rot_zy = rotate_z(center.lon) @ rotate_y(-center.lat)
    # NOTE: the coordinates are permuted because the local z-axis is parallel to the surface normal
    coord_heeq = rot_zy @ u.Quantity([z_local, x_local, y_local])

    return SkyCoord(x=coord_heeq[0, :] + x_center,
                    y=coord_heeq[1, :] + y_center,
                    z=coord_heeq[2, :] + z_center,
                    frame=sunpy.coordinates.HeliographicStonyhurst,
                    representation_type='cartesian')


@u.quantity_input
def to_local(coord, center):
    """
    Transform coordinate to a cartesian frame centered on the active region
    (with the z-axis normal to the surface).

    Parameters
    ----------
    coord : `~astropy.coordinates.SkyCoord`
    center : `~astropy.coordinates.SkyCoord`
        Center of the active region
    """
    center = center.transform_to(sunpy.coordinates.HeliographicStonyhurst)
    x_center, y_center, z_center = center.cartesian.xyz
    xyz_heeq = coord.transform_to(sunpy.coordinates.HeliographicStonyhurst).cartesian.xyz
    if xyz_heeq.shape == (3,):
        xyz_heeq = xyz_heeq[:, np.newaxis]
    x_heeq = xyz_heeq[0, :] - x_center
    y_heeq = xyz_heeq[1, :] - y_center
    z_heeq = xyz_heeq[2, :] - z_center
    rot_yz = rotate_y(center.lat) @ rotate_z(-center.lon)
    coord_local = rot_yz @ u.Quantity([x_heeq, y_heeq, z_heeq])
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
