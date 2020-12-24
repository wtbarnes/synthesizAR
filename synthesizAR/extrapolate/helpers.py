"""
Helper routines for field extrapolation routines and dealing with vector field data
"""
import warnings

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import astropy.time
import astropy.units as u
from astropy.coordinates import SkyCoord
import astropy.constants as const
import yt
import sunpy.coordinates
from sunpy.map import GenericMap, make_fitswcs_header

__all__ = ['from_pfsspack', 'semi_circular_loop', 'synthetic_magnetogram',
           'magnetic_field_to_yt_dataset', 'from_local', 'to_local']


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


@u.quantity_input
def semi_circular_loop(length: u.cm,
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
    observer : `~astropy.coordinates.SkyCoord`, optional
        Observer that defines te HCC coordinate system. Effectively, this is the
        coordinate of the midpoint of the loop.
    obstime : parsable by `~astropy.time.Time`, optional
        Observation time of the HCC frame. If `None`, will default to the `obstime`
        of the `observer`.
    n_points : `int`, optional
        Number of points in the coordinate
    offset : `~astropy.units.Quantity`
        Offset in the direction perpendicular to the arcade
    gamma : `~astropy.units.Quantity`
        Orientation of the arcade relative to the HCC y-axis
    """
    # Calculate a semi-circular loop
    s = np.linspace(0, length, n_points)
    z = length / np.pi * np.sin(np.pi * u.rad * s/length)
    x = np.sqrt(length**2 / np.pi**2 - z**2)
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
    origin = SkyCoord(x=0*u.km, y=0*u.km, z=const.R_sun, frame=hcc_frame)
    # Offset along the y-axis, convenient for creating loop arcades
    offset = offset*np.ones(s.shape)
    return SkyCoord(x=offset * np.sin(gamma) + x * np.cos(gamma) + origin.x,
                    y=offset * np.cos(gamma) + x * np.sin(gamma) + origin.y,
                    z=z + origin.z,
                    frame=origin.frame)


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
