"""
Functions for tracing, filtering, and converting fieldlines
"""
import warnings
import functools

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicStonyhurst, HeliographicCarrington
from sunpy.image.rescale import resample
import sunpy.time
import yt

from synthesizAR.util import is_visible

__all__ = ['filter_streamlines', 'find_seed_points', 'trace_fieldlines', 'peek_fieldlines',
           'from_pfsspack']


@u.quantity_input
def filter_streamlines(streamline, domain_width, close_threshold=0.05,
                       loop_length_range: u.cm =[2.e+9, 5.e+10]*u.cm, **kwargs):
    """
    Check extracted loop to make sure it fits given criteria. Return True if it passes.

    Parameters
    ----------
    streamline : yt streamline object
    close_threshold : `float`
        percentage of domain width allowed between loop endpoints
    loop_length_range : `~astropy.units.Quantity`
        minimum and maximum allowed loop lengths (in centimeters)
    """
    streamline = streamline[np.all(streamline != 0.0, axis=1)]
    loop_length = np.sum(np.linalg.norm(np.diff(streamline, axis=0), axis=1))
    if np.fabs(streamline[0, 2] - streamline[-1, 2]) > close_threshold*domain_width[2]:
        return False
    elif (loop_length > loop_length_range[1].to(u.cm).value
          or loop_length < loop_length_range[0].to(u.cm).value):
        return False
    else:
        return True


def find_seed_points(ds, number_fieldlines, lower_boundary=None, preexisting_seeds=None,
                     mask_threshold=0.1, safety=2, max_failures=1000):
    """
    Given a 3D extrapolated field and the corresponding magnetogram, estimate the locations of the
    seed points for the fieldline tracing through the extrapolated 3D volume.

    Parameters
    ----------
    ds : `~yt.frontends.stream.data_structures.StreamDataset`
        Dataset containing the 3D extrapolated vector field
    number_fieldlines : `int`
        Number of seed points
    lower_boundary : `numpy.ndarray`, optional
        Array to search for seed points on
    preexisting_seeds : `list`, optional
        If a seed point is in this list, it is thrown out
    mask_threshold : `float`, optional
        Fraction of the field strength below (above) which the field is masked. Should be between -1 
        and 1. A value close to +1(-1) means the seed points will be concentrated in areas of more
        positive (negative) field.
    safety : `float`
        Ensures the boundary is not resampled to impossibly small resolutions
    max_failures : `int`
    """
    # Get lower boundary slice
    if lower_boundary is None:
        boundary = ds.r[:, :, 0]['Bz'].reshape(ds.domain_dimensions[:2]).value.T
    else:
        boundary = lower_boundary
    # mask the boundary map and estimate resampled resolution
    if mask_threshold < 0:
        mask_val = np.fabs(mask_threshold) * np.nanmin(boundary)
        mask_func = np.ma.masked_greater
    else:
        mask_val = mask_threshold * np.nanmax(boundary)
        mask_func = np.ma.masked_less
    masked_boundary = np.ma.masked_invalid(mask_func(boundary, mask_val))
    epsilon_area = float(masked_boundary.count()) / float(boundary.shape[0] * boundary.shape[1])
    resample_resolution = int(safety*np.sqrt(number_fieldlines/epsilon_area))

    # resample and mask the boundary map
    boundary_resampled = resample(boundary.T, (resample_resolution, resample_resolution),
                                  method='linear', center=True)
    boundary_resampled = boundary_resampled.T
    masked_boundary_resampled = np.ma.masked_invalid(mask_func(boundary_resampled, mask_val))

    # find the unmasked indices
    unmasked_indices = [(ix, iy) for iy, ix in zip(*np.where(masked_boundary_resampled.mask == 0))]

    if len(unmasked_indices) < number_fieldlines:
        raise ValueError('Requested number of seed points too large. Increase safety factor.')

    x_pos = np.linspace(ds.domain_left_edge[0].value, ds.domain_right_edge[0].value,
                        resample_resolution)
    y_pos = np.linspace(ds.domain_left_edge[1].value, ds.domain_right_edge[1].value,
                        resample_resolution)

    # choose seed points
    seed_points = []
    if preexisting_seeds is None:
        preexisting_seeds = []
    i_fail = 0
    z_pos = ds.domain_left_edge.value[2]
    while len(seed_points) < number_fieldlines and i_fail < max_failures:
        choice = np.random.randint(0, len(unmasked_indices))
        ix, iy = unmasked_indices[choice]
        _tmp = [x_pos[ix], y_pos[iy], z_pos]
        if _tmp not in preexisting_seeds:
            seed_points.append(_tmp)
            i_fail = 0
        else:
            i_fail += 1
        del unmasked_indices[choice]

    if i_fail == max_failures:
        raise ValueError(f'''Could not find desired number of seed points within failure tolerance of
                             {max_failures}. Try increasing safety factor or the mask threshold''')

    return seed_points


def trace_fieldlines(ds, number_fieldlines, max_tries=100, get_seed_points=None, direction=1,
                     **kwargs):
    """
    Trace lines of constant potential through a 3D magnetic field volume.

    Given a YT dataset containing a 3D vector magnetic field, trace a number of streamlines
    through the volume. This function also accepts any of the keyword arguments that can
    be passed to `~synthesizAR.extrapolate.find_seed_points` and
    `~synthesizAR.extrapolate.filter_streamlines`.

    Parameters
    ----------
    ds : `~yt.frontends.stream.data_structures.StreamDataset`
        Dataset containing the 3D extrapolated vector field
    number_fieldlines : `int`
    max_tries : `int`, optional
    get_seed_points : function, optional
        Function that returns a list of seed points
    direction : `int`, optional
        Use +1 to trace from positive to negative field and -1 to trace from negative to positive
        field

    Returns
    -------
    fieldlines : `list`
    """
    get_seed_points = find_seed_points if get_seed_points is None else get_seed_points
    # wrap the streamline filter method so we can pass a loop length range to it
    streamline_filter_wrapper = np.vectorize(filter_streamlines,
                                             excluded=[1]+list(kwargs.keys()))
    fieldlines = []
    seed_points = []
    i_tries = 0
    while len(fieldlines) < number_fieldlines and i_tries < max_tries:
        remaining_fieldlines = number_fieldlines - len(fieldlines)
        seed_points = get_seed_points(ds, remaining_fieldlines,
                                      lower_boundary=kwargs.get('lower_boundary', None),
                                      preexisting_seeds=seed_points,
                                      mask_threshold=kwargs.get('mask_threshold', 0.1),
                                      safety=kwargs.get('safety', 2.))
        yt_unit = ds.domain_width / ds.domain_width.value
        streamlines = yt.visualization.api.Streamlines(ds, seed_points * yt_unit,
                                                       xfield='Bx', yfield='By', zfield='Bz',
                                                       get_magnitude=True,
                                                       direction=direction)
        # FIXME: The reason for this try-catch is that occasionally a streamline will fall out of
        # bounds of the yt volume and the tracing will fail. We can just ignore this and move on
        # This is probably an issue that should be reported upstream to yt as I'm not really sure
        # what this problem is here.
        try:
            streamlines.integrate_through_volume()
        except AssertionError:
            i_tries += 1
            warnings.warn(f'Streamlines out of bounds. Tries left = {max_tries - i_tries}')
            continue
        streamlines.clean_streamlines()
        keep_streamline = streamline_filter_wrapper(streamlines.streamlines, ds.domain_width,
                                                    **kwargs)
        if True not in keep_streamline:
            i_tries += 1
            warnings.warn(f'No acceptable streamlines found. Tries left = {max_tries - i_tries}')
            continue
        else:
            i_tries = 0
        fieldlines += [(stream[np.all(stream != 0.0, axis=1)], mag[np.all(stream != 0.0, axis=1)])
                       for stream, mag, keep in zip(streamlines.streamlines,
                                                    streamlines.magnitudes,
                                                    keep_streamline) if keep]

    if i_tries == max_tries:
        warnings.warn(f'Maxed out number of tries with {len(fieldlines)} acceptable streamlines')

    return fieldlines


def peek_fieldlines(magnetogram, fieldlines, **kwargs):
    """
    Quick plot of streamlines overplotted on magnetogram

    Parameters
    ----------
    magnetogram : `~sunpy.map.Map`
    fieldlines : `list`
    """
    fig = plt.figure(figsize=kwargs.get('figsize', (8, 8)))
    ax = fig.gca(projection=magnetogram)
    # Plot map
    norm = kwargs.get('norm', Normalize(vmin=-1.5e3, vmax=1.5e3))
    magnetogram.plot(axes=ax, title=False, cmap=kwargs.get('cmap', 'hmimag'), norm=norm)
    # Grid
    ax.grid(alpha=0.)
    magnetogram.draw_grid(axes=ax, grid_spacing=10*u.deg, alpha=0.75, color='k')
    # Lines
    line_frequency = kwargs.get('line_frequency', 5)
    for line in fieldlines[::line_frequency]:
        try:
            coord = line.transform_to(magnetogram.coordinate_frame)
        except AttributeError:
            # This try-catch is due to a bug where to convert out of an HEEQ frame
            # one must first transform to a polar HGS frame
            # FIXME:  once this is fixed upstream in SunPy, this can be removed
            coord = line.transform_to(HeliographicStonyhurst).transform_to(
                magnetogram.coordinate_frame)
        # Mask lines behind the solar disk
        i_visible = np.where(is_visible(coord, magnetogram.observer_coordinate))
        coord_visible = SkyCoord(Tx=coord.Tx[i_visible], Ty=coord.Ty[i_visible],
                                 distance=coord.distance[i_visible],
                                 frame=magnetogram.coordinate_frame)
        ax.plot_coord(coord_visible, '-', color=kwargs.get('color', 'k'), lw=kwargs.get('lw', 1),
                      alpha=kwargs.get('alpha', 0.5))

    plt.show()


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
    fieldlines = []
    for i in range(num_fieldlines):
        # NOTE: For an unknown reason, there are a number of invalid points for each line output
        # by pfss
        n_valid = pfss_fieldlines['nstep'][i]
        lon = (pfss_fieldlines['ptph'][i, :] * u.radian).to(u.deg)[:n_valid]
        lat = 90 * u.deg - (pfss_fieldlines['ptth'][i, :] * u.radian).to(u.deg)[:n_valid]
        radius = ((pfss_fieldlines['ptr'][i, :]) * const.R_sun.to(u.cm))[:n_valid]
        coord = SkyCoord(
            lon=lon, lat=lat, radius=radius,
            frame=HeliographicCarrington(
                obstime=sunpy.time.parse_time(pfss_fieldlines['now'].decode('utf-8'))))
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
