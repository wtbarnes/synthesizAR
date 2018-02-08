"""
Functions for tracing, filtering, and converting fieldlines
"""
import warnings
import functools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import astropy.units as u
from astropy.coordinates import SkyCoord
import yt
from sunpy.image.rescale import resample

from synthesizAR.util import heeq_to_hcc_coord

__all__ = ['filter_streamlines', 'find_seed_points', 'trace_fieldlines', 'peek_fieldlines']


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
    loop_length_range : `~astropy.Quantity`
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
                     mask_threshold=0.05, safety=1.2, max_failures=1000):
    """
    Given a 3D extrapolated field and the corresponding magnetogram, estimate the locations of the
    seed points for the fieldline tracing through the extrapolated 3D volume.

    Parameters
    ----------
    volume : `~yt.frontends.stream.data_structures.StreamDataset`
        Dataset containing the 3D extrapolated vector field
    boundary_map : `~sunpy.map.Map`
        HMI magnetogram
    number_fieldlines : `int`
        Number of seed points
    """
    # Get lower boundary slice
    if lower_boundary is None:
        boundary = ds.r[:, :, 0]['Bz'].reshape(ds.domain_dimensions[:2]).value.T
    else:
        boundary = lower_boundary
    # mask the boundary map and estimate resampled resolution
    mask_above = mask_threshold * np.nanmin(boundary)
    masked_boundary = np.ma.masked_invalid(np.ma.masked_greater(boundary, mask_above))
    epsilon_area = float(masked_boundary.count()) / float(boundary.shape[0] * boundary.shape[1])
    resample_resolution = int(safety*np.sqrt(number_fieldlines/epsilon_area))

    # resample and mask the boundary map
    boundary_resampled = resample(boundary.T, (resample_resolution, resample_resolution),
                                  method='linear', center=True)
    boundary_resampled = boundary_resampled.T
    masked_boundary_resampled = np.ma.masked_invalid(
                                    np.ma.masked_greater(boundary_resampled, mask_above))

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


def trace_fieldlines(ds, number_fieldlines, max_tries=100, **kwargs):
    """
    Trace the fieldlines through extrapolated 3D volume
    """
    get_seed_points = kwargs.get('get_seed_points', find_seed_points)
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
        # trace fieldlines
        yt_unit = ds.domain_width / ds.domain_width.value
        streamlines = yt.visualization.api.Streamlines(ds, seed_points * yt_unit,
                                                       xfield='Bx', yfield='By', zfield='Bz',
                                                       get_magnitude=True,
                                                       direction=kwargs.get('direction', -1))
        streamlines.integrate_through_volume()
        streamlines.clean_streamlines()
        # filter
        keep_streamline = streamline_filter_wrapper(streamlines.streamlines, ds.domain_width,
                                                    **kwargs)
        if True not in keep_streamline:
            i_tries += 1
            warnings.warn(f'No acceptable streamlines found. Tries left = {max_tries - i_tries}')
            continue
        else:
            i_tries = 0
        # save strealines
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
        # Convert to proper coordinates
        coord = (heeq_to_hcc_coord(line[:, 0], line[:, 1], line[:, 2],
                                   magnetogram.observer_coordinate)
                 .transform_to(magnetogram.coordinate_frame))
        # Plot line
        ax.plot_coord(coord, '-', color=kwargs.get('color', 'k'), lw=kwargs.get('lw', 1),
                      alpha=kwargs.get('alpha', 0.5))

    plt.show()
    