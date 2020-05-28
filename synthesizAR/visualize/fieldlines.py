"""
Visualizaition functions related to 1D fieldlines
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize
from sunpy.map import GenericMap, make_fitswcs_header
from sunpy.coordinates import Helioprojective
from sunpy.coordinates.ephemeris import get_earth

from synthesizAR.util import is_visible

__all__ = ['plot_fieldlines', 'peek_projections']


def plot_fieldlines(*coords, magnetogram=None, observer=None, check_visible=True, **kwargs):
    """
    Plot fieldlines on the surface of the Sun

    Parameters
    ----------
    coords : `~astropy.coordinates.SkyCoord`
    magnetogram : `~sunpy.map.Map`, optional
        The LOS magnetic field map to overplot the field lines on top of. Useful
        when the field lines are derived from a magnetic field extrapolation.
    observer : `~astropy.coordinates.SkyCoord`, optional
        Position of the observer. If None, defaults to position of Earth at the
        current time. If `magnetogram` is specified, this argument has no effect
        and the observer will be the observer as defined by the HPC coordinate
        frame of the `magnetogram`.
    check_visible : `bool`
        If True, mask coordinates that are obscured by the solar disk.

    Other Parameters
    ----------------
    plot_kwargs : `dict`
        Additional parameters to pass to `~matplotlib.pyplot.plot` when
        drawing field lines.
    grid_kwargs : `dict`
        Additional parameters to pass to `~sunpy.map.Map.draw_grid`
    imshow_kwargs : `dict`
        Additional parameters to pass to `~sunpy.map.Map.plot`

    See Also
    --------
    synthesizAR.util.is_visible
    """
    plot_kwargs = {'color': 'k', 'lw': 1}
    grid_kwargs = {'grid_spacing': 10*u.deg, 'color': 'k', 'alpha': 0.75}
    imshow_kwargs = {'title': False}
    plot_kwargs.update(kwargs.get('plot_kwargs', {}))
    grid_kwargs.update(kwargs.get('grid_kwargs', {}))
    if magnetogram is None:
        # If no magnetogram is given, create a dummy transparent map for some specified
        # observer location
        data = np.ones((10, 10))
        observer = get_earth(Time.now()) if observer is None else observer
        coord = SkyCoord(Tx=0*u.arcsec,
                         Ty=0*u.arcsec,
                         frame=Helioprojective(observer=observer, obstime=observer.obstime))
        meta = make_fitswcs_header(data, coord, scale=(1, 1)*u.arcsec/u.pixel,)
        magnetogram = GenericMap(data, meta)
        # Show the full disk, make the dummy map transparent
        imshow_kwargs.update({'alpha': 0, 'extent': [-1000, 1000, -1000, 1000]})
    else:
        imshow_kwargs.update({'cmap': 'hmimag', 'norm': ImageNormalize(vmin=-1.5e3, vmax=1.5e3)})
    imshow_kwargs.update(kwargs.get('imshow_kwargs', {}))
    # Plot coordinates
    fig = kwargs.get('fig', plt.figure(figsize=kwargs.get('figsize', None)))
    ax = kwargs.get('ax', fig.gca(projection=magnetogram))
    magnetogram.plot(axes=ax, **imshow_kwargs)
    for coord in coords:
        c = coord.transform_to(magnetogram.coordinate_frame)
        if check_visible:
            c = c[is_visible(c, magnetogram.observer_coordinate)]
        if len(c) == 0:
            continue  # Matplotlib throws exception when no points are visible
        ax.plot_coord(c, **plot_kwargs)
    magnetogram.draw_grid(axes=ax, **grid_kwargs)

    return fig, ax


def peek_projections(B_field, **kwargs):
    """
    Quick plot of projections of components of fields along different axes

    .. warning:: These plots are just images and include no spatial information
    """
    norm = kwargs.get('norm', Normalize(vmin=-2e3, vmax=2e3))
    fontsize = kwargs.get('fontsize', 20)
    frames = [
        {'field': 0, 'field_label': 'x', 'axis_label': 'x', 'axis_indices': (2, 1)},
        {'field': 0, 'field_label': 'x', 'axis_label': 'y', 'axis_indices': (0, 2)},
        {'field': 0, 'field_label': 'x', 'axis_label': 'z', 'axis_indices': (0, 1)},
        {'field': 1, 'field_label': 'y', 'axis_label': 'x', 'axis_indices': (2, 1)},
        {'field': 1, 'field_label': 'y', 'axis_label': 'y', 'axis_indices': (0, 2)},
        {'field': 1, 'field_label': 'y', 'axis_label': 'z', 'axis_indices': (0, 1)},
        {'field': 2, 'field_label': 'z', 'axis_label': 'x', 'axis_indices': (2, 1)},
        {'field': 2, 'field_label': 'z', 'axis_label': 'y', 'axis_indices': (0, 2)},
        {'field': 2, 'field_label': 'z', 'axis_label': 'z', 'axis_indices': (0, 1)},
    ]
    fig, axes = plt.subplots(3, 3, figsize=kwargs.get('figsize', (10, 10)))
    ax1_grid, ax2_grid = np.meshgrid(np.linspace(-1, 1, B_field.x.shape[1]),
                                     np.linspace(-1, 1, B_field.x.shape[0]))
    for i, (ax, f) in enumerate(zip(axes.flatten(), frames)):
        b_sum = B_field[f['field']].value.sum(axis=i % 3)
        b_stream_1 = B_field[f['axis_indices'][0]].sum(axis=i % 3).value
        b_stream_2 = B_field[f['axis_indices'][1]].sum(axis=i % 3).value
        if f['axis_label'] != 'z':
            b_sum = b_sum.T
            b_stream_1 = b_stream_1.T
            b_stream_2 = b_stream_2.T
        im = ax.pcolormesh(ax1_grid, ax2_grid, b_sum, norm=norm, cmap=kwargs.get('cmap', 'hmimag'))
        ax.streamplot(ax1_grid[0, :], ax2_grid[:, 0], b_stream_1, b_stream_2,
                      color=kwargs.get('color', 'w'), density=kwargs.get('density', 0.5),
                      linewidth=kwargs.get('linewidth', 2))
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        if i % 3 == 0:
            ax.set_ylabel(f'$B_{f["field_label"]}$', fontsize=fontsize)
        if i > 5:
            ax.set_xlabel(f'$\sum_{f["axis_label"]}$', fontsize=fontsize)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0, right=0.965)
    cax = fig.add_axes([0.975, 0.08, 0.03, 0.9])
    fig.colorbar(im, cax=cax)
    plt.show()
