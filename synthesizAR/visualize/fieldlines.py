"""
Visualizaition functions related to 1D fieldlines
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import astropy.units as u
import astropy.constants as const
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize
from sunpy.map import GenericMap, make_fitswcs_header
from sunpy.coordinates import Helioprojective

from synthesizAR.util import is_visible

__all__ = ['plot_fieldlines', 'peek_fieldlines', 'peek_projections']


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
    norm = kwargs.get('norm', ImageNormalize(vmin=-1.5e3, vmax=1.5e3))
    magnetogram.plot(axes=ax, title=False, cmap=kwargs.get('cmap', 'hmimag'), norm=norm)
    # Grid
    ax.grid(alpha=0.)
    magnetogram.draw_grid(axes=ax, grid_spacing=10*u.deg, alpha=0.75, color='k')
    # Lines
    for line in fieldlines:
        coord = line.transform_to(magnetogram.coordinate_frame)
        # Mask lines behind the solar disk
        i_visible = np.where(is_visible(coord, magnetogram.observer_coordinate))
        coord_visible = SkyCoord(Tx=coord.Tx[i_visible], Ty=coord.Ty[i_visible],
                                 distance=coord.distance[i_visible],
                                 frame=magnetogram.coordinate_frame)
        ax.plot_coord(coord_visible, '-', color=kwargs.get('color', 'k'), lw=kwargs.get('lw', 1),
                      alpha=kwargs.get('alpha', 0.5))

    plt.show()


def plot_fieldlines(*coords, observer=None, check_visible=True, **kwargs):
    """
    Plot fieldlines on the surface of the Sun

    Parameters
    ----------
    coords : `~astropy.coordinates.SkyCoord`
    observer : `~astropy.coordinates.SkyCoord`, optional
        Position of the observer. If None, defaults to (0, 0, 1 AU)
        at the current time.
    check_visible : `bool`
        If True, check that the coordinates are actually visible from
        the given observer location.

    Other Parameters
    ----------------
    plot_kwargs : `dict`
        Additional parameters to pass to `~matplotlib.pyplot.plot`
    grid_kwargs : `dict`
        Additional parameters to pass to `~sunpy.map.Map.draw_grid`
    """
    if observer is None:
        observer = SkyCoord(lat=0*u.deg,
                            lon=0*u.deg,
                            radius=const.au,
                            frame='heliographic_stonyhurst',
                            obstime=Time.now())
    # Dummy map
    data = np.ones((10, 10))
    coord = SkyCoord(Tx=0*u.arcsec,
                     Ty=0*u.arcsec,
                     frame=Helioprojective(observer=observer, obstime=observer.obstime))
    meta = make_fitswcs_header(data, coord, scale=(1, 1)*u.arcsec/u.pixel,)
    dummy_map = GenericMap(data, meta)
    # Plot coordinates
    fig = kwargs.get('fig', plt.figure(figsize=kwargs.get('figsize', None)))
    ax = kwargs.get('ax', fig.gca(projection=dummy_map))
    imshow_kwargs = {'alpha': 0, 'extent': [-1000, 1000, -1000, 1000], 'title': False}
    imshow_kwargs.update(kwargs.get('imshow_kwargs', {}))
    dummy_map.plot(axes=ax, **imshow_kwargs)
    plot_kwargs = kwargs.get('plot_kwargs', {})
    for coord in coords:
        c = coord.transform_to(dummy_map.coordinate_frame)
        if check_visible:
            c = c[is_visible(c, dummy_map.observer_coordinate)]
        if len(c) == 0:
            continue  # Matplotlib throws exception when no points are visible
        ax.plot_coord(c, **plot_kwargs)
    grid_kwargs = kwargs.get('grid_kwargs', {'grid_spacing': 10*u.deg, 'color': 'k'})
    dummy_map.draw_grid(axes=ax, **grid_kwargs)

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
