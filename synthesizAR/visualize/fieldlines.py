"""
Visualizaition functions related to 1D fieldlines
"""
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize
from sunpy.map import GenericMap, make_fitswcs_header
from sunpy.coordinates import Helioprojective

from synthesizAR.util import is_visible

__all__ = ['plot_fieldlines', 'peek_fieldlines']


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
    line_frequency = kwargs.get('line_frequency', 5)
    for line in fieldlines[::line_frequency]:
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
        ax.plot_coord(c, **plot_kwargs)
    grid_kwargs = kwargs.get('grid_kwargs', {'grid_spacing': 10*u.deg, 'color': 'k'})
    dummy_map.draw_grid(axes=ax, **grid_kwargs)

    return fig, ax
