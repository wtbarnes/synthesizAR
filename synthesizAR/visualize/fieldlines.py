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

from synthesizAR.util import is_visible, find_minimum_fov

__all__ = ['set_ax_lims', 'plot_fieldlines']


def set_ax_lims(ax, xlim, ylim, smap):
    """
    Set limits on a `~sunpy.map.Map` plot
    """
    x_lims, y_lims = smap.world_to_pixel(
        SkyCoord(xlim, ylim, frame=smap.coordinate_frame))
    ax.set_xlim(x_lims.value)
    ax.set_ylim(y_lims.value)


def plot_fieldlines(*coords,
                    image_map=None,
                    observer=None,
                    check_visible=False,
                    draw_grid=True,
                    axes_limits=None,
                    **kwargs):
    """
    Plot fieldlines on the surface of the Sun

    Parameters
    ----------
    coords : `~astropy.coordinates.SkyCoord`
    image_map : `~sunpy.map.Map`, optional
        Map to overplot the field lines on top of. Useful
        when the field lines are derived from a magnetic field extrapolation or
        when we want to overlay fieldlines on an EUV image.
    observer : `~astropy.coordinates.SkyCoord`, optional
        Position of the observer. If None, defaults to position of Earth at the
        current time. If `image_map` is specified, this argument has no effect
        and the observer will be the observer as defined by the HPC coordinate
        frame of the `image_map`.
    check_visible : `bool`, optional
        If True, mask coordinates that are obscured by the solar disk.
    draw_grid : `bool`, optional
        If True, draw the HGS grid
    axes_limits : `tuple`, optional
        Tuple of world coordinates (axis1, axis2)

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
    if image_map is None:
        # If no image_map is given, create a dummy transparent map for some specified
        # observer location
        data = np.ones((1000, 1000))  # make this big to give more fine-grained control in w2pix
        observer = get_earth(Time.now()) if observer is None else observer
        coord = SkyCoord(Tx=0*u.arcsec,
                         Ty=0*u.arcsec,
                         frame=Helioprojective(observer=observer, obstime=observer.obstime))
        meta = make_fitswcs_header(data, coord, scale=(1, 1)*u.arcsec/u.pixel,)
        image_map = GenericMap(data, meta)
        # Show the full disk, make the dummy map transparent
        imshow_kwargs.update({'alpha': 0})
    else:
        imshow_kwargs.update({
            'cmap': 'hmimag', 'norm': ImageNormalize(vmin=-1.5e3, vmax=1.5e3)
        })
    imshow_kwargs.update(kwargs.get('imshow_kwargs', {}))
    # Plot coordinates
    fig = kwargs.get('fig', plt.figure(figsize=kwargs.get('figsize', None)))
    ax = kwargs.get('ax', fig.add_subplot(111, projection=image_map))
    image_map.plot(axes=ax, **imshow_kwargs)
    transformed_coords = []
    for coord in coords:
        c = coord.transform_to(image_map.coordinate_frame)
        if check_visible:
            c = c[is_visible(c, image_map.observer_coordinate)]
        transformed_coords.append(c)
        if len(c) == 0:
            continue  # Matplotlib throws exception when no points are visible
        ax.plot_coord(c, **plot_kwargs)
    if draw_grid:
        image_map.draw_grid(axes=ax, **grid_kwargs)
    if axes_limits is None:
        transformed_coords = SkyCoord(transformed_coords)
        blc, trc = find_minimum_fov(transformed_coords, padding=(10, 10)*u.arcsec)
        axes_limits = (u.Quantity([blc.Tx, trc.Tx]), u.Quantity([blc.Ty, trc.Ty]))
    set_ax_lims(ax, *axes_limits, image_map)
    return fig, ax, image_map
