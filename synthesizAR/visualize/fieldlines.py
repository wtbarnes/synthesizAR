"""
Visualizaition functions related to 1D fieldlines
"""
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
from astropy.time import Time
from astropy.coordinates import SkyCoord
from sunpy.util.metadata import MetaDict
from sunpy.map import GenericMap

from synthesizAR.util import is_visible

__all__ = ['plot_fieldlines']


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
    observer = observer.transform_to('heliographic_stonyhurst')
    # Dummy map
    data = np.ones((10, 10))
    meta = MetaDict({
        'ctype1': 'HPLN-TAN',
        'ctype2': 'HPLT-TAN',
        'cunit1': 'arcsec',
        'cunit2': 'arcsec',
        'crpix1': (data.shape[0] + 1)/2.,
        'crpix2': (data.shape[1] + 1)/2.,
        'cdelt1': 1.0,
        'cdelt2': 1.0,
        'crval1': 0.0,
        'crval2': 0.0,
        'hgln_obs': observer.lon.to(u.deg).value,
        'hglt_obs': observer.lat.to(u.deg).value,
        'dsun_obs': observer.radius.to(u.m).value,
        'dsun_ref': observer.radius.to(u.m).value,
        'rsun_ref': const.R_sun.to(u.m).value,
        'rsun_obs': ((const.R_sun/observer.radius).decompose()*u.radian).to(u.arcsec).value,
        't_obs': observer.obstime.iso,
        'date-obs': observer.obstime.iso,
    })
    dummy_map = GenericMap(data, meta)
    # Plot coordinates
    fig = kwargs.get('fig', plt.figure(figsize=kwargs.get('figsize', None)))
    ax = kwargs.get('ax', fig.gca(projection=dummy_map))
    dummy_map.plot(alpha=0, extent=[-1000, 1000, -1000, 1000], title=False, axes=ax)
    plot_kwargs = kwargs.get('plot_kwargs', {})
    for coord in coords:
        c = coord.transform_to(dummy_map.coordinate_frame)
        if check_visible:
            c = c[is_visible(c, dummy_map.observer_coordinate)]
        ax.plot_coord(c, **plot_kwargs)
    grid_kwargs = kwargs.get('grid_kwargs', {'grid_spacing': 10*u.deg, 'color': 'k'})
    dummy_map.draw_grid(axes=ax, **grid_kwargs)

    return fig, ax
