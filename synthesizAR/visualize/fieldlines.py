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

__all__ = ['plot_fieldlines']


def plot_fieldlines(*coords, lon=0*u.deg, lat=0*u.deg, **kwargs):
    """
    Plot fieldlines on the surface of the Sun

    Parameters
    ----------
    coords : `~astropy.coordinates.SkyCoord`
    lon : `~astropy.units.Quantity`
        Heliographic longitude of observer
    lat : `~astropy.units.Quantity`
        Heliographic latitude of observer

    Other Parameters
    ----------------
    plot_kwargs : `dict`
        Additional parameters to pass to `~matplotlib.pyplot.plot`
    grid_kwargs : `dict`
        Additional parameters to pass to `~sunpy.map.Map.draw_grid`
    """
    # Dummy map
    data = np.ones((10, 10))
    time_now = Time.now()
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
        'hgln_obs': lon.to(u.deg).value,
        'hglt_obs': lat.to(u.deg).value,
        'dsun_obs': const.au.to(u.m).value,
        'dsun_ref': const.au.to(u.m).value,
        'rsun_ref': const.R_sun.to(u.m).value,
        'rsun_obs': ((const.R_sun/const.au).decompose()*u.radian).to(u.arcsec).value,
        't_obs': time_now.iso,
        'date-obs': time_now.iso,
    })
    dummy_map = GenericMap(data, meta)
    # Plot coordinates
    fig = kwargs.get('fig', plt.figure(figsize=kwargs.get('figsize', None)))
    ax = kwargs.get('ax', fig.gca(projection=dummy_map))
    dummy_map.plot(alpha=0, extent=[-1000, 1000, -1000, 1000], title=False, axes=ax)
    plot_kwargs = kwargs.get('plot_kwargs', {})
    for coord in coords:
        ax.plot_coord(coord.transform_to(dummy_map.coordinate_frame), **plot_kwargs)
    grid_kwargs = kwargs.get('grid_kwargs', {'grid_spacing': 10*u.deg, 'color': 'k'})
    dummy_map.draw_grid(axes=ax, **grid_kwargs)

    return fig, ax
