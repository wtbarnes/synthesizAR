"""
Plotting functions for easily and quickily visualizing synthesized AIA results
"""
import os

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.animation
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.map import Map

from synthesizAR.util import get_keys

__all__ = ['plot_aia_channels', 'make_aia_animation']


def plot_aia_channels(aia, time: u.s, root_dir, corners=None, figsize=None, norm=None, fontsize=14, 
                      **kwargs):
    """
    Plot maps of the EUV channels of AIA for a given timestep

    Parameters
    ----------
    aia : `synthesizAR.instruments.InstrumentSDOAIA`
    time : `astropy.Quantity`
    root_dir : `str`
    figsize : `tuple`, optional
    """
    if figsize is None:
        figsize = (15, 10)
    if norm is None:
        norm = matplotlib.colors.SymLogNorm(1e-6, vmin=1, vmax=5e3)
    with h5py.File(aia.counts_file, 'r') as hf:
        reference_time = u.Quantity(hf['time'], get_keys(hf['time'].attrs, ('unit', 'units')))
    i_time = np.where(reference_time == time)[0][0]
    fig_format = os.path.join(root_dir, f'{aia.name}', '{}', f'map_t{i_time:06d}.fits')
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(wspace=0., hspace=0., top=0.95)
    ims = {}
    for i, channel in enumerate(aia.channels):
        tmp = Map(fig_format.format(channel['name']))
        if corners is not None:
            blc = SkyCoord(*corners[0], frame=tmp.coordinate_frame)
            trc = SkyCoord(*corners[1], frame=tmp.coordinate_frame)
            tmp = tmp.submap(blc, trc)
        ax = fig.add_subplot(2, 3, i+1, projection=tmp)
        ims[channel['name']] = tmp.plot(annotate=False, title=False, norm=norm)
        lon, lat = ax.coords
        lon.grid(alpha=0)
        lat.grid(alpha=0)
        if i % 3 == 0:
            lat.set_axislabel(r'solar-y [arcsec]', fontsize=fontsize)
        else:
            lat.set_ticks_visible(False)
            lat.set_ticklabel_visible(False)
        if i > 2:
            lon.set_axislabel(r'solar-x [arcsec]', fontsize=fontsize)
        else:
            lon.set_ticks_visible(False)
            lon.set_ticklabel_visible(False)
        ax.text(0.1*tmp.dimensions.x.value, 0.9*tmp.dimensions.y.value,
                r'${}$ $\mathrm{{\mathring{{A}}}}$'.format(channel['name']),
                color='w', fontsize=fontsize)
    fig.suptitle(r'$t={:.0f}$ {}'.format(time.value, time.unit.to_string()), fontsize=fontsize)
    if kwargs.get('use_with_animation', False):
        return fig, ims


def make_aia_animation(aia, start_time: u.s, stop_time: u.s, root_dir, figsize=None, norm=None, 
                       fontsize=14, **kwargs):
    """
    Build animation from a series of synthesized AIA observations
    """
    with h5py.File(aia.counts_file, 'r') as hf:
        reference_time = u.Quantity(hf['time'], get_keys(hf['time'].attrs, ('unit', 'units')))
    start_index = np.where(reference_time == start_time)[0][0]
    stop_index = np.where(reference_time == stop_time)[0][0]
    fig_format = os.path.join(root_dir, f'{aia.name}', '{}', 'map_t{:06d}.fits')
    fig, ims = plot_aia_channels(aia, start_time, root_dir, figsize=figsize, norm=norm,
                                 fontsize=fontsize, use_with_animation=True)

    def update_fig(i):
        for channel in aia.channels:
            tmp = Map(fig_format.format(channel['name'], i))
            ims[channel['name']].set_array(tmp.data)
        fig.suptitle(f'$t={reference_time[i].value:.0f}$ {reference_time.unit.to_string()}',
                     fontsize=fontsize)
        return [ims[k] for k in ims]

    animator_settings = {'interval': 50, 'blit': True}
    animator_settings.update(kwargs.get('animator_settings', {}))
    animation = matplotlib.animation.FuncAnimation(fig, update_fig,
                                                   frames=range(start_index, stop_index),
                                                   **animator_settings)

    return animation
