"""
Wrapper functions for producing time lag and cross-correlation maps from data cubes.
"""
import astropy.units as u
import itertools
import sunkit_image.time_lag
import sunpy.map

from synthesizAR.instruments.sdo import _AIA_CHANNEL_WAVELENGTHS

import synthesizAR.analysis.time_lag.map_sources  # NOQA


__all__ = ['get_aia_channel_combinations', 'make_time_lag_map', 'make_cross_correlation_map']


def get_aia_channel_combinations():
    """
    Convenience function for listing all possible AIA channel pairs.
    This is useful for computing time lags.
    """
    channel_list = [f"{chan.to_value('Angstrom'):.0f}" for chan in _AIA_CHANNEL_WAVELENGTHS]
    channel_combinations = list(itertools.combinations(channel_list, 2))
    channel_combinations = channel_combinations[:5] + [sorted(c, key=lambda x: float(x), reverse=True) for c in channel_combinations[5:]]
    return channel_combinations


def _get_meta_and_time(cube_a, cube_b, lag_bounds):
    time_a = cube_a.axis_world_coords('time')[0]
    time_b = cube_b.axis_world_coords('time')[0]
    if not (time_a == time_b).all():
        raise ValueError('Time axes of both cubes must be the same')
    time = (time_a - time_a[0]).to('s')
    if lag_bounds is None:
        lag_bounds = u.Quantity([-time[-1]/2, time[-1]/2])
    meta = cube_a.meta.copy()
    stale_keys = ['bunit', 'date_sim', 'wavelnth', 'waveunit', 'instrume', 'telescop', 'obsrvtry', 'detector']
    for k in stale_keys:
        _ = meta.pop(k)
    meta['chan_a'] = cube_a.meta.get('wavelnth')
    meta['chan_b'] = cube_b.meta.get('wavelnth')
    return time, lag_bounds, meta


@u.quantity_input
def make_time_lag_map(cube_a, cube_b, lag_bounds: u.s=None):
    """
    Coordinate-aware wrapper around `~sunkit_image.time_lag.time_lag`

    Parameters
    ----------
    cube_a : `~ndcube.NDCube`
    cube_b : `~ndcube.NDCube`
    lag_bounds : `~astropy.units.Quantity`

    Return
    ------
    : `~sunpy.map.GenericMap`
    """
    time, lag_bounds, meta = _get_meta_and_time(cube_a, cube_b, lag_bounds)
    data = sunkit_image.time_lag.time_lag(cube_a.data, cube_b.data, time, lag_bounds=lag_bounds)
    meta['bunit'] = time.unit.to_string(format='FITS')
    meta['measrmnt'] = 'time_lag'
    return sunpy.map.Map(data, meta)


@u.quantity_input
def make_cross_correlation_map(cube_a, cube_b, lag_bounds: u.s=None):
    """
    Coordinate-aware wrapper around `~sunkit_image.time_lag.max_cross_correlation`

    Parameters
    ----------
    cube_a : `~ndcube.NDCube`
    cube_b : `~ndcube.NDCube`
    lag_bounds : `~astropy.units.Quantity`

    Return
    ------
    : `~sunpy.map.GenericMap`
    """
    time, lag_bounds, meta = _get_meta_and_time(cube_a, cube_b, lag_bounds)
    data = sunkit_image.time_lag.max_cross_correlation(cube_a.data, cube_b.data, time, lag_bounds=lag_bounds)
    meta['measrmnt'] = 'max_cross_correlation'
    return sunpy.map.Map(data, meta)
