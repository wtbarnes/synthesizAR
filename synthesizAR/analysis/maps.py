"""
Return sunpy maps or map-like objects of model quantities at given times
"""
import numpy as np
import h5py
import astropy.units as u
from matplotlib import cm, colors
from sunpy.map import GenericMap

from synthesizAR.util import is_visible, get_keys

__all__ = ['make_los_velocity_map', 'make_temperature_map']


@u.quantity_input
def make_los_velocity_map(time: u.s, field, instr, **kwargs):
    """
    Return map of LOS velocity at a given time for a given instrument resolution.
    """
    plot_settings = {'cmap': cm.get_cmap('bwr'),
                     'norm': colors.SymLogNorm(10, vmin=-1e8, vmax=1e8)}
    plot_settings.update(kwargs.get('plot_settings', {}))
    
    bins, bin_range = instr.make_detector_array(field)
    visible = is_visible(instr.total_coordinates, instr.observer_coordinate)
    hist_coordinates, _, _ = np.histogram2d(instr.total_coordinates.Tx.value,
                                            instr.total_coordinates.Ty.value,
                                            bins=(bins.x.value, bins.y.value),
                                            range=(bin_range.x.value, bin_range.y.value),
                                            weights=visible)
    with h5py.File(instr.counts_file, 'r') as hf:
        try:
            i_time = np.where(u.Quantity(hf['time'],
                              get_keys(hf['time'].attrs), ('unit', 'units')) == time)[0][0]
        except IndexError:
            raise IndexError(f'{time} is not a valid time in observing time for {instr.name}')
        v_x = u.Quantity(hf['velocity_x'][i_time, :],
                         get_keys(hf['velocity_x'].attrs, ('unit', 'units')))
        v_y = u.Quantity(hf['velocity_y'][i_time, :], 
                         get_keys(hf['velocity_y'].attrs, ('unit', 'units')))
        v_z = u.Quantity(hf['velocity_z'][i_time, :], 
                         get_keys(hf['velocity_z'].attrs, ('unit', 'units')))
        v_los = instr.los_velocity(v_x, v_y, v_z)

    hist, _, _ = np.histogram2d(instr.total_coordinates.Tx.value,
                                instr.total_coordinates.Ty.value,
                                bins=(bins.x.value, bins.y.value),
                                range=(bin_range.x.value, bin_range.y.value),
                                weights=v_los.value * visible)
    hist /= np.where(hist_coordinates == 0, 1, hist_coordinates)
    meta = instr.make_fits_header(field, instr.channels[0])
    del meta['wavelnth']
    del meta['waveunit']
    meta['bunit'] = v_los.unit.to_string()
    meta['detector'] = 'LOS Velocity'
    meta['comment'] = 'LOS velocity calculated by synthesizAR'

    return GenericMap(hist.T, meta, plot_settings=plot_settings)


@u.quantity_input
def make_temperature_map(time: u.s, field, instr, **kwargs):
    """
    Return map of column-averaged electron temperature at a given time for a given instrument
    resolution.
    """
    plot_settings = {'cmap': cm.get_cmap('inferno')}
    plot_settings.update(kwargs.get('plot_settings', {}))
    bins, bin_range = instr.make_detector_array(field)
    visible = is_visible(instr.total_coordinates, instr.observer_coordinate)
    hist_coordinates, _, _ = np.histogram2d(instr.total_coordinates.Tx.value,
                                            instr.total_coordinates.Ty.value,
                                            bins=(bins.x.value, bins.y.value),
                                            range=(bin_range.x.value, bin_range.y.value),
                                            weights=visible)
    with h5py.File(instr.counts_file, 'r') as hf:
        try:
            i_time = np.where(u.Quantity(hf['time'],
                              get_keys(hf['time'].attrs), ('unit', 'units')) == time)[0][0]
        except IndexError:
            raise IndexError(f'{time} is not a valid time in observing time for {instr.name}')
        weights = np.array(hf['electron_temperature'][i_time, :])
        units = u.Unit(get_keys(hf['electron_temperature'].attrs, ('unit', 'units')))
    hist, _, _ = np.histogram2d(instr.total_coordinates.Tx.value,
                                instr.total_coordinates.Ty.value,
                                bins=(bins.x.value, bins.y.value),
                                range=(bin_range.x.value, bin_range.y.value),
                                weights=weights * visible)
    hist /= np.where(hist_coordinates == 0, 1, hist_coordinates)
    meta = instr.make_fits_header(field, instr.channels[0])
    del meta['wavelnth']
    del meta['waveunit']
    meta['bunit'] = units.to_string()
    meta['detector'] = 'Electron Temperature'
    meta['comment'] = 'Column-averaged electron temperature calculated by synthesizAR'

    return GenericMap(hist.T, meta, plot_settings=plot_settings)
