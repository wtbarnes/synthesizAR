"""
Return sunpy maps or map-like objects of model quantities at given times
"""
import numpy as np
import h5py
import astropy.units as u
from matplotlib import cm, colors
from sunpy.map import GenericMap

from synthesizAR.util import is_visible
from .cube import EMCube

__all__ = ['make_los_velocity_map', 'make_temperature_map', 'make_emission_measure_map']


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
            i_time = np.where(np.array(hf['time'])*u.Unit(hf['time'].attrs.get(
                'unit', hf['time'].attrs.get('units'))) == time)[0][0]
        except IndexError:
            raise IndexError(f'{time} is not a valid time in observing time for {instr.name}')
        v_x = u.Quantity(hf['velocity_x'][i_time, :], hf['velocity_x'].attrs.get(
            'unit', hf['velocity_x']['units']))
        v_y = u.Quantity(hf['velocity_y'][i_time, :], hf['velocity_y'].attrs.get(
            'unit', hf['velocity_y']['units']))
        v_z = u.Quantity(hf['velocity_z'][i_time, :], hf['velocity_z'].attrs.get(
            'unit', hf['velocity_z']['units']))
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
            i_time = np.where(np.array(hf['time'])*u.Unit(hf['time'].attrs.get(
                'unit', hf['time']['units'])) == time)[0][0]
        except IndexError:
            raise IndexError(f'{time} is not a valid time in observing time for {instr.name}')
        weights = np.array(hf['electron_temperature'][i_time, :])
        units = u.Unit(hf['electron_temperature'].attrs.get('unit',
                                                            hf['electron_temperature']['units']))
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


@u.quantity_input
def make_emission_measure_map(time: u.s, field, instr, temperature_bin_edges=None, **kwargs):
    """
    Compute true emission meausure in each pixel as a function of electron temperature.

    Parameters
    ----------
    time : `~astropy.units.Quantity`
    field : `~synthesizAR.Field`
    instr : `~synthesizAR.instruments.InstrumentBase`
    temperature_bin_edges : `~astropy.units.Quantity`

    Other Parameters
    ----------------
    plot_settings : `dict`

    Returns
    -------
    `~synthesizAR.maps.EMCube`
    """
    plot_settings = {'cmap': cm.get_cmap('magma'),
                     'norm': colors.SymLogNorm(1, vmin=1e25, vmax=1e29)}
    plot_settings.update(kwargs.get('plot_settings', {}))

    # read unbinned temperature and density
    with h5py.File(instr.counts_file, 'r') as hf:
        try:
            i_time = np.where(np.array(hf['time'])*u.Unit(hf['time'].attrs.get(
                'unit', hf['time'].attrs.get('units'))) == time)[0][0]
        except IndexError:
            raise IndexError(f'{time} is not a valid time in observing time for {instr.name}')
        unbinned_temperature = np.array(hf['electron_temperature'][i_time, :])
        temperature_unit = u.Unit(hf['electron_temperature'].attrs.get(
            'unit', hf['electron_temperature'].attrs.get('units')))
        unbinned_density = np.array(hf['density'][i_time, :])
        density_unit = u.Unit(hf['density'].attrs.get('unit', hf['density'].attrs.get('units')))

    # setup bin edges and weights
    if temperature_bin_edges is None:
        temperature_bin_edges = 10.**(np.arange(5.5, 7.5, 0.1))*u.K
    bins, bin_range = instr.make_detector_array(field)
    x_bin_edges = (np.diff(bin_range.x) / bins.x.value
                   * np.arange(bins.x.value + 1) + bin_range.x[0])
    y_bin_edges = (np.diff(bin_range.y) / bins.y.value
                   * np.arange(bins.y.value + 1) + bin_range.y[0])
    dh = np.diff(bin_range.z).cgs[0] / bins.z * (1. * u.pixel)
    visible = is_visible(instr.total_coordinates, instr.observer_coordinate)
    emission_measure_weights = (unbinned_density**2) * dh * visible
    # bin in x,y,T space with emission measure weights
    xyT_coordinates = np.append(np.stack([instr.total_coordinates.Tx,
                                          instr.total_coordinates.Ty], axis=1),
                                unbinned_temperature[:, np.newaxis], axis=1)
    hist, _ = np.histogramdd(xyT_coordinates,
                             bins=[x_bin_edges, y_bin_edges, temperature_bin_edges.value],
                             weights=emission_measure_weights)

    meta_base = instr.make_fits_header(field, instr.channels[0])
    del meta_base['wavelnth']
    del meta_base['waveunit']
    meta_base['detector'] = r'Emission measure'
    meta_base['comment'] = 'LOS Emission Measure distribution'
    em_unit = density_unit * density_unit * dh.unit
    data = np.transpose(hist, (1, 0, 2)) * em_unit

    return EMCube(data, meta_base, temperature_bin_edges, plot_settings=plot_settings)
