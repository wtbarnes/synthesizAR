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
    
    visible = is_visible(instr.total_coordinates, instr.observer_coordinate)
    hist_coordinates, _, _ = np.histogram2d(instr.total_coordinates.Tx.value,
                                            instr.total_coordinates.Ty.value,
                                            bins=(instr.bins.x.value, instr.bins.y.value),
                                            range=(instr.bin_range.x.value, instr.bin_range.y.value),
                                            weights=visible)
    with h5py.File(instr.counts_file, 'r') as hf:
        try:
            i_time = np.where(np.array(hf['time'])*u.Unit(hf['time'].attrs['units']) == time)[0][0]
        except IndexError:
            raise IndexError(f'{time} is not a valid time in observing time for {instr.name}')
        weights = np.array(hf['los_velocity'][i_time, :])
        units = u.Unit(hf['los_velocity'].attrs['units'])

    hist, _, _ = np.histogram2d(instr.total_coordinates.Tx.value,
                                instr.total_coordinates.Ty.value,
                                bins=(instr.bins.x.value, instr.bins.y.value),
                                range=(instr.bin_range.x.value, instr.bin_range.y.value),
                                weights=weights * visible)
    hist /= np.where(hist_coordinates == 0, 1, hist_coordinates)
    meta = instr.make_fits_header(field, instr.channels[0])
    del meta['wavelnth']
    del meta['waveunit']
    meta['bunit'] = units.to_string()
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

    visible = is_visible(instr.total_coordinates, instr.observer_coordinate)
    hist_coordinates, _, _ = np.histogram2d(instr.total_coordinates.Tx.value,
                                            instr.total_coordinates.Ty.value,
                                            bins=(instr.bins.x.value, instr.bins.y.value),
                                            range=(instr.bin_range.x.value, instr.bin_range.y.value),
                                            weights=visible)
    with h5py.File(instr.counts_file, 'r') as hf:
        try:
            i_time = np.where(np.array(hf['time'])*u.Unit(hf['time'].attrs['units']) == time)[0][0]
        except IndexError:
            raise IndexError(f'{time} is not a valid time in observing time for {instr.name}')
        weights = np.array(hf['electron_temperature'][i_time, :])
        units = u.Unit(hf['electron_temperature'].attrs['units'])
    hist, _, _ = np.histogram2d(instr.total_coordinates.Tx.value,
                                instr.total_coordinates.Ty.value,
                                bins=(instr.bins.x.value, instr.bins.y.value),
                                range=(instr.bin_range.x.value, instr.bin_range.y.value),
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
    Return a cube of maps showing the true emission meausure in each pixel
    as a function of electron temperature.
    """
    plot_settings = {'cmap': cm.get_cmap('magma'),
                     'norm': colors.SymLogNorm(1, vmin=1e25, vmax=1e29)}
    plot_settings.update(kwargs.get('plot_settings', {}))

    # read unbinned temperature and density
    with h5py.File(instr.counts_file, 'r') as hf:
        try:
            i_time = np.where(np.array(hf['time'])*u.Unit(hf['time'].attrs['units']) == time)[0][0]
        except IndexError:
            raise IndexError(f'{time} is not a valid time in observing time for {instr.name}')
        unbinned_temperature = np.array(hf['electron_temperature'][i_time, :])
        temperature_unit = u.Unit(hf['electron_temperature'].attrs['units'])
        unbinned_density = np.array(hf['density'][i_time, :])
        density_unit = u.Unit(hf['density'].attrs['units'])

    # setup bin edges and weights
    if temperature_bin_edges is None:
        temperature_bin_edges = 10.**(np.arange(5.5, 7.5, 0.1))*u.K
    x_bin_edges = (np.diff(instr.bin_range.x) / instr.bins.x.value
                   * np.arange(instr.bins.x.value + 1) + instr.bin_range.x[0])
    y_bin_edges = (np.diff(instr.bin_range.y) / instr.bins.y.value
                   * np.arange(instr.bins.y.value + 1) + instr.bin_range.y[0])
    dh = np.diff(instr.bin_range.z).cgs[0] / instr.bins.z * (1. * u.pixel)
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
