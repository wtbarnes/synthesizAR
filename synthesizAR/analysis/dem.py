"""
Very simple tools for analyzing differential emission measure data
"""
import warnings

import numpy as np
from matplotlib import cm, colors
import h5py
import astropy.units as u
from sunpy.map import MapSequence, GenericMap
from sunpy.util.metadata import MetaDict

from synthesizAR.util import is_visible, get_keys

__all__ = ['EMCube', 'make_emission_measure_map']


class EMCube(MapSequence):
    """
    Container for the emission measure at each pixel of a map for a range of temperatures.

    .. warning:: This object will be likely be moved out of this package in the near future.

    Parameters
    ----------
    data : `~astropy.units.Quantity`
        3D array (2D spatial + temperature) cube of emission measure data
    header : `~sunpy.util.metadata.MetaDict`
    temperature_bin_edges : `~astropy.units.Quantity`
        Should have same shape as third dimension of `data`

    Other Parameters
    ----------------
    plot_settings : `dict`
    """

    @u.quantity_input
    def __init__(self, data, header, temperature_bin_edges: u.K, **kwargs):
        # TODO: refactor to use NDCube
        self.temperature_bin_edges = temperature_bin_edges
        # sanitize header
        meta_base = header.copy()
        meta_base['temp_unit'] = self.temperature_bin_edges.unit.to_string()
        meta_base['bunit'] = data.unit.to_string()
        # build map list
        map_list = []
        for i in range(self.temperature_bin_edges.shape[0] - 1):
            tmp = GenericMap(data[:, :, i], meta_base)
            tmp.meta['temp_a'] = self.temperature_bin_edges[i].value
            tmp.meta['temp_b'] = self.temperature_bin_edges[i+1].value
            tmp.plot_settings.update(kwargs.get('plot_settings', {}))
            map_list.append(tmp)

        # call super method
        super().__init__(map_list)

    @property
    def temperature_bin_centers(self,):
        return (self.temperature_bin_edges[1:] + self.temperature_bin_edges[:-1])/2.

    @property
    def total_emission(self):
        """
        Sum the emission measure over all temperatures.
        """
        tmp_meta = self[0].meta.copy()
        tmp_meta['temp_a'] = self.temperature_bin_edges[0]
        tmp_meta['temp_b'] = self.temperature_bin_edges[-1]
        return GenericMap(self.as_array().sum(axis=2), tmp_meta,
                          plot_settings=self[0].plot_settings)

    def get_1d_distribution(self, bottom_left_corner, top_right_corner):
        """
        Mean emission measure distribution over area defined by corners.

        Parameters
        ----------
        bottom_left_corner : `~astropy.coordinates.SkyCoord`
        top_right_corner : `~astropy.coordinates.SkyCoord`
        """
        em_list = []
        for i in range(self.temperature_bin_edges.shape[0] - 1):
            em_list.append(self[i].submap(bottom_left_corner, top_right_corner).data.mean())

        return u.Quantity(em_list, u.Unit(self[0].meta['bunit']))

    def make_slope_map(self, temperature_bounds=None, em_threshold=None, rsquared_tolerance=0.5,
                       full=False):
        """
        Calculate emission measure slope :math:`a` in each pixel

        Create map of emission measure slopes by fitting :math:`\mathrm{EM}\sim T^a` for a
        given temperature range. A slope is masked if a value between the `temperature_bounds`
        is less than :math:`\mathrm{EM}`. Additionally, the "goodness-of-fit" is evaluated using
        the correlation coefficient, :math:`r^2=1 - R_1/R_0`, where :math:`R_1` and :math:`R_0`
        are the residuals from the first and zeroth order polynomial fits, respectively. We mask
        the slope if :math:`r^2` is less than `rsquared_tolerance`.

        Parameters
        ----------
        temperature_bounds : `~astropy.units.Quantity`, optional
        em_threshold : `~astropy.units.Quantity`, optional
            Mask slope if any emission measure in the fit interval is below this value
        rsquared_tolerance : `float`
            Mask any slopes with a correlation coefficient, :math:`r^2`, below this value
        full : `bool`
            If True, return maps of the intercept and :math:`r^2` values as well
        """
        if temperature_bounds is None:
            temperature_bounds = u.Quantity((1e6, 4e6), u.K)
        if em_threshold is None:
            em_threshold = u.Quantity(1e25, u.cm**(-5))
        # Get temperature fit array
        index_temperature_bounds = np.where(np.logical_and(
            self.temperature_bin_centers >= temperature_bounds[0],
            self.temperature_bin_centers <= temperature_bounds[1]))
        temperature_fit = np.log10(
            self.temperature_bin_centers[index_temperature_bounds].to(u.K).value)
        if temperature_fit.size < 3:
            warnings.warn(f'Fitting to fewer than 3 points in temperature space: {temperature_fit}')
        # Cut on temperature
        data = u.Quantity(self.as_array()[:, :, index_temperature_bounds].squeeze(),
                          self[0].meta['bunit'])
        # Get EM fit array
        em_fit = np.log10(data.value.reshape((np.prod(data.shape[:2]),) + data.shape[2:]).T)
        em_fit[np.logical_or(np.isinf(em_fit), np.isnan(em_fit))] = 0.0  # Filter before fitting
        # Fit to first-order polynomial
        coefficients, rss, _, _, _ = np.polyfit(temperature_fit, em_fit, 1, full=True)
        # Create masks from EM threshold and correlation coefficient
        em_mask = np.any(data < em_threshold, axis=2)
        _, rss_flat, _, _, _ = np.polyfit(temperature_fit, em_fit, 0, full=True)
        rss = 0.*rss_flat if rss.size == 0 else rss  # returns empty residual when fit is exact
        rsquared = 1. - rss/rss_flat
        rsquared_mask = rsquared.reshape(data.shape[:2]) < rsquared_tolerance
        # Rebuild into a map
        base_meta = self[0].meta.copy()
        base_meta['temp_a'] = 10.**temperature_fit[0]
        base_meta['temp_b'] = 10.**temperature_fit[-1]
        base_meta['bunit'] = ''
        base_meta['detector'] = 'EM slope'
        base_meta['comment'] = 'Linear fit to log-transformed LOS EM'
        plot_settings = self[0].plot_settings.copy()
        plot_settings['norm'] = None
        m = GenericMap(coefficients[0, :].reshape(data.shape[:2]), base_meta,
                       mask=np.stack((em_mask, rsquared_mask), axis=2).any(axis=2),
                       plot_settings=plot_settings)

        return (m, coefficients, rsquared) if full else m

    def __getitem__(self, key):
        """
        Override the MapCube indexing so that an `EMCube` object is returned.
        """
        if type(self.temperature_bin_edges[key].value) == np.ndarray and \
           len(self.temperature_bin_edges[key].value) > 1:
            tmp_data = u.Quantity(self.as_array()[:, :, key], u.Unit(self.maps[0].meta['bunit']))
            tmp_meta = self.maps[0].meta.copy()
            tmp = EMCube(tmp_data, tmp_meta, self.temperature_bin_edges[key],
                         plot_settings=self.maps[0].plot_settings)
        else:
            tmp = self.maps[key]
            tmp.meta['temp_a'] = self.temperature_bin_edges[key].value
            tmp.meta['temp_b'] = self.temperature_bin_edges[key+1].value

        return tmp

    def save(self, filename):
        """
        Save emission measure cube to an HDF5 file.
        """
        with h5py.File(filename, 'x') as hf:
            dset_data = hf.create_dataset('emission_measure', data=self.as_array())
            dset_data.attrs['unit'] = self[0].meta['bunit']
            dset_temperature_bin_edges = hf.create_dataset(
                'temperature_bin_edges', data=self.temperature_bin_edges.value)
            dset_temperature_bin_edges.attrs['unit'] = self.temperature_bin_edges.unit.to_string()
            meta_group = hf.create_group('meta')
            for key in self[0].meta:
                meta_group.attrs[key] = self[0].meta[key]

    @classmethod
    def restore(cls, filename, **kwargs):
        """
        Restore `EMCube` from an HDF5 file.
        """
        header = MetaDict()
        with h5py.File(filename, 'r') as hf:
            data = u.Quantity(hf['emission_measure'],
                              get_keys(hf['emission_measure'].attrs, ('unit', 'units')))
            temperature_bin_edges = u.Quantity(
                hf['temperature_bin_edges'],
                get_keys(hf['temperature_bin_edges'].attrs, ('unit', 'units')))
            for key in hf['meta'].attrs:
                header[key] = hf['meta'].attrs[key]

        return cls(data, header, temperature_bin_edges, **kwargs)


@u.quantity_input
def make_emission_measure_map(time: u.s, field, instr, temperature_bin_edges=None,
                              **kwargs) -> EMCube:
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
    """
    plot_settings = {'cmap': cm.get_cmap('magma'),
                     'norm': colors.SymLogNorm(1, vmin=1e25, vmax=1e29)}
    plot_settings.update(kwargs.get('plot_settings', {}))

    # read unbinned temperature and density
    with h5py.File(instr.counts_file, 'r') as hf:
        try:
            i_time = np.where(u.Quantity(hf['time'],
                              get_keys(hf['time'].attrs), ('unit', 'units')) == time)[0][0]
        except IndexError:
            raise IndexError(f'{time} is not a valid time in observing time for {instr.name}')
        unbinned_temperature = np.array(hf['electron_temperature'][i_time, :])
        temperature_unit = u.Unit(get_keys(hf['electron_temperature'].attrs, ('unit', 'units')))
        unbinned_density = np.array(hf['density'][i_time, :])
        density_unit = u.Unit(get_keys(hf['density'].attrs, ('unit', 'units')))

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
