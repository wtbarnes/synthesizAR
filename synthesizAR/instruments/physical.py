"""
Instruments for calculating physical quantities, rather than
observed counts, projected along a LOS
"""
from dataclasses import dataclass
import warnings

import astropy.units as u
import astropy.wcs
import numpy as np
from scipy.interpolate import interp1d
import ndcube
from sunpy.map import GenericMap

from synthesizAR.instruments import ChannelBase, InstrumentBase
from synthesizAR.util import los_velocity
from synthesizAR.util.decorators import return_quantity_as_tuple
from synthesizAR.instruments.util import add_wave_keys_to_header, extend_celestial_wcs

__all__ = [
    'InstrumentDEM',
    'InstrumentLOSVelocity',
    'InstrumentTemperature'
]


@dataclass
class ChannelDEM(ChannelBase):
    bin_edges: u.Quantity = None

    def __post_init__(self):
        self.log_bin_edges = np.log10(self.bin_edges.to('K').value)
        self.name = f'{self.log_bin_edges[0]:.2f}-{self.log_bin_edges[1]:.2f}'


class InstrumentDEM(InstrumentBase):
    name = 'DEM'

    @u.quantity_input
    def __init__(self, *args, temperature_bin_edges: u.K, **kwargs):
        self.temperature_bin_edges = temperature_bin_edges
        n_bins = temperature_bin_edges.shape[0]-1
        bin_edges = [temperature_bin_edges[[i, i+1]] for i in range(n_bins)]
        self.channels = [ChannelDEM(bin_edges=be) for be in bin_edges]
        super().__init__(*args, **kwargs)

    @property
    @u.quantity_input
    def temperature_bin_centers(self) -> u.K:
        return (self.temperature_bin_edges[1:] + self.temperature_bin_edges[:-1])/2

    def get_instrument_name(self, channel):
        # This ensures that the temperature bin labels are in the header
        return f'{self.name}_{channel.name}'

    @staticmethod
    @return_quantity_as_tuple
    def calculate_intensity_kernel(loop, channel, **kwargs):
        T = loop.electron_temperature
        n = loop.density
        bin_mask = np.where(
            np.logical_and(T >= channel.bin_edges[0], T < channel.bin_edges[1]), 1, 0)
        kernel = n**2 * bin_mask
        return kernel

    def dem_maps_to_cube(self, dem, time_index):
        """
        Convert a list of DEM maps to a DEM NDCube
        """
        # NOTE: this is the format that .observe returns
        return type(self).dem_maps_list_to_cube(
            [dem[c.name][time_index] for c in self.channels],
            self.temperature_bin_centers,
        )

    @staticmethod
    def dem_maps_list_to_cube(dem_maps, temperature_bin_centers):
        # Construct WCS
        compound_wcs = extend_celestial_wcs(dem_maps[0].wcs,
                                            temperature_bin_centers,
                                            'temperature',
                                            'phys.temperature')
        # Stack arrays
        dem_array = u.Quantity([d.quantity for d in dem_maps])

        return ndcube.NDCube(dem_array, wcs=compound_wcs, meta=dem_maps[0].meta)

    @staticmethod
    def calculate_intensity(dem, spectra, header, meta=None):
        """
        Compute intensity from a DEM and a temperature-dependent spectra

        Parameters
        ----------
        dem: `~ndcube.NDCube`
            The first axis should correspond to temperature
        spectra: `~ndcube.NDCube`
        header: `dict` or header-like
            Header information corresponding to the spatial axes of the DEM cube
        meta: `dict`, optional
            Additional metadata
        wavelength
        """
        temperature_bin_centers = dem.axis_world_coords(0)[0]
        wavelength_spectra = spectra.axis_world_coords(1)[0]
        temperature_spectra = spectra.axis_world_coords(0)[0].to(temperature_bin_centers.unit)
        # Interpolate spectral cube to DEM temperatures
        f_interp = interp1d(temperature_spectra.value, spectra.data,
                            axis=0, bounds_error=False, fill_value=0.0)
        spectra_interp = f_interp(temperature_bin_centers.value)
        spectra_interp = spectra_interp * spectra.unit
        # Take dot product between DEM and spectra
        intensity = np.tensordot(spectra_interp, u.Quantity(dem.data, dem.unit), axes=([0], [0]))
        # Construct cube
        wave_header = add_wave_keys_to_header(wavelength_spectra, header)
        wave_header['BUNIT'] = intensity.unit.to_string()
        wave_header['NAXIS'] = len(intensity.shape)
        wave_header['WCSAXES'] = len(intensity.shape)
        meta = {} if meta is None else meta
        meta = {**meta, **wave_header}
        wcs = astropy.wcs.WCS(header=wave_header)
        return ndcube.NDCube(intensity, wcs=wcs, meta=meta)

    def make_slope_map(self, dem, time_index, temperature_bounds=None, em_threshold=None,
                       rsquared_tolerance=0.5, full=False):
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
        data = u.Quantity([dem[c.name][time_index].quantity for c in self.channels]).T
        # Get EM fit array
        em_fit = np.log10(data.value.reshape((np.prod(data.shape[:2]),) + data.shape[2:]).T)
        em_fit[np.logical_or(np.isinf(em_fit), np.isnan(em_fit))] = 0.0  # Filter before fitting
        # Fit to first-order polynomial
        coefficients, rss, _, _, _ = np.polyfit(temperature_fit, em_fit, 1, full=True,)
        # Create mask from correlation coefficient EM threshold
        _, rss_flat, _, _, _ = np.polyfit(temperature_fit, em_fit, 0, full=True,)
        rss = 0.*rss_flat if rss.size == 0 else rss  # returns empty residual when fit is exact
        rsquared = 1. - rss/rss_flat
        rsquared_mask = rsquared.reshape(data.shape[:2]) < rsquared_tolerance
        em_mask = np.any(data < em_threshold, axis=2)
        # Rebuild into a map
        base_meta = self[0].meta.copy()
        base_meta['temp_a'] = 10.**temperature_fit[0]
        base_meta['temp_b'] = 10.**temperature_fit[-1]
        base_meta['bunit'] = ''
        base_meta['detector'] = 'EM slope'
        base_meta['comment'] = 'Linear fit to log-transformed LOS EM'
        plot_settings = self[0].plot_settings.copy()
        plot_settings['norm'] = None
        m = GenericMap(
            coefficients[0, :].reshape(data.shape[:2]),
            base_meta,
            mask=np.stack((em_mask, rsquared_mask), axis=2).any(axis=2),
            plot_settings=plot_settings
        )

        return (m, coefficients, rsquared) if full else m


class InstrumentQuantityBase(InstrumentBase):

    @u.quantity_input
    def __init__(self, *args, **kwargs):
        self.channels = [ChannelBase(name=self.name)]
        super().__init__(*args, average_over_los=True, **kwargs)


class InstrumentLOSVelocity(InstrumentQuantityBase):
    name = 'los_velocity'

    @staticmethod
    def calculate_intensity_kernel(loop, *args, **kwargs):
        observer = kwargs.get('observer')
        if observer is None:
            raise ValueError('Must pass in observer to compute LOS velocity.')
        return los_velocity(loop.velocity_xyz, observer)


class InstrumentTemperature(InstrumentQuantityBase):
    name = 'temperature'

    @staticmethod
    def calculate_intensity_kernel(loop, *args, **kwargs):
        return loop.electron_temperature
