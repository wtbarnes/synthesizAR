"""
Analysis functions related to differential emission measures
"""
import warnings

import astropy.units as u
import numpy as np
import sunpy.map

__all__ = ['make_slope_map']


def make_slope_map(dem,
                   temperature_bounds=None,
                   em_threshold=None,
                   rsquared_tolerance=0.5,
                   mask_negative=True):
    r"""
    Calculate emission measure slope :math:`a` in each pixel

    Create map of emission measure slopes by fitting :math:`\mathrm{EM}\sim T^a` for a
    given temperature range. A slope is masked if a value between the `temperature_bounds`
    is less than :math:`\mathrm{EM}`. Additionally, the "goodness-of-fit" is evaluated using
    the correlation coefficient, :math:`r^2=1 - R_1/R_0`, where :math:`R_1` and :math:`R_0`
    are the residuals from the first and zeroth order polynomial fits, respectively. We mask
    the slope if :math:`r^2` is less than `rsquared_tolerance`.

    Parameters
    ----------
    dem : `ndcube.NDCube`
    temperature_bounds : `~astropy.units.Quantity`, optional
    em_threshold : `~astropy.units.Quantity`, optional
        Mask slope if any emission measure in the fit interval is below this value
    rsquared_tolerance : `float`
        Mask any slopes with a correlation coefficient, :math:`r^2`, below this value
    mask_negative : `bool`
    """
    # TODO: move this somewhere more visible, e.g. synthesizAR
    if temperature_bounds is None:
        temperature_bounds = u.Quantity((1e6, 4e6), u.K)
    if em_threshold is None:
        em_threshold = u.Quantity(1e25, u.cm**(-5))
    # Get temperature fit array
    temperature_bin_centers = dem.axis_world_coords(0)[0]
    index_temperature_bounds = np.where(np.logical_and(
        temperature_bin_centers >= temperature_bounds[0],
        temperature_bin_centers <= temperature_bounds[1]
    ))
    temperature_fit = np.log10(
        temperature_bin_centers[index_temperature_bounds].to_value(u.K))
    if temperature_fit.size < 3:
        warnings.warn(f'Fitting to fewer than 3 points in temperature space: {temperature_fit}')
    # Cut on temperature
    data = u.Quantity(dem.data, dem.unit).T
    data = data[...,index_temperature_bounds].squeeze()
    # Get EM fit array
    em_fit = np.log10(data.value.reshape((np.prod(data.shape[:2]),) + data.shape[2:]).T)
    em_fit[np.logical_or(np.isinf(em_fit), np.isnan(em_fit))] = 0.0  # Filter before fitting
    # Fit to first-order polynomial
    coefficients, rss, _, _, _ = np.polyfit(temperature_fit, em_fit, 1, full=True,)
    slope_data = coefficients[0, :].reshape(data.shape[:2])
    # Apply masks
    _, rss_flat, _, _, _ = np.polyfit(temperature_fit, em_fit, 0, full=True,)
    rss = 0.*rss_flat if rss.size == 0 else rss  # returns empty residual when fit is exact
    rsquared = 1. - rss/rss_flat
    rsquared_mask = rsquared.reshape(data.shape[:2]) < rsquared_tolerance
    em_mask = np.any(data < em_threshold, axis=2)
    mask_list = [rsquared_mask, em_mask]
    if dem.mask is not None:
        mask_list.append(dem.mask.T[..., index_temperature_bounds].squeeze().any(axis=2))
    if mask_negative:
        mask_list.append(slope_data < 0)
    combined_mask = np.stack(mask_list, axis=2).any(axis=2).T
    # Build new map
    header = dem.wcs.low_level_wcs._wcs[0].to_header()
    header['temp_a'] = 10.**temperature_fit[0]
    header['temp_b'] = 10.**temperature_fit[-1]
    slope_map = sunpy.map.GenericMap(slope_data.T, header, mask=combined_mask)
    return slope_map
