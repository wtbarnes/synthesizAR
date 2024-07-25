"""
Analysis functions related to differential emission measures
"""
import astropy.units as u
import numpy as np
import sunpy.map

__all__ = ['log_log_linear_fit', 'make_slope_map']


def log_log_linear_fit(x, y, x_a, x_b, apply_log_transform=True):
    """
    Perform a power-law fit by calculating a linear fit to
    a log-transformed quantity over a specified range.

    Parameters
    ----------
    x
    y
    apply_log_transform

    Returns
    -------
    coefficients
    x_fit
    y_fit
    r_squared
    """
    if apply_log_transform:
        x = np.log10(x)
        y = np.log10(y)
        x_a = np.log10(x_a)
        x_b = np.log10(x_b)
    idx = np.where(np.logical_and(x>=x_a, x<=x_b))
    x_fit = x[idx]
    y_fit = y[idx]
    # Do not weight non-finite entries
    weights = np.where(np.logical_and(y_fit>0, np.isfinite(y_fit)), 1, 0)
    # Ignore fits on two or less points or where all but two or less of the
    # weights are zero
    if x_fit.size < 3 or np.where(weights == 1)[0].size < 3:
        raise ValueError('Fitting to fewer than three points')
    coeff, rss, _, _, _ = np.polyfit(x_fit, y_fit, 1, full=True, w=weights)
    # Calculate the zeroth-order fit in order to find the correlaton
    _, rss_flat, _, _, _ = np.polyfit(x_fit, y_fit, 0, full=True, w=weights)
    rsquared = 1 - rss[0]/rss_flat[0]
    return coeff, x_fit, y_fit, rsquared


def _iterate_fit_range(x, y, x_a_min, tol):
    """
    Iterate on bounds over which to fit
    """
    i_max = np.argmax(y)
    r2_vals = []
    coefs = []
    x_fit = []
    y_fit = []
    for i in range(i_max):
        x_b = x[i_max-i]
        x_a = max(x_b-tol, x_a_min),
        if x_b - x_a < tol:
            break
        coef, xf, yf, r2 = log_log_linear_fit(x, y, x_a, x_b, apply_log_transform=False)
        coefs.append(coef)
        r2_vals.append(r2)
        x_fit.append(xf)
        y_fit.append(yf)
    # Find values that maximize r^2 value
    i_max = np.argmax(r2_vals)
    return coefs[i_max], x_fit[i_max], y_fit[i_max], r2_vals[i_max]


def make_slope_map(dem,
                   temperature_bounds=None,
                   max_upper_bound=None,
                   em_threshold=None,
                   rsquared_tolerance=0.5,
                   iterate_bounds=False,
                   mask_negative=False):
    r"""
    Calculate emission measure slope :math:`a` in each pixel

    Create map of emission measure slopes by fitting :math:`\mathrm{EM}\sim T^a` for a
    given temperature range. Additionally, the "goodness-of-fit" is evaluated using
    the correlation coefficient, :math:`r^2=1 - R_1/R_0`, where :math:`R_1` and :math:`R_0`
    are the residuals from the first and zeroth order polynomial fits, respectively. We mask
    the slope if :math:`r^2` is less than `rsquared_tolerance`.

    Parameters
    ----------
    dem : `ndcube.NDCube`
    temperature_bounds : `~astropy.units.Quantity`, optional
        Range over which to fit the EM distribution. Defaults to the
        temperature at which the EM distribution is peaked and 25% of
        that value.
    em_threshold : `~astropy.units.Quantity`, optional
        Mask slope if the total emission measure in a pixel is below this value
    rsquared_tolerance : `float`, optional
        Mask any slopes with a correlation coefficient, :math:`r^2`, below this value
    mask_negative : `bool`
        Mask the pixel if the slope is negative.
    """
    # Unwrap EM into list of values that are above threshold
    if em_threshold is None:
        em_threshold = u.Quantity(1e25, u.cm**(-5))
    total_dem = u.Quantity(dem.data.sum(axis=0), dem.unit)
    is_valid = np.where(total_dem>=em_threshold)
    log_em_valid = np.log10(dem.data[:, *is_valid]).T
    log_em_valid = np.where(np.isfinite(log_em_valid), log_em_valid, 0.0)

    # Get temperature bounds
    # NOTE: This allows for passing the following:
    # 1. None
    # 2. (None, None)
    # 3. (None, scalar), (scalar, None), (scalar, scalar)
    # 4. (array, scalar), (scalar, array), (array, array)
    temperature_bin_centers = dem.axis_world_coords(0)[0]
    if temperature_bounds is None:
        temperature_bounds = (None, None)
    temperature_bounds = list(temperature_bounds)
    # Upper bound
    if temperature_bounds[1] is None:
        # Default to the temperature at which the EM peaks
        idx_peak = np.argmax(log_em_valid, axis=1) - 1
        temperature_bounds[1] = temperature_bin_centers[idx_peak]
    if not temperature_bounds[1].shape:
        temperature_bounds[1] = np.tile(temperature_bounds[1], log_em_valid.shape[0])
    # Apply limit to upper bound
    if max_upper_bound is not None:
        temperature_bounds[1] = np.where(temperature_bounds[1]>max_upper_bound,
                                         max_upper_bound,
                                         temperature_bounds[1])
    # Lower bound
    if temperature_bounds[0] is None:
        # Default to 0.25 of the upper bound, i.e. 1 MK to 4 MK
        temperature_bounds[0] = 0.25*temperature_bounds[1]
    if not temperature_bounds[0].shape:
        temperature_bounds[0] = np.tile(temperature_bounds[0], log_em_valid.shape[0])
    temperature_bounds = u.Quantity(temperature_bounds).T
    log_temperature_bounds = np.log10(temperature_bounds.to_value('K'))
    log_temperature_bin_centers = np.log10(temperature_bin_centers.to_value('K'))

    # Iterate through all valid EM arrays and temperature bounds
    slopes = np.full(log_em_valid.shape[:1], np.nan)
    rsquared = np.zeros(slopes.shape)
    for i, (log_em, (log_Ta, log_Tb)) in enumerate(zip(log_em_valid, log_temperature_bounds)):
        try:
            if iterate_bounds:
                coeff, _, _, r2 = _iterate_fit_range(log_temperature_bin_centers,
                                                     log_em,
                                                     log_Ta,
                                                     0.6)
            else:
                coeff, _, _, r2 = log_log_linear_fit(log_temperature_bin_centers,
                                                     log_em,
                                                     log_Ta,
                                                     log_Tb,
                                                     apply_log_transform=False)
        except ValueError:
            # Don't set any value if there aren't enough points to fit
            continue
        slopes[i] = coeff[0]
        rsquared[i] = r2

    # Rebuild into arrays
    slope_array = np.full(total_dem.shape, np.nan)
    slope_array[is_valid] = slopes
    rsquared_array = np.zeros(slope_array.shape)
    rsquared_array[is_valid] = rsquared

    # Apply masks
    rsquared_mask = rsquared_array < rsquared_tolerance
    nan_mask = np.isnan(slope_array)  # This includes cases where total EM is below threshold
    mask_list = [rsquared_mask, nan_mask]
    if dem.mask is not None:
        mask_list.append(dem.mask.all(axis=0))
    if mask_negative:
        mask_list.append(slope_array < 0)
    combined_mask = np.array(mask_list).any(axis=0)

    # Build new map
    header = dem.wcs.low_level_wcs._wcs[0].to_header()
    slope_map = sunpy.map.GenericMap(slope_array, header, mask=combined_mask)
    return slope_map
