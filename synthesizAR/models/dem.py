"""
Analytical differential emission measure (DEM) models.
"""
import astropy.units as u
import numpy as np

__all__ = ['guennou_dem']


@u.quantity_input
def guennou_dem(temperature: u.K, T_max: u.K, EM_max: u.cm**(-5), alpha, sigma, sigma_fw=0.15) -> u.cm**(-5):
    r"""
    Analytical DEM model of :cite:t:`guennou_can_2013` comprised of power-law and Gaussian components.

    Parameters
    ----------
    temperature: `~astropy.units.Quantity`
    T_max: `~astropy.units.Quantity`
        Temperature at which the emission measure is maximum
    EM_max: `~astropy.units.quantity`
        Maximum value of the emission measure
    alpha: `float`
        Power-law index characterizing the cool part of the distribution. This is also called the
        emission measure "slope" in log-log space.
    sigma: `float`
        Width of the Gaussian describing the hot part of the distribution.
        This width is parameterized in :math:`\log_{10}(T)`.
    sigma_fw: `float`, optional
        Width of the Gaussian connecting the cool power-law and hot Gaussian parts of the model.
        In general, this does not need to change.
    """
    T_0 = _tangent_point(T_max, alpha, sigma_fw)
    dem_low = _guennou_dem_low(temperature, T_0, T_max, alpha, sigma_fw)
    dem_high = _guennou_dem_high(temperature, T_max, sigma) * sigma / sigma_fw
    dem = _guennou_dem_connection(temperature, T_max, sigma_fw)
    dem[temperature < T_0] = dem_low[temperature < T_0]
    dem[temperature > T_max] = dem_high[temperature > T_max]
    dem = dem * EM_max * sigma_fw * np.sqrt(2*np.pi)
    return dem


def _tangent_point(T_P, alpha, sigma):
    "Point of tangency between Gaussian and power-law in log-log space"
    return T_P * 10**(-alpha * (sigma**2) * np.log(10))


def _gaussian(x, sigma):
    return np.exp(-((x/sigma)**2)/2)/sigma/np.sqrt(2*np.pi)


def _guennou_dem_low(temperature, T_0, T_P, alpha, sigma):
    x = np.log10(T_0.to_value('K')) - np.log10(T_P.to_value('K'))
    return _gaussian(x, sigma) * (temperature / T_0).decompose()**alpha


def _guennou_dem_high(temperature, T_P, sigma):
    x = np.log10(temperature.to_value('K')) - np.log10(T_P.to_value('K'))
    return _gaussian(x, sigma)


def _guennou_dem_connection(temperature, T_P, sigma):
    x = np.log10(temperature.to_value('K')) - np.log10(T_P.to_value('K'))
    return _gaussian(x, sigma)
