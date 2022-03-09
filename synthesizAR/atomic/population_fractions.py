"""
Functions for computing population fractions as a function of time
"""
import warnings

import numpy as np
import astropy.units as u
from scipy.interpolate import interp1d

__all__ = ['equilibrium_ionization', 'non_equilibrium_ionization']


@u.quantity_input
def equilibrium_ionization(element, temperature, **kwargs):
    """
    Compute the ionization fraction in equilibrium for a given temperature array.

    Parameters
    ----------
    element : `~fiasco.Element`
    temperature : `~astropy.units.Quantity`
    """
    interp_kwargs = {
        'kind': 'cubic',
        'fill_value': 'extrapolate',
    }
    interp_kwargs.update(kwargs)
    f_interp = interp1d(element.temperature.to(temperature.unit).value,
                        element.equilibrium_ionization.value,
                        axis=0,
                        **interp_kwargs)
    ioneq_interp = f_interp(temperature.value)
    return u.Quantity(ioneq_interp)


@u.quantity_input
def non_equilibrium_ionization(element, time: u.s, temperature: u.K, density: u.cm**(-3),
                               check_solution=True):
    """
    Compute the ionization fraction in non-equilibrium for a given temperature and density
    timeseries.

    Parameters
    ----------
    element : `~fiasco.Element`
        Element to compute the non-equilibrium population fractions for.
    time : `~astropy.units.Quantity`
    temperature : `~astropy.units.Quantity`
    density : `~astropy.units.Quantity`
    check_solution : `bool`, optional
        If True, check that the conditions of [1]_ are satisfied

    References
    ----------
    .. [1] Macneice, P., 1984, Sol Phys, `90, 357 <http://adsabs.harvard.edu/abs/1984SoPh...90..357M>`_
    """
    # Find index of the rate_matrix array (corresponding to the temperature) that is closest to
    # each value of the input temperature. This is then used to select appropriate rate_matrix
    # slice at each time step.
    interpolate_indices = [np.abs(element.temperature - t).argmin() for t in temperature]
    y = np.zeros(time.shape + (element.atomic_number + 1,))
    # Initialize with the equilibrium populations
    y[0, :] = element.equilibrium_ionization[interpolate_indices[0], :]

    identity = u.Quantity(np.eye(element.atomic_number + 1))
    for i in range(1, time.shape[0]):
        dt = time[i] - time[i-1]
        term1 = identity - density[i] * dt/2. * element._rate_matrix[interpolate_indices[i], ...]
        term2 = identity + density[i-1] * dt/2. * element._rate_matrix[interpolate_indices[i-1], ...]
        y[i, :] = np.linalg.inv(term1) @ term2 @ y[i-1, :]
        y[i, :] = np.fabs(y[i, :])
        y[i, :] /= y[i, :].sum()

    if check_solution:
        eps_d = 0.1
        eps_r = 0.6
        if (np.fabs(y[1:, :] - y[:-1, :]) > eps_d).any():
            warnings.warn('Condition 1 of Macneice et al. (1984) is not satisfied. '
                          'Consider choosing a smaller timestep.')
        if np.logical_or(y[1:, :]/y[:-1, :] > 10**(eps_r), y[1:, :]/y[:-1, :] < 10**(-eps_r)).any():
            warnings.warn('Condition 2 of Macneice et al. (1984) is not satisfied. '
                          'Consider choosing a smaller timestep.')

    return u.Quantity(y)
