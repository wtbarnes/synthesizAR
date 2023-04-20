"""
Functions for computing population fractions as a function of time
"""
import warnings

import numpy as np
import astropy.units as u
from scipy.interpolate import interp1d

__all__ = ['equilibrium_ionization', 'non_equilibrium_ionization', 'effective_temperature']


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

    Compute the time-dependent ionization fractions of a given element out of equilibrium.
    This method is described in [1]_ and in more detail in the Appendix of [2]_.

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
    .. [2] Barnes, W.T. et al., 2019, ApJ, `880, 56 <https://ui.adsabs.harvard.edu/abs/2019ApJ...880...56B>`_
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


def effective_temperature(element, time, temperature, density, **kwargs):
    """
    Compute the effective temperature for a plasma out of ionization equilibrium.

    For a given time-dependent temperature and density, use the time-dependent
    ionization fractions of a given element to compute the temperature one would
    infer if the plasma were assumed to be in equilibrium. This method is described
    in detail in [3]_. This method is a good proxy for understanding how the assumption
    of ionization equilibrium may lead to an underestimation of the actual underlying
    plasma temperature.

    References
    ----------
    .. [3] Bradshaw, S.J., 2009, A&A, `502, 409 <http://adsabs.harvard.edu/abs/2009A%26A...502..409B>`_
    """
    frac_nei = non_equilibrium_ionization(element, time, temperature, density, **kwargs)
    # NOTE: For each timestep, the line below does the following:
    # 1. Compute the absolute difference between the ionization state out of equilibrium and the
    #    equilibrium state at each temperature.
    # 2. Sum these differences over all ionization states for each temperature
    # 3. Find the (temperature) index whose sum is the smallest.
    # 4. Use that index to find the corresponding equilibrium temperature
    # In this way, the effective temperature is the equilibrium temperature that has the set of
    # ionization states that most closely matches the ionization state out of equilibrium at that
    # given timestep.
    ioneq = element.equilibrium_ionization
    return element.temperature[[(np.fabs(ioneq - nei)).sum(axis=1).argmin() for nei in frac_nei]]
