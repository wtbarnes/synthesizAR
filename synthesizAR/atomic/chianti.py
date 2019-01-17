"""
Wrappers for aggregating CHIANTI data and doing fundamental atomic physics calculations.

Note
----
Question of whether this can all be done with the fiasco ion object or whether some
sort of wrapper will always be needed
"""
import warnings

import numpy as np
import astropy.units as u
import fiasco

__all__ = ['Element']


class Element(fiasco.Element):
        
    @u.quantity_input
    def non_equilibrium_ionization(self, time: u.s, temperature: u.K, density: u.cm**(-3),
                                   rate_matrix=None, initial_condition=None, check_solution=True):
        """
        Compute the ionization fraction in non-equilibrium for a given temperature and density
        timeseries.

        Parameters
        ----------
        time : `~astropy.units.Quantity`
        temperature : `~astropy.units.Quantity`
        density : `~astropy.units.Quantity`
        rate_matrix : `~astropy.units.Quantity`, optional
            Precomputed matrix of ionization and recombination rates
        initial_condition : `~astropy.units.Quantity`
            Precomputed initial conditions; use equilibrium solution by default
        check_solution : `bool`, optional
            If True, check that the conditions of [1]_ are satisfied

        References
        ----------
        .. [1] Macneice, P., 1984, Sol Phys, `90, 357 <http://adsabs.harvard.edu/abs/1984SoPh...90..357M>`_
        """
        if rate_matrix is None:
            rate_matrix = self._rate_matrix()
        if initial_condition is None:
            initial_condition = self.equilibrium_ionization(rate_matrix=rate_matrix)
        interpolate_indices = [np.abs(self.temperature - t).argmin() for t in temperature]
        y = np.zeros(time.shape + (self.atomic_number + 1,))
        y[0, :] = initial_condition[interpolate_indices[0], :]

        identity = u.Quantity(np.eye(self.atomic_number + 1))
        for i in range(1, time.shape[0]):
            dt = time[i] - time[i-1]
            term1 = identity - density[i] * dt/2. * rate_matrix[interpolate_indices[i], :, :]
            term2 = identity + density[i-1] * dt/2. * rate_matrix[interpolate_indices[i-1], :, :]
            y[i, :] = np.linalg.inv(term1) @ term2 @ y[i-1, :]
            y[i, :] = np.fabs(y[i, :])
            y[i, :] /= y[i, :].sum()

        if check_solution:
            eps_d = 0.1
            eps_r = 0.6
            if (np.fabs(y[1:, :] - y[:-1, :]) > eps_d).any():
                warnings.warn('Condition 1 of Macneice et al. (1984) is not satisfied. Consider '
                              'choosing a smaller timestep.')
            if np.logical_or(y[1:, :]/y[:-1, :] > 10**(eps_r), y[1:, :]/y[:-1, :] < 10**(-eps_r)).any():
                warnings.warn('Condition 2 of Macneice et al. (1984) is not satisfied. Consider '
                              'choosing a smaller timestep.')

        return u.Quantity(y)
