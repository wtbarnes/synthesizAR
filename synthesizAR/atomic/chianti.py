"""
Wrappers for aggregating CHIANTI data and doing fundamental atomic physics calculations.

Note
----
Question of whether this can all be done with the fiasco ion object or whether some
sort of wrapper will always be needed
"""
import os
import warnings

import numpy as np
import h5py
from scipy.interpolate import interp1d
import astropy.units as u
import astropy.constants as const
import plasmapy.atomic
import fiasco

__all__ = ['Element']


class Element(fiasco.Element):
        
    @u.quantity_input
    def non_equilibrium_ionization(self, time: u.s, temperature: u.K, density: u.cm**(-3),
                                   rate_matrix=None, initial_condition=None):
        """
        Compute the ionization fraction in non-equilibrium for a given temperature and density
        timeseries.
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

        return u.Quantity(y)
