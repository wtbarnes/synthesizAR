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

__all__ = ['Element', 'Ion', 'list_elements']


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


class Ion(fiasco.Ion):
    """
    Subclass of fiasco ion that adds functionality for calculating level populations.
    """
    
    @fiasco.util.needs_dataset('scups')
    def effective_collision_strength(self):
        """
        Calculate the effective collision strength or the Maxwellian-averaged collision
        strength, typically denoted by upsilon.
        
        Note
        ----
        Need a more efficient way of calculating upsilon for all transitions. Current method is 
        slow for ions with many transitions, e.g. Fe IX and Fe XI
        """
        energy_ratio = np.outer(const.k_B.cgs*self.temperature,
                                1.0/self._scups['delta_energy'].to(u.erg))
        upsilon = np.array(list(map(self.burgess_tully_descale, self._scups['bt_t'],
                                    self._scups['bt_upsilon'], energy_ratio.T, self._scups['bt_c'],
                                    self._scups['bt_type'])))
        upsilon = u.Quantity(np.where(upsilon > 0., upsilon, 0.))
        return upsilon.T
    
    @fiasco.util.needs_dataset('elvlc', 'scups')
    def electron_collision_rate(self):
        """
        Calculates the collision rate for de-exciting and exciting collisions for electrons
        """
        c = (const.h.cgs**2)/((2. * np.pi * const.m_e.cgs)**(1.5) * np.sqrt(const.k_B.cgs))
        upsilon = self.effective_collision_strength()
        omega_upper = 2.*self._elvlc['J'][self._scups['upper_level'] - 1] + 1.
        omega_lower = 2.*self._elvlc['J'][self._scups['lower_level'] - 1] + 1.
        dex_rate = c*upsilon/np.sqrt(self.temperature[:, np.newaxis])/omega_upper
        energy_ratio = np.outer(1./const.k_B.cgs/self.temperature,
                                self._scups['delta_energy'].to(u.erg))
        ex_rate = omega_upper/omega_lower*dex_rate*np.exp(-energy_ratio)
        
        return dex_rate, ex_rate
    
    @fiasco.util.needs_dataset('psplups', default=(None, None))
    def proton_collision_rate(self):
        """
        Calculates the collision rate for de-exciting and exciting collisions for protons
        """
        # Create scaled temperature--these are not stored in the file
        bt_t = np.vectorize(np.linspace, excluded=[0, 1], otypes='O')(
            0, 1, [ups.shape[0] for ups in self._psplups['bt_rate']])
        # Get excitation rates directly from scaled data
        energy_ratio = np.outer(const.k_B.cgs*self.temperature,
                                1.0/self._psplups['delta_energy'].to(u.erg))
        ex_rate = np.array(list(map(self.burgess_tully_descale, bt_t,
                                    self._psplups['bt_rate'], energy_ratio.T, self._psplups['bt_c'], 
                                    self._psplups['bt_type'])))
        ex_rate = u.Quantity(np.where(ex_rate > 0., ex_rate, 0.), u.cm**3/u.s).T
        # Calculation de-excitation rates from excitation rate
        omega_upper = 2.*self._elvlc['J'][self._psplups['upper_level'] - 1] + 1.
        omega_lower = 2.*self._elvlc['J'][self._psplups['lower_level'] - 1] + 1.
        dex_rate = (omega_lower / omega_upper) * ex_rate*np.exp(1. / energy_ratio)
        
        return dex_rate, ex_rate
    
    @fiasco.util.needs_dataset('wgfa', 'elvlc', 'scups')
    @u.quantity_input
    def level_populations(self, density: u.cm**(-3), include_protons=True):
        """
        Calculate populations of all energy levels as a function temperature and density.

        Note
        ----
        There are two very inefficient loops over temperature and density. This could be
        vectorized, but for some ions (e.g. Fe IX, XI) the resulting arrays would be too
        large to fit into memory.
        """
        def collect(a, b, c, axis):
            return c[np.where(a == b)].sum(axis=axis)
        collect_v = np.vectorize(collect, excluded=[0, 2, 3])
        level = self._elvlc['level']
        upper_level = self._scups['upper_level']
        lower_level = self._scups['lower_level']
        coeff_matrix = np.zeros(self.temperature.shape + level.shape + level.shape)/u.s
        
        # Radiative decays
        a_diagonal = collect_v(
            self._wgfa['upper_level'], level, self._wgfa['A'].value, None) * self._wgfa['A'].unit
        # Decay out of current level
        coeff_matrix[:, level - 1, level - 1] -= a_diagonal
        # Decay into current level from upper levels
        coeff_matrix[:, self._wgfa['lower_level']-1, self._wgfa['upper_level']-1] += self._wgfa['A']

        # Proton and electron collision rates
        dex_rate_e, ex_rate_e = self.electron_collision_rate()
        ex_diagonal = np.array([collect(lower_level, l, ex_rate_e.value.T, 0)
                                for l in level]).T*ex_rate_e.unit
        dex_diagonal = np.array([collect(upper_level, l, dex_rate_e.value.T, 0)
                                 for l in level]).T*dex_rate_e.unit
        if include_protons and self._psplups is not None:
            p2e_ratio = proton_ratio(self.temperature)
            dex_rate_p, ex_rate_p = self.proton_collision_rate()
            upper_level_p = self._psplups['upper_level']
            lower_level_p = self._psplups['lower_level']
            ex_diagonal_p = np.array([collect(lower_level_p, l, ex_rate_p.value.T, 0)
                                      for l in level]).T*ex_rate_p.unit
            dex_diagonal_p = np.array([collect(upper_level_p, l, dex_rate_p.value.T, 0)
                                       for l in level]).T*dex_rate_p.unit

        populations = np.zeros(self.temperature.shape + density.shape + level.shape)
        b = np.zeros(self.temperature.shape+level.shape)
        b[:, -1] = 1.0
        for i_d, d in enumerate(density):
            coeff_matrix_copy = coeff_matrix.copy()
            # Excitation and de-excitation out of current state
            coeff_matrix_copy[:, level - 1, level - 1] -= d*(dex_diagonal + ex_diagonal)
            # De-excitation from upper states and excitation from lower states
            coeff_matrix_copy[:, lower_level - 1, upper_level - 1] += d*dex_rate_e
            coeff_matrix_copy[:, upper_level - 1, lower_level - 1] += d*ex_rate_e

            # Same processes as above, but for protons
            if include_protons and self._psplups is not None:
                coeff_matrix_copy[:, level-1, level-1] -= d*p2e_ratio[:, np.newaxis]*(
                    dex_diagonal_p + ex_diagonal_p)
                coeff_matrix_copy[:, lower_level_p - 1, upper_level_p - 1] += (
                    d*p2e_ratio[:, np.newaxis]*dex_rate_p)
                coeff_matrix_copy[:, upper_level_p - 1, lower_level_p - 1] += (
                    d*p2e_ratio[:, np.newaxis]*ex_rate_p)

            coeff_matrix_copy[:, -1, :] = 1. * coeff_matrix_copy.unit
            pop = np.linalg.solve(coeff_matrix_copy.value, b)
            pop = np.where(pop < 0., 0., pop)
            pop /= pop.sum(axis=1)[:, np.newaxis]
            populations[:, i_d, :] = pop

        return u.Quantity(populations)

    @fiasco.util.needs_dataset('wgfa', default=(None, None))
    @u.quantity_input
    def emissivity(self, density: u.cm**(-3), include_energy=False, **kwargs):
        """
        Calculate emissivity for all lines as a function of temperature and density
        """
        populations = self.level_populations(
            density, include_protons=kwargs.get('include_protons', True))
        if populations is None:
            return (None, None)
        wavelengths = np.fabs(self._wgfa['wavelength'])
        # Exclude 0 wavelengths which correspond to two-photon decays
        upper_levels = self._wgfa['upper_level'][wavelengths != 0*u.angstrom]
        a_values = self._wgfa['A'][wavelengths != 0*u.angstrom]
        wavelengths = wavelengths[wavelengths != 0*u.angstrom]
        if include_energy:
            energy = const.h.cgs*const.c.cgs/wavelengths.to(u.cm)
        else:
            energy = 1.*u.photon
        emissivity = populations[:, :, upper_levels - 1]*(a_values*energy)
        
        return wavelengths, emissivity


@u.quantity_input
def proton_ratio(temperature: u.K):
    """
    Calculate ratio between proton and electron density as a function of temperature. 
    See Eq. 7 of [1]_.

    References
    ----------
    .. [1] Young, P. et al., 2003, ApJS, `144 135 <http://adsabs.harvard.edu/abs/2003ApJS..144..135Y>`_
    """
    denominator = None
    for el_name in list_elements():
        el = fiasco.Element(el_name, temperature=temperature)
        for ion in el:
            if denominator is None:
                denominator = (ion._ioneq['chianti']['ionization_fraction']
                               * ion.abundance * ion.charge_state)
            else:
                denominator += (ion._ioneq['chianti']['ionization_fraction']
                                * ion.abundance * ion.charge_state)

    el_h = fiasco.Element('hydrogen', temperature=temperature)
    numerator = el_h[1].abundance*el_h[1]._ioneq['chianti']['ionization_fraction']
    f_ratio = interp1d(el_h[1]._ioneq['chianti']['temperature'], numerator/denominator,
                       fill_value='extrapolate')

    return f_ratio(temperature.value)


def list_elements():
    """
    List all available elements in the CHIANTI database.
    """
    with h5py.File(fiasco.defaults['hdf5_dbase_root']) as hf:
        elements = []
        for k in hf.keys():
            try:
                elements.append(plasmapy.atomic.atomic_symbol(k.capitalize()))
            except plasmapy.atomic.names.InvalidParticleError:
                continue
    return elements
