"""
A collection of commonly used models for estimating energy density and heating rates.
"""
import astropy.units as u
import numpy as np

__all__ = ['b_over_l_scaling', 'free_magnetic_energy_density']


@u.quantity_input
def b_over_l_scaling(strand,
                     H_0=0.0738*u.Unit('erg cm-3 s-1'),
                     alpha=0.3,
                     beta=0.2,
                     B_0=76*u.G,
                     L_0=29*u.Mm) -> u.Unit('erg cm-3 s-1'):
    """
    Heating rate dependent on the strand length and average field strength along the strand.

    .. note:: The default values for all parameters are taken from
              :cite:t:`ugarte-urra_magnetic_2019`.

    Parameters
    ----------
    strand: `synthesizAR.Strand`
    H_0: `~astropy.units.Quantity`, optional
        Nominal heating rate.
    alpha: `float`, optional
        Dependence on average field strength.
    beta: `float`, optional
        Dependence on strand length.
    B0: `~astropy.units.Quantity`, optional
        Nominal value of the field strength.
    L0: `~astropy.units.Quantity`, optional
        Nominal value of the strand length.
    """
    B_avg = strand.field_strength_average
    return H_0 * ((B_avg / B_0)**alpha) * ((L_0 / strand.length)**beta)


@u.quantity_input
def free_magnetic_energy_density(strand, stress_level=0.3) -> u.Unit('erg cm-3'):
    """
    Calculate available free energy of the magnetic field using
    Eq. 1 of :cite:t:`reep_diagnosing_2013`.

    Parameters
    ----------
    strand: `synthesizAR.Strand`
    stress_level: `float`
    """
    B_avg = strand.field_strength_average.to_value('G')
    # FIXME: Deal with the units properly here. In later versions of astropy,
    # there is a built-in equivalency that will potentially help with this.
    # See https://docs.astropy.org/en/latest/units/equivalencies.html#magnetic-flux-density-and-field-strength-equivalency
    energy_density = ((stress_level*B_avg)**2)/(8.*np.pi)
    return u.Quantity(energy_density, 'erg cm-3')
