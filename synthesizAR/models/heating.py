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
    r"""
    Heating rate dependent on the strand length and average field strength along the strand,

    .. math::

        H = H_0\left(\frac{\bar{B}}{B_0}\right)^\alpha\left(\frac{L_0}{L}\right)^\beta,

    where :math:`H_0,B_0,L_0` are the nominal heating rate, field strength, and loop length,
    respectively, :math:`\bar{B}` is the average field strength, and :math:`L` is the full
    loop length.

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
    B_0: `~astropy.units.Quantity`, optional
        Nominal value of the field strength.
    L_0: `~astropy.units.Quantity`, optional
        Nominal value of the strand length.
    """
    B_avg = strand.field_strength_average
    return H_0 * ((B_avg / B_0)**alpha) * ((L_0 / strand.length)**beta)


@u.quantity_input
def free_magnetic_energy_density(strand, stress_level=0.3) -> u.Unit('erg cm-3'):
    r"""
    Calculate available free energy of the magnetic field using
    Eq. 1 of :cite:t:`reep_diagnosing_2013`,

    .. math::

        E_B = \frac{(\epsilon B_p)^2}{8\pi},

    where :math:`B_p` is the potential component of the field and :math:`epsilon`
    parameterizes the stress level.

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
