"""
Useful models for cooling of coronal loops.
"""
import astropy.constants as const
import astropy.units as u
import numpy as np

from .scaling_laws import KAPPA_0

__all__ = ['cargill_cooling_time']


@u.quantity_input
def cargill_cooling_time(strand, heating_rate: u.Unit('erg cm-3 s-1')) -> u.s:
    r"""
    Estimate loop cooling time for a given heating rate.

    The cooling time can be estimated using Eq. A2 of
    :cite:t:`cargill_active_2014`,

    .. math::

        \tau_{cool}=\left(\frac{2-\alpha}{1-\alpha}\right)3k_B\left(\frac{1}{\kappa_0^{4-2\alpha}\chi^7}\frac{L^{8-4\alpha}}{(n_0T_0)^{3+2\alpha}}\right)^{1/(11-2\alpha)}.

    The initial temperature :math:`T_0` and density :math:`n_0` are estimated
    from the EBTEL initial conditions using the input heating rate.

    Parameters
    ----------
    strand: `~synthesizAR.Strand`
    heating_rate: `~astropy.units.Quantity`
        Heating rate used to determine the initial temperature and density from which
        the plasmaw will cool.
    alpha: `float`, optional
        The temperature dependence of the radiative loss function. By default,
        this is :math:`\alpha=-\frac{1}{2}`.
    """
    half_length = strand.length/2
    # set some constants
    alpha = -0.5
    chi = 6e-20*u.erg*u.cm**3/u.s/(u.K**alpha)
    c1,c2,c3 = 2.0,0.9,0.6
    gamma = 5./3.
    # estimate max n0T0
    T0 = c2*(7.*half_length**2*heating_rate/2./KAPPA_0)**(2./7.)
    top_term = heating_rate - 2.*KAPPA_0*(T0**(3.5))/(7.*(c2**2.5)*c3*(half_length**2)*gamma)
    bottom_term = c1*chi*(T0**alpha)*(1. - c2/c3/gamma)
    n0 = np.sqrt(top_term/bottom_term)
    n0T0 = n0*T0
    # Cargill cooling expression
    term1 = (2. - alpha)/(1. - alpha)
    term2 = (KAPPA_0**(4. - 2.*alpha))*(chi**7)
    term3 = ((half_length)**(8. - 4.*alpha))/(n0T0**(3+2.*alpha))
    return term1*3.*const.k_B*(1/term2*term3)**(1/(11. - 2.*alpha))
