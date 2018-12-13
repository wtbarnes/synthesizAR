"""
Implementations of various coronal loop scaling laws
"""
import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy.special import beta, betaincinv

__all__ = ['MartensScalingLaws']


class MartensScalingLaws(object):
    """
    Coronal loop scaling laws of [1]_

    Parameters
    ----------
    s : `~astropy.units.Quantity`
        Field-aligned loop coordinate
    loop_length : `~astropy.units.Quantity`
        Loop half-length
    maximum_temperature : `~astropy.units.Quantity`
        Maximum temperature at loop apex
    base_temperature : `~astropy.units.Quantity`, optional
        Temperature at the loop base, i.e. the chromosphere
    alpha : `float`, optional
        Temperature dependence of the heating rate
    gamma : `float`, optional
        Temperature dependence of the radiative loss rate

    References
    ----------
    .. [1] Martens, P., 2010, ApJ, `714, 1290 <http://adsabs.harvard.edu/abs/2010ApJ...714.1290M>`_
    """

    @u.quantity_input
    def __init__(self, s: u.cm, loop_length: u.cm, max_temperature: u.K, alpha=0, gamma=0.5,
                 base_temperature: u.K=0.01*u.MK):
        self.max_temperature = max_temperature
        self.base_temperature = base_temperature
        self.s = s
        self.loop_length = loop_length
        self.alpha = alpha
        self.gamma = gamma

    @property
    @u.quantity_input
    def temperature(self,) -> u.K:
        x = (self.s/self.loop_length).decompose()
        lam = self._lambda
        beta_term = betaincinv(self._lambda+1, 0.5, x.value)**(1./(2 + self.gamma + self.alpha))
        return self.max_temperature * beta_term + self.base_temperature

    @property
    @u.quantity_input
    def density(self,) -> u.cm**(-3):
        return self.pressure / 2. / const.k_B / self.temperature

    @property
    @u.quantity_input
    def pressure(self,) -> u.dyne/(u.cm**2):
        kappa_0 = 1e-6*u.erg/u.cm/u.s/(u.K**(7/2))
        chi = (10**(-18.8))*u.erg*(u.cm**3)/u.s*(u.K**(self.gamma))
        chi_0 = chi/(4.*(const.k_B**2))
        coeff = np.sqrt(kappa_0 / chi_0 * (3 - 2*self.gamma))/(4 + 2*self.gamma + 2*self.alpha)
        beta_func = beta(self._lambda + 1, 0.5)
        p_0 = self.max_temperature**((11+2*self.gamma)/4) * coeff * beta_func / self.loop_length
        return np.ones(self.s.shape) * p_0

    @property
    def _lambda(self):
        mu = -2*(2+self.gamma)/7
        nu = 2*self.alpha/7
        return (1.-2*nu + mu)/(2*(nu-mu))


class RTVScalingLaws(object):

    def __init__(self,):
        raise NotImplementedError('RTV scaling laws not yet implemented')
