"""
Implementations of various coronal loop scaling laws
"""
import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy.special import beta, betaincinv

__all__ = ['MartensScalingLaws', 'RTVScalingLaws']

KAPPA_0 = 1e-6 * u.erg / u.cm / u.s * u.K**(-7/2)


class MartensScalingLaws(object):
    """
    Coronal loop scaling laws of [1]_

    Parameters
    ----------
    s : `~astropy.units.Quantity`
        Field-aligned loop coordinate for half of symmetric, semi-circular loop
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
    def __init__(self, s: u.cm, max_temperature: u.K, alpha=0, gamma=0.5,
                 base_temperature: u.K = 0.01*u.MK, chi=None):
        self.max_temperature = max_temperature
        self.base_temperature = base_temperature
        self.s = s
        self.alpha = alpha
        self.gamma = gamma
        self.chi = 10**(-18.8) * u.erg * u.cm**3 / u.s * u.K**(0.5) if chi is None else chi

    @property
    @u.quantity_input
    def loop_length(self,) -> u.cm:
        return np.diff(self.s).sum()

    @property
    def x(self,):
        return (self.s/self.loop_length).decompose()

    @property
    @u.quantity_input
    def temperature(self,) -> u.K:
        beta_term = betaincinv(self._lambda+1, 0.5, self.x.value)**(1./(2 + self.gamma + self.alpha))
        return self.max_temperature * beta_term + self.base_temperature

    @property
    @u.quantity_input
    def density(self,) -> u.cm**(-3):
        return self.pressure / 2. / const.k_B / self.temperature

    @property
    @u.quantity_input
    def pressure(self,) -> u.dyne/(u.cm**2):
        chi_0 = self.chi/(4.*(const.k_B**2))
        coeff = np.sqrt(KAPPA_0 / chi_0 * (3 - 2*self.gamma))/(4 + 2*self.gamma + 2*self.alpha)
        beta_func = beta(self._lambda + 1, 0.5)
        p_0 = self.max_temperature**((11+2*self.gamma)/4) * coeff * beta_func / self.loop_length
        return np.ones(self.s.shape) * p_0

    @property
    def _lambda(self):
        mu = -2*(2+self.gamma)/7
        nu = 2*self.alpha/7
        return (1.-2*nu + mu)/(2*(nu-mu))


class RTVScalingLaws(object):
    """
    Coronal loop scaling laws of [1]_

    Parameters
    ----------
    loop_length : `~astropy.units.Quantity`
        Loop half-length
    pressure : `~astropy.units.Quantity`
        Constant pressure
    heating_rate : `~astropy.units.Quantity`, optional
        Uniform heating rate
    chi : `~astropy.units.Quantity`, optional
        Coefficient for radiative loss function

    References
    ----------
    .. [1] Rosner, R., W.H. Tucker, G.S. Vaiana, 1978, ApJ, `220, 643 <http://adsabs.harvard.edu/abs/1978ApJ...220..643R>`_
    """

    @u.quantity_input
    def __init__(self, loop_length: u.cm, pressure=None, heating_rate=None, chi=None,):
        if (pressure is None and heating_rate is None) or (pressure is not None and heating_rate is not None):
            raise ValueError('Must specify either pressure or heating rate.')
        self.loop_length = loop_length
        self._pressure = pressure
        self._heating_rate = heating_rate
        self.chi = 10**(-18.8) * u.erg * u.cm**3 / u.s * u.K**(0.5) if chi is None else chi

    @property
    def c1(self,):
        return (3 / const.k_B * np.sqrt(self.chi / 2 / KAPPA_0))**(1/3)

    @property
    def c2(self,):
        return self.c1**(-5/2) * 7 / 8 * self.chi / (const.k_B**2)

    @property
    @u.quantity_input
    def max_temperature(self,) -> u.K:
        return self.c1 * (self.pressure * self.loop_length)**(1/3)

    @property
    @u.quantity_input
    def heating_rate(self,) -> u.erg / u.s / (u.cm**3):
        if self._heating_rate is not None:
            return self._heating_rate
        else:
            return self.c2 * self.pressure**(7/6) * self.loop_length**(-5/6)

    @property
    @u.quantity_input
    def pressure(self,) -> u.dyne/(u.cm**2):
        if self._pressure is not None:
            return self._pressure
        else:
            return (self.heating_rate / self.c2)**(6/7) * self.loop_length**(5/7)

    @property
    @u.quantity_input
    def density(self,) -> u.cm**(-3):
        return self.pressure / (2 * const.k_B * self.max_temperature)
