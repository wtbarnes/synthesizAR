"""
Implementations of various coronal loop scaling laws
"""
import numpy as np
import astropy.units as u
import astropy.constants as const
import sunpy.sun.constants as sun_const
from scipy.special import beta, betaincinv
from scipy.integrate import cumtrapz

__all__ = ['Isothermal', 'MartensScalingLaws', 'RTVScalingLaws']

KAPPA_0 = 1e-6 * u.erg / u.cm / u.s * u.K**(-7/2)


class Isothermal(object):
    r"""
    Hydrostatic loop solutions for an isothermal atmosphere

    Parameters
    ----------
    s : `~astropy.units.Quantity`
        Field-aligned loop coordinate
    r : `~astropy.units.Quantity`
        Radial distance as a function of `s`
    temperature : `~astropy.units.Quantity`
    pressure0 : `~astropy.units.Quantity`
        Pressure at :math:`r=R_{\odot}`
    """

    @u.quantity_input
    def __init__(self, s: u.cm, r: u.cm, temperature: u.K, pressure0: u.dyne/u.cm**2):
        self.s = s
        self.r = r
        self.temperature = temperature
        self.pressure0 = pressure0

    @property
    def _integral(self):
        # Add points to the front in the case that s[0] does not
        # correspond to R_sun as we do not know the initial value
        # at that point
        r = np.append(const.R_sun, self.r)
        s = np.append(-np.diff(self.s)[0], self.s)
        integrand = 1/r**2 * np.gradient(r) / np.gradient(s)
        # Integrate over the whole loop
        return cumtrapz(integrand.to('cm-2').value, s.to('cm').value) / u.cm

    @property
    @u.quantity_input
    def pressure(self) -> u.dyne / u.cm**2:
        return self.pressure0 * np.exp(-const.R_sun**2 / self.pressure_scale_height * self._integral)

    @property
    @u.quantity_input
    def pressure_scale_height(self) -> u.cm:
        return 2 * const.k_B * self.temperature / const.m_p / sun_const.equatorial_surface_gravity

    @property
    @u.quantity_input
    def density(self) -> u.cm**(-3):
        return self.pressure / (2*const.k_B*self.temperature)


class MartensScalingLaws(object):
    """
    Coronal loop scaling laws of [1]_

    Parameters
    ----------
    s : `~astropy.units.Quantity`
        Field-aligned loop coordinate for half of symmetric, semi-circular loop
    loop_length : `~astropy.units.Quantity`
        Loop half-length
    heating_constant : `astropy.units.Quantity`
        Constant of proportionality that relates the actual heating rate to the
        scaling with temperature and pressure. The actual units will depend on
        `alpha` and `beta`. See Eq. 2 of [1]_.
    alpha : `float`, optional
        Temperature dependence of the heating rate
    beta : `float`, optional
        Pressure depndence of the heating rate
    gamma : `float`, optional
        Temperature dependence of the radiative loss rate
    chi : `astropy.units.Quantity`, optional
        Constant of proportionality relating the actual radiative losses to the
        scaling with temperature. May need to adjust this based on the value of
        `gamma`.

    References
    ----------
    .. [1] Martens, P., 2010, ApJ, `714, 1290 <http://adsabs.harvard.edu/abs/2010ApJ...714.1290M>`_
    """

    @u.quantity_input
    def __init__(self, s: u.cm, loop_length: u.cm, heating_constant, alpha=0, beta=0, gamma=0.5,
                 chi=10**(-18.8) * u.erg * u.cm**3 / u.s * u.K**(0.5)):
        self.s = s
        self.loop_length = loop_length
        self.heating_constant = heating_constant
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.chi = chi
        self.chi_0 = self.chi/(4.*(const.k_B**2))

    @property
    def x(self,):
        x = (self.s/self.loop_length).decompose()
        if (x > 1).any():
            raise ValueError()
        return x

    @property
    @u.quantity_input
    def max_temperature(self,) -> u.K:
        coeff_1 = np.sqrt(KAPPA_0 / self.chi_0 * (3 - 2*self.gamma))/(
            4 + 2*self.gamma + 2*self.alpha)
        coeff_2 = (7/2 + self.alpha)/(3/2 - self.gamma)
        beta_func = beta(self._lambda + 1, 0.5)
        index = self.alpha + 11/4*self.beta + self.gamma*self.beta/2 - 7/2
        return (self.chi_0 * coeff_2 / self.heating_constant
                * (coeff_1*beta_func/self.loop_length)**(2-self.beta))**(1/index)

    @property
    @u.quantity_input
    def temperature(self,) -> u.K:
        beta_term = betaincinv(self._lambda+1, 0.5, self.x.value)**(1./(2 + self.gamma + self.alpha))
        return self.max_temperature * beta_term

    @property
    @u.quantity_input
    def pressure(self,) -> u.dyne/(u.cm**2):
        coeff = np.sqrt(KAPPA_0 / self.chi_0 * (3 - 2*self.gamma))/(4 + 2*self.gamma + 2*self.alpha)
        beta_func = beta(self._lambda + 1, 0.5)
        p_0 = self.max_temperature**((11+2*self.gamma)/4) * coeff * beta_func / self.loop_length
        return np.ones(self.s.shape) * p_0

    @property
    @u.quantity_input
    def heating_rate(self,) -> u.erg/(u.cm**3)/u.s:
        return self.heating_constant * (self.pressure**self.beta) * (self.temperature**self.alpha)

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
