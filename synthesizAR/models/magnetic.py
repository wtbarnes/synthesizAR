"""
Analytic magnetic field profiles
"""
import astropy.units as u
import numpy as np

__all__ = ['mikic_magnetic_field_profile']


@u.quantity_input
def mikic_magnetic_field_profile(s: u.cm,
                                 length: u.Mm,
                                 ell: u.Mm=14*u.Mm,
                                 B_0: u.G=1*u.G,
                                 B_1: u.G=10*u.G):
    r"""
    Magnetic field strength along a strand as prescribed by Eq. 4 of :cite:t:`mikic_importance_2013`

    The analytic magnetic field profile along the strand as a function of the field-aligned coordinate
    :math:`s` is given by,

    .. math::

        B(s) = B_0 + B_1(\exp{-s/\ell} + \exp{-(L-s)/\ell})

    Parameters
    ----------
    s: `~astropy.units.Quantity`
        Field-aligned coordinate
    length: `~astropy.units.Quantity`
        Strand length
    ell: `~astropy.units.Quantity`, optional
        Scale height of the magnetic field
    B_0: `~astropy.units.Quantity`, optional
    B_1: `~astropy.units.Quantity`, optional
    """
    return B_0 + B_1*(np.exp(-s/ell) + np.exp(-(length-s)/ell))
