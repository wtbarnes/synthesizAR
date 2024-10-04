"""
A collection of commonly used heating models
"""
import astropy.units as u
import numpy as np

from scipy.interpolate import interp1d

__all__ = ['b_over_l_scaling']


@u.quantity_input
def b_over_l_scaling(loop, H_0=0.0738*u.Unit('erg cm-3 s-1'), alpha=0.3, beta=0.2, B_0=76*u.G,
                     L_0=29*u.Mm) -> u.Unit('erg cm-3 s-1'):
    """
    Heating rate dependent on the loop length and average field strength along the loop.

    .. note:: The default values for all parameters are taken from
              :cite:t:`ugarte-urra_magnetic_2019`.

    Parameters
    ----------
    loop: `synthesizAR.Loop`
    H_0: `~astropy.units.Quantity`, optional
        Nominal heating rate.
    alpha: `float`, optional
        Dependence on average field strength.
    beta: `float`, optional
        Dependence on loop length.
    B0: `~astropy.units.Quantity`, optional
        Nominal value of the field strength.
    L0: `~astropy.units.Quantity`, optional
        Nominal value of the loop length.
    """
    f_interp = interp1d(loop.field_aligned_coordinate.value, loop.field_strength.value,)
    B_center = f_interp(loop.field_aligned_coordinate_center.value)
    B_avg = np.average(B_center, weights=np.diff(loop.field_aligned_coordinate))
    B_avg = B_avg * loop.field_strength.unit
    return H_0 * ((B_avg / B_0)**alpha) * ((L_0 / loop.length)**beta)
