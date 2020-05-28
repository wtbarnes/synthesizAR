"""
Various helpers for configuring/loading HYDRAD simulations
"""
import numpy as np
import astropy.units as u
import astropy.constants as const
import sunpy.sun.constants as sun_const

__all__ = ['cross_section_coefficients', 'gravity_coefficients']


def cross_section_coefficients(loop):
    # NOTE: this is reversed because numpy returns the coefficients
    # in descending polynomial order, but HYDRAD expects them in
    # ascending order.
    return np.polyfit(loop.field_aligned_coordinate_norm.value,
                      loop.field_strength.to(u.G).value, 6)[::-1]


def gravity_coefficients(loop):
    r_hat = u.Quantity(np.stack([
        np.sin(loop.coordinate.spherical.lat)*np.cos(loop.coordinate.spherical.lon),
        np.sin(loop.coordinate.spherical.lat)*np.sin(loop.coordinate.spherical.lon),
        np.cos(loop.coordinate.spherical.lat)
    ]))
    r_hat_dot_s_hat = (r_hat * loop.coordinate_direction).sum(axis=0)
    g_parallel = -sun_const.surface_gravity * (
        (const.R_sun / loop.coordinate.spherical.distance)**2) * r_hat_dot_s_hat
    # NOTE: this is reversed because numpy returns the coefficients
    # in descending polynomial order, but HYDRAD expects them in
    # ascending order.
    return np.polyfit(loop.field_aligned_coordinate_norm.value,
                      g_parallel.to(u.cm/(u.s**2)).value, 6)[::-1]
