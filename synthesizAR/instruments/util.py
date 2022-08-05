"""
Utility functions for building instrument classes
"""
import copy

import astropy.units as u

__all__ = ['add_wave_keys_to_header']


@u.quantity_input
def add_wave_keys_to_header(wavelength_array: u.angstrom, header):
    header_copy = copy.deepcopy(header)
    wave_keys = {
        'CDELT3': (wavelength_array[1] - wavelength_array[0]).value,
        'CTYPE3': 'WAVE',
        'CUNIT3': wavelength_array.unit.to_string(),
        'CRPIX3': 1,
        'CRVAL3': wavelength_array[0].value,
        'NAXIS3': wavelength_array.shape[0],
    }
    for wk in wave_keys:
        header_copy[wk] = wave_keys[wk]
    return header_copy
