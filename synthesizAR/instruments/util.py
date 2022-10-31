"""
Utility functions for building instrument classes
"""
import copy

import astropy.units as u
from ndcube.extra_coords.table_coord import QuantityTableCoordinate
from ndcube.wcs.wrappers import CompoundLowLevelWCS

__all__ = ['get_wave_keys', 'add_wave_keys_to_header', 'extend_celestial_wcs']


@u.quantity_input
def get_wave_keys(wavelength_array: u.angstrom):
    return {
        'CDELT3': (wavelength_array[1] - wavelength_array[0]).value,
        'CTYPE3': 'WAVE',
        'CUNIT3': wavelength_array.unit.to_string(),
        'CRPIX3': 1,
        'CRVAL3': wavelength_array[0].value,
        'NAXIS3': wavelength_array.shape[0],
    }


@u.quantity_input
def add_wave_keys_to_header(wavelength_array: u.angstrom, header):
    header_copy = copy.deepcopy(header)
    wave_keys = get_wave_keys(wavelength_array)
    for wk in wave_keys:
        header_copy[wk] = wave_keys[wk]
    return header_copy


def extend_celestial_wcs(celestial_wcs, array, name, physical_type):
    """
    Add an additional 3rd axis corresponding to a `~astropy.units.Quantity`
    to a 2D celestial WCS
    """
    temp_table = QuantityTableCoordinate(array,
                                         names=name,
                                         physical_types=physical_type)
    mapping = list(range(celestial_wcs.pixel_n_dim))
    mapping.extend(
        [celestial_wcs.pixel_n_dim] * temp_table.wcs.pixel_n_dim
    )
    return CompoundLowLevelWCS(celestial_wcs, temp_table.wcs, mapping=mapping)
