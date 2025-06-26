"""
Utility functions for building instrument classes
"""
import astropy.units as u
import astropy.wcs
import copy
import ndcube
import numpy as np
import xarray

from ndcube.extra_coords.table_coord import QuantityTableCoordinate
from ndcube.wcs.wrappers import CompoundLowLevelWCS

__all__ = [
    'get_wave_keys',
    'add_wave_keys_to_header',
    'extend_celestial_wcs',
    'read_cube_from_dataset',
    'write_cube_to_netcdf',
]


@u.quantity_input
def get_wave_keys(wavelength_array: u.angstrom, index=3):
    return {
        f'CDELT{index}': (wavelength_array[1] - wavelength_array[0]).value,
        f'CTYPE{index}': 'WAVE',
        f'CUNIT{index}': wavelength_array.unit.to_string(),
        f'CRPIX{index}': 1,
        f'CRVAL{index}': wavelength_array[0].value,
        f'NAXIS{index}': wavelength_array.shape[0],
    }


@u.quantity_input
def add_wave_keys_to_header(wavelength_array: u.angstrom, header):
    header_copy = copy.deepcopy(header)
    index = header_copy.get('WCSAXES', 0) + 1
    wave_keys = get_wave_keys(wavelength_array, index=index)
    for wk in wave_keys:
        header_copy[wk] = wave_keys[wk]
    return header_copy


def extend_celestial_wcs(celestial_wcs, *extra_coords, **kwargs):
    """
    Add additional axes as extra coords to an existing WCS

    Parameters
    ----------
    celestial_wcs: `~astropy.wcs.WCS`
    extra_coords: `list`
        Each member of the list can be an extra coord or a tuple.
        If the latter, it is assumed this is a three-tuple specifying
        the array, name, and physical type of the `QuantityTableCoordinate`
    """
    wcses = []
    for ec in extra_coords:
        if isinstance(ec, tuple):
            array, name, physical_type = ec
            ec = QuantityTableCoordinate(array, names=name, physical_types=physical_type)
        wcses.append(ec.wcs)
    return CompoundLowLevelWCS(celestial_wcs, *wcses, **kwargs)


def read_cube_from_dataset(filename, axis_name, physical_type):
    """
    Read an `~ndcube.NDCube` from an `xarray` dataset.

    This function reads a data cube from a netCDF file and rebuilds it
    as an NDCube. The assumption is that the attributes on the stored
    data array have the keys necessary to reconstitute a celestial FITS
    WCS and that the axis denoted by `axis_name` is the additional axis
    along which to extend that celestial WCS. This works only for 3D cubes
    where two of the axes correspond to spatial, celestial axes.

    Parameters
    ----------
    filename: `str`, path-like
        File to read from, usually a netCDF file
    axis_name: `str`
        The addeded coordinate along which to extend the celestial WCS.
    physical_type: `str`
        The physical type of `axis_name` as denoted by the IVOA designation.
    """
    ds = xarray.load_dataset(filename)
    meta = ds.attrs
    data = u.Quantity(ds['data'].data, meta.pop('unit'))
    mask = ds['mask'].data
    celestial_wcs = astropy.wcs.WCS(header=meta)
    axis_array = u.Quantity(ds[axis_name].data, ds[axis_name].attrs.get('unit'))
    combined_wcs = extend_celestial_wcs(celestial_wcs, (axis_array, axis_name, physical_type))
    return ndcube.NDCube(data, wcs=combined_wcs, meta=meta, mask=mask)


def write_cube_to_netcdf(filename, axis_name, cube):
    """
    Write a `~ndcube.NDCube` to a netCDF file.

    This function writes an NDCube to a netCDF file by first expressing
    it as an `xarray.DataArray`. This works only for 3D cubes where two of
    the axes correspond to spatial, celestial axes.

    Parameters
    ----------
    cube: `ndcube.NDCube`
    axis_name: `str`
    filename: `str` or path-like
    """
    # FIXME: This is not a general solution and is probably really brittle
    celestial_wcs = cube.wcs.low_level_wcs._wcs[0]
    wcs_keys = dict(celestial_wcs.to_header())
    axis_array = cube.axis_world_coords(axis_name)[0]
    axis_coord = xarray.Variable(axis_name,
                                 axis_array.value,
                                 attrs={'unit': axis_array.unit.to_string()})
    if (mask:=cube.mask) is None:
        mask = np.full(cube.data.shape, False)
    cube_xa = xarray.Dataset(
        {'data': ([axis_name, 'lat', 'lon'], cube.data),
         'mask': ([axis_name, 'lat', 'lon'], mask)},
        coords={
            axis_name: axis_coord,
        },
        attrs={**wcs_keys, 'unit': cube.unit.to_string()}
    )
    cube_xa.to_netcdf(filename)
