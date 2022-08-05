"""
Instruments for calculating physical quantities, rather than
observed counts, projected along a LOS
"""
from dataclasses import dataclass

import astropy.units as u
import astropy.wcs
import numpy as np
from scipy.interpolate import interp1d
import ndcube
from ndcube.extra_coords.table_coord import QuantityTableCoordinate, MultipleTableCoordinate
from ndcube.wcs.wrappers import CompoundLowLevelWCS

from synthesizAR.instruments import ChannelBase, InstrumentBase
from synthesizAR.util import los_velocity
from synthesizAR.instruments.util import add_wave_keys_to_header

__all__ = [
    'InstrumentDEM',
    'InstrumentLOSVelocity',
    'InstrumentTemperature'
]


@dataclass
class ChannelDEM(ChannelBase):
    bin_edges: u.Quantity

    def __post_init__(self):
        self.log_bin_edges = np.log10(self.bin_edges.to('K').value)
        self.name = f'{self.log_bin_edges[0]:.2f}-{self.log_bin_edges[1]:.2f}'


class InstrumentDEM(InstrumentBase):
    name = 'DEM'

    @u.quantity_input
    def __init__(self, *args, temperature_bin_edges: u.K, **kwargs):
        self.temperature_bin_edges = temperature_bin_edges
        n_bins = temperature_bin_edges.shape[0]-1
        bin_edges = [temperature_bin_edges[[i, i+1]] for i in range(n_bins)]
        self.channels = [ChannelDEM(0*u.angstrom, None, be) for be in bin_edges]
        super().__init__(*args, **kwargs)

    @property
    @u.quantity_input
    def temperature_bin_centers(self) -> u.K:
        return (self.temperature_bin_edges[1:] + self.temperature_bin_edges[:-1])/2

    @staticmethod
    def calculate_intensity_kernel(loop, channel, **kwargs):
        T = loop.electron_temperature
        n = loop.density
        bin_mask = np.where(
            np.logical_and(T >= channel.bin_edges[0], T < channel.bin_edges[1]), 1, 0)
        kernel = n**2 * bin_mask
        return kernel

    def dem_maps_to_cube(self, dem, time_index):
        """
        Convert a list of DEM maps to a DEM NDCube
        """
        # NOTE: this is the format that .observe returns
        dem_list = [dem[c.name][time_index] for c in self.channels]
        # Construct WCS
        celestial_wcs = dem_list[0].wcs
        temp_table = QuantityTableCoordinate(self.temperature_bin_centers,
                                             names='temperature',
                                             physical_types='phys.temperature')
        temp_table_coord = MultipleTableCoordinate(temp_table)
        mapping = list(range(celestial_wcs.pixel_n_dim))
        mapping.extend([celestial_wcs.pixel_n_dim] * temp_table_coord.wcs.pixel_n_dim)
        compound_wcs = CompoundLowLevelWCS(celestial_wcs, temp_table_coord.wcs, mapping=mapping)
        # Stack arrays
        dem_array = u.Quantity([d.quantity for d in dem_list])

        return ndcube.NDCube(dem_array, wcs=compound_wcs, )

    @staticmethod
    def calculate_intensity(dem, spectra, header,
                            meta=None,
                            wavelength_instr=None,
                            response_instr=None):
        """
        Compute intensity from a DEM and a temperature-dependent spectra

        Parameters
        ----------
        dem: `~ndcube.NDCube`
            The first axis should correspond to temperature
        spectra: `~ndcube.NDCube`
        header: `dict` or header-like
            Header information corresponding to the spatial axes of the DEM cube
        meta: `dict`, optional
            Additional metadata
        wavelength
        """
        temperature_bin_centers = dem.axis_world_coords(0)[0]
        wavelength_spectra = spectra.axis_world_coords(1)[0]
        temperature_spectra = spectra.axis_world_coords(0)[0].to(temperature_bin_centers.unit)
        # Interpolate spectral cube to DEM temperatures
        spectra_interp = interp1d(temperature_spectra.value, spectra.data, axis=0)(
                                  temperature_bin_centers.value)
        # If a wavelength response and wavelength array are passed in, then interpolate the
        # spectra to that wavelength
        if response_instr and wavelength_instr:
            spectra_interp = interp1d(wavelength_spectra.value, spectra_interp, axis=1)(
                                      wavelength_instr.to_value(wavelength_spectra.unit))
            wave_header = add_wave_keys_to_header(wavelength_instr, header)
        else:
            response_instr = np.ones(wavelength_spectra.shape)
            wave_header = add_wave_keys_to_header(wavelength_spectra, header)
        spectra_interp = spectra_interp * spectra.unit * response_instr
        # Take dot product between DEM and spectra
        intensity = np.tensordot(spectra_interp, u.Quantity(dem.data, dem.unit), axes=([0], [0]))
        # Construct cube
        wave_header['BUNIT'] = intensity.unit.to_string()
        wave_header['NAXIS'] = len(intensity.shape)
        wave_header['WCSAXES'] = len(intensity.shape)
        meta = {} if meta is None else meta
        meta = {**meta, **wave_header}
        wcs = astropy.wcs.WCS(header=wave_header)
        return ndcube.NDCube(intensity, wcs=wcs, meta=meta)


class InstrumentQuantityBase(InstrumentBase):

    @u.quantity_input
    def __init__(self, *args, **kwargs):
        self.channels = [ChannelBase(1, 0*u.angstrom, self.name)]
        super().__init__(*args, average_over_los=True, **kwargs)


class InstrumentLOSVelocity(InstrumentQuantityBase):
    name = 'los_velocity'

    @staticmethod
    def calculate_intensity_kernel(loop, *args, **kwargs):
        observer = kwargs.get('observer')
        if observer is None:
            raise ValueError('Must pass in observer to compute LOS velocity.')
        return los_velocity(loop.velocity_xyz, observer)


class InstrumentTemperature(InstrumentQuantityBase):
    name = 'temperature'

    @staticmethod
    def calculate_intensity_kernel(loop, *args, **kwargs):
        return loop.electron_temperature
