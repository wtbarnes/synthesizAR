"""
Instruments for calculating physical quantities, rather than
observed counts, projected along a LOS
"""
from dataclasses import dataclass

import astropy.units as u
import numpy as np
import ndcube
from ndcube.extra_coords.table_coord import QuantityTableCoordinate, MultipleTableCoordinate
from ndcube.wcs.wrappers import CompoundLowLevelWCS

from synthesizAR.instruments import ChannelBase, InstrumentBase
from synthesizAR.util import los_velocity

__all__ = ['InstrumentDEM', 'InstrumentLOSVelocity', 'InstrumentTemperature']


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
        bin_edges = [temperature_bin_edges[[i,i+1]] for i in range(temperature_bin_edges.shape[0]-1)]
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
