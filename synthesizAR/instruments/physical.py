"""
Instruments for calculating physical quantities, rather than
observed counts, projected along a LOS
"""
from dataclasses import dataclass
from multiprocessing import Value

import astropy.units as u
import numpy as np

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
        bin_edges = [temperature_bin_edges[[i,i+1]] for i in range(temperature_bin_edges.shape[0]-1)]
        self.channels = [ChannelDEM(1, 0*u.angstrom, None, be) for be in bin_edges]
        super().__init__(*args, **kwargs)

    @staticmethod
    def calculate_intensity_kernel(loop, channel, **kwargs):
        T = loop.electron_temperature
        n = loop.density
        bin_mask = np.where(np.logical_and(T>=channel.bin_edges[0], T<channel.bin_edges[1]), 1, 0)
        kernel = n**2 * bin_mask
        return kernel


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
        return los_velocity(loop.velocity_x, loop.velocity_y, loop.velocity_z, observer)


class InstrumentTemperature(InstrumentQuantityBase):
    name = 'temperature'

    @staticmethod
    def calculate_intensity_kernel(loop, *args, **kwargs):
        return loop.electron_temperature