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

__all__ = ['InstrumentDEM']


@dataclass
class Channel(ChannelBase):
    bin_edges: u.Quantity

    def __post_init__(self):
        self.log_bin_edges = np.log10(self.bin_edges.to('K').value)
        self.name = f'{self.log_bin_edges[0]:.2f}-{self.log_bin_edges[1]:.2f}'


class InstrumentDEM(InstrumentBase):

    @u.quantity_input
    def __init__(self, observing_time: u.s, observer, temperature_bin_edges: u.K, cadence: u.s, resolution: u.arcsec/u.pix, **kwargs):
        self.telescope = 'DEM'
        self.detector = 'DEM'
        self.name = 'DEM'
        bin_edges = [temperature_bin_edges[[i,i+1]] for i in range(temperature_bin_edges.shape[0]-1)]
        self.channels = [Channel(1, 0*u.angstrom, None, be) for be in bin_edges]
        self.cadence = cadence
        self.resolution = resolution
        super().__init__(observing_time, observer, **kwargs)

    @staticmethod
    def calculate_intensity_kernel(loop, channel, **kwargs):
        T = loop.electron_temperature
        n = loop.density
        bin_mask = np.where(np.logical_and(T>=channel.bin_edges[0], T<channel.bin_edges[1]), 1, 0)
        kernel = n**2 * bin_mask
        return kernel


class InstrumentQuantityBase(InstrumentBase):

    @u.quantity_input
    def __init__(self, name, observing_time: u.s, observer, cadence: u.s, resolution: u.arcsec/u.pix, **kwargs):
        self.telescope = name
        self.detector = name
        self.name = name
        self.channels = [ChannelBase(1, 0*u.angstrom, name)]
        self.cadence = cadence
        self.resolution = resolution
        super().__init__(observing_time, observer, average_over_los=True, **kwargs)


class InstrumentLOSVelocity(InstrumentQuantityBase):

    @u.quantity_input
    def __init__(self, *args, **kwargs):
        super().__init__('los_velocity', *args, **kwargs)

    @staticmethod
    def calculate_intensity_kernel(loop, channel, **kwargs):
        observer = kwargs.get('observer')
        if observer is None:
            raise ValueError('Must pass in observer to compute LOS velocity.')
        return los_velocity(loop.velocity_x, loop.velocity_y, loop.velocity_z, observer)


class InstrumentTemperature(InstrumentQuantityBase):

    @u.quantity_input
    def __init__(self, *args, **kwargs):
        super().__init__('temperature', *args, **kwargs)

    @staticmethod
    def calculate_intensity_kernel(loop, channel, **kwargs):
        return loop.electron_temperature
