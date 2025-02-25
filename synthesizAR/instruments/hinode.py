"""
Class for Hinode/EIS instrument. Holds information about spectral, temporal, and spatial resolution
and other instrument-specific information.
"""
import astropy.units as u
import numpy as np
import xrtpy

from dataclasses import dataclass

from synthesizAR.instruments import ChannelBase, InstrumentBase
from synthesizAR.util.decorators import return_quantity_as_tuple

__all__ = ['InstrumentHinodeXRT']


@dataclass
class ChannelXRT(ChannelBase):
    temperature: u.Quantity = None
    response: u.Quantity = None
    filter_wheel_1: str = None
    filter_wheel_2: str = None
    psf_width: u.Quantity = (1, 1)*u.pixel

    def __post_init__(self):
        self.name = f'{self.filter_wheel_1}_{self.filter_wheel_2}'


class InstrumentHinodeXRT(InstrumentBase):
    name = 'Hinode_XRT'

    def __init__(self, observing_time, observer, filters, **kwargs):
        resolution = kwargs.pop('resolution', [2.05719995499, 2.05719995499] * u.arcsec/u.pixel)
        cadence = kwargs.pop('cadence', 20 * u.s)
        self._filters = filters
        super().__init__(
            observing_time=observing_time,
            observer=observer,
            resolution=resolution,
            cadence=cadence,
            **kwargs
        )

    @property
    def channels(self):
        channels = []
        for f in self._filters:
            trf = xrtpy.response.TemperatureResponseFundamental(f, self.observer.obstime)
            # Assign filter wheel
            if f in xrtpy.response.effective_area.index_mapping_to_fw1_name:
                filter_wheel_1 = f.replace("-", "_")
                filter_wheel_2 = 'Open'
            elif f in xrtpy.response.effective_area.index_mapping_to_fw2_name:
                filter_wheel_1 = 'Open'
                filter_wheel_2 = f.replace("-", "_")
            else:
                raise ValueError(f'{f} is not a valid XRT filter wheel choice.')
            c = ChannelXRT(
                temperature=trf.CHIANTI_temperature,
                response=trf.temperature_response(),
                filter_wheel_1=filter_wheel_1,
                filter_wheel_2=filter_wheel_2,
            )
            channels.append(c)
        return channels

    @property
    def observatory(self):
        return 'Hinode'

    @property
    def detector(self):
        return 'XRT'

    @property
    def telescope(self):
        return 'Hinode'

    @property
    def _expected_unit(self):
        return u.DN / (u.pix * u.s)

    def get_instrument_name(self, channel):
        return self.detector

    def get_header(self, *args, **kwargs):
        header = super().get_header(*args, **kwargs)
        if (channel := kwargs.get('channel')):
            header['EC_FW1_'] = channel.filter_wheel_1
            header['EC_FW2_'] = channel.filter_wheel_2
        return header

    @staticmethod
    @return_quantity_as_tuple
    def calculate_intensity_kernel(loop, channel, **kwargs):
        K_T = np.interp(loop.electron_temperature,
                        channel.temperature,
                        channel.response)
        return K_T * loop.density**2
