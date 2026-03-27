"""
Classes for Hinode EIS and XRT
"""
import astropy.units as u
import numpy as np
import sunpy.map
import xrtpy

from astropy.stats import gaussian_fwhm_to_sigma
from dataclasses import dataclass
from scipy.interpolate import interpn

from synthesizAR.instruments import ChannelBase, InstrumentBase
from synthesizAR.util.decorators import return_quantity_as_tuple

__all__ = ['InstrumentHinodeEIS', 'InstrumentHinodeXRT']


@dataclass
class ChannelEIS:
    """
    Parameters
    ----------
    line_id: str
        EIS line identification.
    transition: str
        Label of the transition, including configuration and angular momentum
        of upper and lower energy level.
    """
    line_id: str
    transition: str
    # See https://solarb.mssl.ucl.ac.uk/SolarB/eis_docs/eis_notes/08_COMA/eis_swnote_08.pdf
    psf_width: u.Quantity = (3, 3)*u.pixel*gaussian_fwhm_to_sigma

    def __post_init__(self):
        el, ion, wave = self.line_id.split()
        self.ion_name = f'{el} {ion}'
        self.wavelength = float(wave) * u.AA
        self.name = self.line_id
        self.channel = self.wavelength


class InstrumentHinodeEIS(InstrumentBase):
    """
    Instrument object for computing intensities as observed by the EUV Imaging Spectrometer onboard _Hinode_.

    This instrument class produces an intensity map per timestep in physical units, i.e. :math:`erg cm^{-2} s^{-1} sr^{-1}`.
    It does not include a wavelength dimension. The modeled intensity should be interpreted as the peak intensity of the line profile.

    Parameters
    ----------
    observing_time: `~astropy.units.Quantity`
        Observing time range
    observer: `~astropy.coordinates.SkyCoord`
        Observer location
    lines: `list`
        List of tuples specifying the EIS lines to synthesize. Each entry should be a tuple of strings
        where the first entry is the line ID and the second is the transition, e.g.
        ``('S XIII 256.686', '2s 2p 1P1 -- 2s2 1S0')``. It is best if the line IDs are consistent with those
        used by the EIS software.
    """
    name = 'Hinode_EIS'

    def __init__(self, observing_time, observer, lines, **kwargs):
        self._lines = lines
        plate_scale = kwargs.pop('plate_scale', [2.99519992, 1.0]*u.arcsec/u.pixel)
        fov_width = kwargs.pop('fov_width', [87, 512]*u.pixel)
        cadence = kwargs.pop('cadence', 41.89655172413793*u.s)
        super().__init__(
            observing_time=observing_time,
            observer=observer,
            plate_scale=plate_scale,
            cadence=cadence,
            fov_width=fov_width,
            **kwargs,
        )

    @property
    def channels(self):
        return [ChannelEIS(*line) for line in self._lines]

    @property
    def observatory(self):
        return 'Hinode'

    @property
    def telescope(self):
        return 'Hinode'

    @property
    def detector(self):
        return 'EIS'

    def get_instrument_name(self, channel):
        # This is explicitly not just EIS so that the eispac EIS map source
        # is not used for these time-dependent files
        return f'{self.telescope}_{self.detector}'

    @property
    def _expected_unit(self):
        return u.Unit('erg cm-2 s-1 sr-1')

    def get_header(self, *args, **kwargs):
        header = super().get_header(*args, **kwargs)
        if (channel:=kwargs.get('channel')) is not None:
            header['line_id'] = channel.line_id
            header['measrmnt'] = 'intensity'
            header['lvl_num'] = 2
        return header

    @staticmethod
    @return_quantity_as_tuple
    def calculate_intensity_kernel(loop, channel, **kwargs):
        if (em_model:=kwargs.get('emission_model')) is None:
            raise ValueError('Must pass an EmissionModel instance to .observe()')
        # Get line emissivity from model
        ion = em_model[channel.ion_name]
        wavelength, em_ion = em_model.get_line_emissivity(ion, transition=channel.transition)
        # Reshape temperature and density arrays
        n = loop.density
        T = loop.electron_temperature
        Tn_flat = np.stack((T.value.flatten(), n.value.flatten()), axis=1)
        # Interpolate emissivity to loop n, T
        em_flat = interpn(
            (em_model.temperature.to_value(T.unit), em_model.density.to_value(n.unit)),
            em_ion.squeeze().value,
            Tn_flat,
            method='linear',
            fill_value=0.0,  # No emission outside of model bounds
            bounds_error=False,
        )
        em_ion_interp = np.reshape(em_flat, T.shape)
        em_ion_interp = u.Quantity(np.where(em_ion_interp<0, 0, em_ion_interp), em_ion.unit)
        # Apply conversion factors
        ionization_fraction = loop.get_ionization_fraction(ion)
        erg_per_ph = wavelength.to('erg', equivalencies=u.equivalencies.spectral()) / u.photon
        kernel = erg_per_ph*ionization_fraction*em_ion_interp*n**2/(4*np.pi*u.steradian)
        return kernel

    @staticmethod
    def build_raster_map(maps):
        """
        Build a single map from a series of per-exposure maps.

        Parameters
        ----------
        maps: `list` or `~sunpy.map.MapSequence`
        """
        # TODO: make this more flexible to allow for producing multiple rasters
        # in the case where the number of maps exceeds the x-dimension of the map
        maps = sunpy.map.Map(maps, sequence=True)
        if not maps.all_same_shape:
            raise ValueError('All maps must be the same shape')
        shape = maps[0].data.shape
        idx_last = min(shape[1]-1, len(maps)-1)
        # Update metadata
        new_meta = maps[0].meta.copy()
        new_meta['date-beg'] = maps[0].date.isot
        new_meta['date-end'] = maps[idx_last].date.isot
        new_meta['date-avg'] = (maps[0].date + (maps[idx_last].date - maps[0].date)/2).isot
        new_meta['date-obs'] = maps[0].date_average.isot
        # This is reset here to ensure that the EISPAC map source is used.
        new_meta['instrume'] = 'EIS'
        # Build raster map
        new_data = np.nan*np.ones(shape)
        for i in range(idx_last+1):
            new_data[:,shape[1]-1-i] = maps[i].data[:,shape[1]-1-i]
        return sunpy.map.Map(new_data, new_meta)


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
    """
    Instrument object for computing images as observed by Hinode XRT.

    Parameters
    ----------
    observing_time: `~astropy.units.Quantity`
        Observing time range
    observer: `~astropy.coordinates.SkyCoord`
        Observer location
    filters: `list`
        List of filter label strings. It is assumed that the other filter is open.
    """
    name = 'Hinode_XRT'

    def __init__(self, observing_time, observer, filters, **kwargs):
        plate_scale = kwargs.pop('plate_scale', [2.05719995499, 2.05719995499] * u.arcsec/u.pixel)
        cadence = kwargs.pop('cadence', 20 * u.s)
        self._filters = filters
        super().__init__(
            observing_time=observing_time,
            observer=observer,
            plate_scale=plate_scale,
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
