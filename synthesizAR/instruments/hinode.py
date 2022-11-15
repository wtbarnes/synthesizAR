"""
Class for Hinode/EIS instrument. Holds information about spectral, temporal, and spatial resolution
and other instrument-specific information.
"""
from dataclasses import dataclass
import json
import pkg_resources

import numpy as np
from scipy.ndimage import gaussian_filter
import astropy.units as u
import astropy.constants as const
import xrtpy

from synthesizAR.util import SpatialPair
from synthesizAR.instruments import InstrumentBase, ChannelBase

__all__ = ['InstrumentHinodeEIS', 'InstrumentHinodeXRT']


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
        super().__init__(observing_time, observer, **kwargs)
        self.channels = self._setup_channels(filters)

    def _setup_channels(self, filters):
        channels = []
        for f in filters:
            trf = xrtpy.response.TemperatureResponseFundamental(f, self.observer.obstime)
            # Assign filter wheel
            if f in xrtpy.response.effective_area.index_mapping_to_fw1_name:
                filter_wheel_1 = f.replace("-", "_")
                filter_wheel_2 = 'Open'
            elif f in xrtpy.response.effective_area.index_mapping_to_fw2_name:
                filter_wheel_1 = 'Open'
                filter_wheel_2 = f.replace("-", "_")
            c = ChannelXRT(
                temperature=trf.CHIANTI_temperature,
                # NOTE: switching from DN to counts here because DN is not
                # supported by the FITS standard
                response=trf.temperature_response()*u.ct/u.DN,
                filter_wheel_1=filter_wheel_1,
                filter_wheel_2=filter_wheel_2,
            )
            channels.append(c)
        return channels

    @property
    def cadence(self) -> u.s:
        return 20 * u.s

    @property
    def resolution(self) -> u.arcsec / u.pix:
        return [2.05719995499, 2.05719995499] * u.arcsec/u.pixel

    @property
    def observatory(self):
        return 'Hinode'

    @property
    def detector(self):
        return 'XRT'

    @property
    def telescope(self):
        return 'Hinode'

    def get_instrument_name(self, channel):
        return self.detector

    def get_header(self, channel, *args):
        header = super().get_header(channel, *args)
        header['EC_FW1_'] = channel.filter_wheel_1
        header['EC_FW2_'] = channel.filter_wheel_2
        return header

    @staticmethod
    def calculate_intensity_kernel(loop, channel, **kwargs):
        K_T = np.interp(loop.electron_temperature,
                        channel.temperature,
                        channel.response)
        return K_T * loop.density**2


class InstrumentHinodeEIS(InstrumentBase):
    """
    Class for Extreme-ultraviolet Imaging Spectrometer (EIS) instrument on the Hinode spacecraft.
    Converts emissivity calculations for each loop into detector units based on the spectral,
    spatial, and temporal resolution along with the instrument response functions.
    """

    def __init__(self, observing_time, observer_coordinate, window=None, apply_psf=True):
        self.name = 'Hinode_EIS'
        self.cadence = 10.0*u.s
        self.resolution = SpatialPair(x=1.0*u.arcsec/u.pixel, y=2.0*u.arcsec/u.pixel, z=None)
        self.fits_template['telescop'] = 'Hinode'
        self.fits_template['instrume'] = 'EIS'
        self.fits_template['detector'] = 'EIS'
        self.fits_template['waveunit'] = 'angstrom'
        self.apply_psf = apply_psf
        self.window = 0.5*u.angstrom if window is None else window
        super().__init__(observing_time, observer_coordinate=observer_coordinate)
        self._setup_channels()

    def _setup_channels(self):
        """
        Read instrument properties from files. This is a temporary solution and requires that the
        detector files all be collected into the same directory and be formatted in a specific way.

        .. warning:: This method will be modified once EIS response functions become
                    available in a different format.
        """
        hinode_fn = pkg_resources.resource_filename('synthesizAR',
                                                    'instruments/data/hinode_eis.json')
        with open(hinode_fn, 'r') as f:
            eis_info = json.load(f)

        self.channels = []
        for key in eis_info:
            if key != 'name' and key != 'description':
                self.channels.append({
                    'wavelength': eis_info[key]['wavelength']*u.Unit(eis_info[key]['wavelength_units']),
                    'name': key,
                    'response': {
                        'x': eis_info[key]['response_x']*u.Unit(eis_info[key]['response_x_units']),
                        'y': eis_info[key]['response_y']*u.Unit(eis_info[key]['response_y_units'])},
                    'spectral_resolution': eis_info[key]['spectral_resolution']*u.Unit(eis_info[key]['spectral_resolution_units']),
                    'gaussian_width': {'x': (3.*u.arcsec)/self.resolution.x,
                                       'y': (3.*u.arcsec)/self.resolution.y},
                    'instrument_width': eis_info[key]['instrument_width']*u.Unit(eis_info[key]['instrument_width_units']),
                    'wavelength_range': [eis_info[key]['response_x'][0],
                                         eis_info[key]['response_x'][-1]]*u.Unit(eis_info[key]['response_x_units'])
                })

        self.channels = sorted(self.channels, key=lambda x: x['wavelength'])

    def make_fits_header(self, field, channel):
        """
        Extend base method to include extra wavelength dimension.
        """
        header = super().make_fits_header(field, channel)
        header['wavelnth'] = channel['wavelength'].value
        header['naxis3'] = len(channel['response']['x'])
        header['ctype3'] = 'wavelength'
        header['cunit3'] = 'angstrom'
        header['cdelt3'] = np.fabs(np.diff(channel['response']['x']).value[0])
        return header

    def build_detector_file(self, file_template, chunks, field):
        """
        Build HDF5 files to store detector counts
        """
        additional_fields = ['{}'.format(line.value) for line in field.loops[0].resolved_wavelengths]
        super().build_detector_file(file_template, chunks, additional_fields=additional_fields)

    def flatten(self, loop, interp_s, hf, start_index):
        """
        Flatten loop emission to HDF5 file for given number of wavelengths
        """
        for wavelength in loop.resolved_wavelengths:
            emiss, ion_name = loop.get_emission(wavelength, return_ion_name=True)
            dset = hf['{}'.format(str(wavelength.value))]
            dset.attrs['ion_name'] = ion_name
            self.interpolate_and_store(emiss, loop, interp_s, dset, start_index)

    @staticmethod
    def compute_emission():
        """
        Compute intensity for transitions observed by EIS
        """
        pass

    def flatten_parallel(self, loops, interpolated_loop_coordinates, save_path, emission_model):
        """
        Build task graph for computing EIS spectra
        """
        pass

    def detect(self, hf, channel, i_time, header, temperature, los_velocity):
        """
        Calculate response of Hinode/EIS detector for given loop object.
        """
        import plasmapy
        # trim the instrument response to the appropriate wavelengths
        trimmed_indices = []
        for w in channel['model_wavelengths']:
            indices = np.where(np.logical_and(channel['response']['x'] >= w-self.window,
                                              channel['response']['x'] <= w+self.window))
            trimmed_indices += indices[0].tolist()
        trimmed_indices = list(sorted(set(trimmed_indices+[0, len(channel['response']['x'])-1])))
        response_x = channel['response']['x'][trimmed_indices]
        response_y = channel['response']['y'][trimmed_indices]

        # compute the response
        counts = np.zeros(temperature.shape+response_x.shape)
        for wavelength in channel['model_wavelengths']:
            # thermal width + instrument width
            ion_name = hf['{}'.format(str(wavelength.value))].attrs['ion_name']
            ion_mass = plasmapy.atomic.ion_mass(ion_name.split(' ')[0].capitalize()).cgs
            thermal_velocity = 2.*const.k_B.cgs*temperature/ion_mass
            thermal_velocity = np.expand_dims(thermal_velocity, axis=2)*thermal_velocity.unit
            line_width = ((wavelength**2)/(2.*const.c.cgs**2)*thermal_velocity
                          + (channel['instrument_width']/(2.*np.sqrt(2.*np.log(2.))))**2)
            # doppler shift due to LOS velocity
            doppler_shift = wavelength*los_velocity/const.c.cgs
            doppler_shift = np.expand_dims(doppler_shift, axis=2)*doppler_shift.unit
            # combine emissivity with instrument response function
            dset = hf['{}'.format(str(wavelength.value))]
            hist, edges = np.histogramdd(self.total_coordinates.value,
                                         bins=[self.bins.x, self.bins.y, self.bins.z],
                                         range=[self.bin_range.x, self.bin_range.y, self.bin_range.z],
                                         weights=np.array(dset[i_time, :]))
            emiss = np.dot(hist, np.diff(edges[2])).T
            emiss = (np.expand_dims(emiss, axis=2)
                     * u.Unit(get_keys(dset.attrs, ('unit', 'units')))*self.total_coordinates.unit)
            intensity = emiss*response_y/np.sqrt(2.*np.pi*line_width)
            intensity *= np.exp(-((response_x - wavelength - doppler_shift)**2)/(2.*line_width))
            if not hasattr(counts, 'unit'):
                counts = counts*intensity.unit
            counts += intensity

        header['bunit'] = counts.unit.to_string()
        if self.apply_psf:
            counts = (gaussian_filter(counts.value, (channel['gaussian_width']['y'].value,
                                                     channel['gaussian_width']['x'].value, 0))
                      * counts.unit)

        from synthesizAR.analysis import EISCube
        return EISCube(data=counts, header=header, wavelength=response_x)
