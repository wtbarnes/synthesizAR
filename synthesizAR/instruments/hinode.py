"""
Class for Hinode/EIS instrument. Holds information about spectral, temporal, and spatial resolution
and other instrument-specific information.
"""
from dataclasses import dataclass
import json
import pkg_resources

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from sunpy.map import Map
import astropy.units as u
import astropy.io.fits
import astropy.constants as const
import astropy.convolution
import h5py
import plasmapy
import dask

from synthesizAR.util import SpatialPair
from synthesizAR.instruments import InstrumentBase, ChannelBase
from synthesizAR.analysis import EISCube

__all__ = ['InstrumentHinodeEIS', 'InstrumentHinodeXRT']


@dataclass
class ChannelXRT(ChannelBase):
    temperature: u.Quantity
    response: u.Quantity
    filter_wheel_1: str
    filter_wheel_2: str

    def __post_init__(self):
        self.name = f'{self.filter_wheel_1}_{self.filter_wheel_2}'


class InstrumentHinodeXRT(InstrumentBase):
    name = 'Hinode_XRT'

    def __init__(self, observing_time, observer, **kwargs):
        self.channels = self._setup_channels()
        cadence = 20 * u.s
        resolution = [2.05719995499, 2.05719995499] * u.arcsec/u.pixel
        super().__init__(observing_time, observer, cadence, resolution, **kwargs)

    def _setup_channels(self):
        with open(pkg_resources.resource_filename('synthesizAR', 'instruments/data/hinode_xrt.json'), 'r') as f:
            info = json.load(f)
        channels = []
        for k in info:
            if k in ('name', 'description'):
                continue
            x = np.array(info[k]['temperature_response_x'], np.float64)
            y = np.array(info[k]['temperature_response_y'], np.float64)
            x = x * u.Unit(info[k]['temperature_response_x_units'])
            y = y * u.Unit(info[k]['temperature_response_y_units'])
            c = ChannelXRT(
                name=k,
                channel=50*u.angstrom,  # this is just a filler value, not important
                temperature=x,
                response=y,
                filter_wheel_1=info[k]['filter_wheel_1'],
                filter_wheel_2=info[k]['filter_wheel_2'],
            )
            channels.append(c)
        return channels

    @property
    def detector(self):
        return self.name.split('_')[-1]

    @property
    def telescope(self):
        return self.name.split('_')[0]

    def get_instrument_name(self, channel):
        return self.detector

    def get_header(self, channel, coordinates):
        header = super().get_header(channel, coordinates)
        header['EC_FW1_'] = channel.filter_wheel_1
        header['EC_FW2_'] = channel.filter_wheel_2
        return header

    @staticmethod
    def calculate_intensity_kernel(loop, channel, **kwargs):
        return np.interp(loop.electron_temperature, channel.temperature, channel.response) * loop.density**2

    def psf_smooth(self, counts, header):
        """
        Apply point-spread-function smoothing to XRT image using the PSF given in [1]_

        .. note:: This is not currently used anywhere. Just leaving it in here.

        References
        ----------
        .. [1] Afshari, M., et al., AJ, `152, 107 <http://iopscience.iop.org/article/10.3847/0004-6256/152/4/107/meta>`_
        """
        tmp = Map(counts, header)

        # Define the PSF
        def point_spread_function(x, y):
            r = np.sqrt(x**2 + y**2)
            a = 1.31946
            gamma = 1.24891
            if r <= 3.4176:
                return a*np.exp(-(r/a)**2)/(gamma**2 + r**2)
            elif 3.4176 < r <= 5:
                return 0.03/r
            elif 5 < r <= 11.1:
                return 0.15/r
            elif r >= 11.1:
                return (11.1)**2*0.15/(r**4)

        # Create lon/lat grid
        lon = np.linspace(tmp.bottom_left_coord.Tx.value, tmp.top_right_coord.Tx.value, 250+1)
        lat = np.linspace(tmp.bottom_left_coord.Ty.value, tmp.top_right_coord.Ty.value, 250+1)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        # Make PSF kernel
        psf_kernel = astropy.convolution.kernels.CustomKernel(
                        np.vectorize(point_spread_function)(lon_grid, lat_grid))
        # Convolve with image
        counts_blurred = astropy.convolution.convolve_fft(counts, psf_kernel)

        return counts_blurred


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

        return EISCube(data=counts, header=header, wavelength=response_x)
