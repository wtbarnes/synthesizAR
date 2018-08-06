"""
Class for Hinode/EIS instrument. Holds information about spectral, temporal, and spatial resolution
and other instrument-specific information.
"""

import os
import json
import pkg_resources

import numpy as np
from scipy.interpolate import splrep, splev, interp1d
from scipy.ndimage.filters import gaussian_filter
from sunpy.util.metadata import MetaDict
from sunpy.map import Map, MapCube, GenericMap
from sunpy.io.fits import get_header
import sunpy.cm
import astropy.units as u
import astropy.io.fits
import astropy.constants as const
import astropy.convolution
import h5py
import plasmapy
import dask

from synthesizAR.util import SpatialPair, get_keys
from synthesizAR.instruments import InstrumentBase
from synthesizAR.maps import EISCube


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


class InstrumentHinodeXRT(InstrumentBase):

    def __init__(self, observing_time, observer_coordinate, apply_psf=True):
        self.name = 'Hinode_XRT'
        self.cadence = 20*u.s
        self.resolution = SpatialPair(x=2.05719995499*u.arcsec/u.pixel,
                                      y=2.05719995499*u.arcsec/u.pixel, z=None)
        self.fits_template['telescop'] = 'Hinode'
        self.fits_template['instrume'] = 'XRT'
        self.fits_template['waveunit'] = 'keV'
        self.apply_psf = apply_psf
        super().__init__(observing_time, observer_coordinate)
        self._setup_channels()

    def _setup_channels(self):
        """
        Setup XRT channel properties

        Notes
        -----
        Need temperature response functions only for now
        TODO: include wavelength response functions
        """
        fn = pkg_resources.resource_filename('synthesizAR', 'instruments/data/hinode_xrt.json')
        with open(fn, 'r') as f:
            info = json.load(f)

        self.channels = []
        for k in info:
            if k in ('name', 'description'):
                continue
            x = np.array(info[k]['temperature_response_x'], np.float64)
            y = np.array(info[k]['temperature_response_y'], np.float64)
            name = f"{info[k]['filter_wheel_1']}-{info[k]['filter_wheel_2']}"
            self.channels.append({
                'name': name,
                'temperature_response_spline': splrep(x, y),
                'wavelength_range': None,
            })

    def make_fits_header(self, field, channel):
        """
        Build XRT FITS header file
        """
        header = super().make_fits_header(field, channel)
        header['EC_FW1_'], header['EC_FW2_'] = channel['name'].split('-')
        return header

    def build_detector_file(self, file_template, dset_shape, chunks, *args, parallel=False):
        """
        Allocate space for counts data.
        """
        additional_fields = [channel['name'] for channel in self.channels]
        super().build_detector_file(file_template, dset_shape, chunks, *args, additional_fields=additional_fields, 
                                    parallel=parallel)

    @staticmethod
    def calculate_counts_simple(channel, loop, *args):
        """
        Use temperature response to calculate XRT intensity
        """
        response_function = (splev(np.ravel(loop.electron_temperature), channel['temperature_response_spline'])
                             * u.count*u.cm**5/u.s/u.pixel)
        counts = np.reshape(np.ravel(loop.density**2)*response_function, loop.density.shape)
        return counts

    def flatten_parallel(self, loops, interpolated_loop_coordinates, save_path, emission_model=None):
        """
        Interpolate intensity in each channel to temporal resolution of the instrument
        and appropriate spatial scale. Returns a dask task.
        """
        tasks = {}
        for channel in self.channels:
            tasks[channel['name']] = []
            flattened_emissivities = []
            for loop, interp_s in zip(loops, interpolated_loop_coordinates):
                y = dask.delayed(self.calculate_counts_simple)(channel, loop, emission_model, flattened_emissivities)
                tmp_path = save_path.format(channel['name'], loop.name)
                task = dask.delayed(self.interpolate_and_store)(y, loop, self.observing_time, interp_s, tmp_path)
                tasks[channel['name']].append(task)

        return tasks

    @staticmethod
    def _detect(counts_filename, observer_coordinate, channel, i_time, header, bins, bin_range,
                apply_psf):
        """
        For a given channel and timestep, map the intensity along the loop to the 3D field and
        return the XRT data product.

        Parameters
        ----------
        counts_filename : `str`
        channel : `dict`
        i_time : `int`
        header : `~sunpy.util.metadata.MetaDict`
        bins : `SpatialPair`
        bin_range : `SpatialPair`
        apply_psf : `bool`

        Returns
        -------
        XRT data product : `~sunpy.Map`
        """
        with h5py.File(counts_filename, 'r') as hf:
            weights = np.array(hf[channel['name']][i_time, :])
            units = u.Unit(get_keys(hf[channel['name']].attrs, ('unit', 'units')))

        hpc_coordinates = self.total_coordinates
        dz = np.diff(bin_range.z).cgs[0] / bins.z * (1. * u.pixel)
        visible = is_visible(hpc_coordinates, observer_coordinate)
        hist, _, _ = np.histogram2d(hpc_coordinates.Tx.value, hpc_coordinates.Ty.value,
                                    bins=(bins.x.value, bins.y.value),
                                    range=(bin_range.x.value, bin_range.y.value),
                                    weights=visible * weights * dz.value)
        header['bunit'] = (units * coordinates.unit).to_string()

        if apply_psf:
            counts = InstrumentHinodeXRT.psf_smooth(hist.T, header)
        return Map(counts, header)
        
    def detect(self, channel, i_time, field, parallel=False):
        header = self.make_fits_header(field, channel)
        parameters = (self.counts_file, self.observer_coordinate, channel, i_time, header, 
                      self.bins, self.bin_range, self.apply_psf)
        if parallel:
            return dask.delayed(self._detect)(*parameters)
        else:
            return self._detect(*parameters)
    
    @staticmethod
    def psf_smooth(counts, header):
        """
        Apply point-spread-function smoothing to XRT image using the PSF given in [1]_

        References
        ----------
        .. [1] Afshari, M., et al., AJ, `152, 107 <http://iopscience.iop.org/article/10.3847/0004-6256/152/4/107/meta>`_
        """
        tmp = Map(counts, header)

        # Define the PSF
        def point_spread_function(x, y):
            r = np.sqrt(x**2 + y**2)
            a = 1.31946
            sigma = 2.19256
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


class EISCube(object):
    """
    Spectral and spatial cube for holding Hinode EIS data
    """

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and os.path.exists(args[0]):
            data, header, wavelength = self._restore_from_file(args[0], **kwargs)
        elif all([k in kwargs for k in ['data', 'header', 'wavelength']]):
            data = kwargs.get('data')
            header = kwargs.get('header')
            wavelength = kwargs.get('wavelength')
        else:
            raise ValueError('''EISCube can only be initialized with a valid FITS file or NumPy
                                array with an associated wavelength and header.''')
        # check dimensions
        if data.shape[-1] != wavelength.shape[0]:
            raise ValueError('''Third dimension of data cube must have the same length as
                                wavelength.''')
        self.meta = header.copy()
        self.wavelength = wavelength
        self.data = data
        self.cmap = kwargs.get('cmap', sunpy.cm.get_cmap('hinodexrt'))
        self._fix_header()

    def __repr__(self):
        return f'''synthesizAR {type(self).__name__}
-----------------------------------------
Telescope : {self.meta['telescop']}
Instrument : {self.meta['instrume']}
Area : x={self[0].xrange}, y={self[0].yrange}
Dimension : {u.Quantity(self[0].dimensions)}
Scale : {u.Quantity(self[0].scale)}
Wavelength range : {u.Quantity([self.wavelength[0], self.wavelength[-1]])}
Wavelength dimension : {len(self.wavelength)}'''

    def __getitem__(self, key):
        """
        Overriding indexing. If key is just one index, returns a normal `Map` object. Otherwise,
        another `EISCube` object is returned.
        """
        if type(self.wavelength[key].value) == np.ndarray and len(self.wavelength[key].value) > 1:
            new_meta = self.meta.copy()
            new_meta['wavelnth'] = (self.wavelength[key][0].value+self.wavelength[key][-1].value)/2.
            return EISCube(data=self.data[:, :, key], header=new_meta,
                           wavelength=self.wavelength[key])
        else:
            meta_map2d = self.meta.copy()
            meta_map2d['naxis'] = 2
            for k in ['naxis3', 'ctype3', 'cunit3', 'cdelt3']:
                del meta_map2d[k]
            meta_map2d['wavelnth'] = self.wavelength[key].value
            tmp_map = Map(self.data[:, :, key], meta_map2d)
            tmp_map.plot_settings.update({'cmap': self.cmap})
            return tmp_map

    def submap(self, bottom_left_corner, top_right_corner):
        """
        Crop to spatial area designated by corners

        .. warning:: It is faster to crop in wavelength space first and then crop in
                     coordinate space.
        """
        # call submap on each slice in wavelength
        new_data = []
        for i in range(self.wavelength.shape[0]):
            new_data.append(self[i].submap(bottom_left_corner, top_right_corner).data)
        new_data = np.stack(new_data, axis=2)*self.data.unit
        # fix metadata
        new_meta = self[0].submap(bottom_left_corner, top_right_corner).meta.copy()
        for key in ['wavelnth', 'naxis3', 'ctype3', 'cunit3', 'cdelt3']:
            new_meta[key] = self.meta[key]

        return EISCube(data=new_data, header=new_meta, wavelength=self.wavelength)

    def __add__(self, x):
        """
        Allow EISCubes to be added together
        """
        if isinstance(x, EISCube):
            assert np.all(self.wavelength == x.wavelength), 'Wavelength ranges must be equal in order to add EISCubes'
            key_checks = ['cdelt1', 'cdelt2', 'crpix1', 'crpix2', 'ctype1', 'ctype2', 'crval1',
                          'crval2']
            for k in key_checks:
                assert self.meta[k] == x.meta[k], f'{k} keys in metadata do not match'
            data = self.data + x.data
        else:
            # if x is not an instance of EISCube, let numpy/astropy decide whether it can
            # be added to the data attribute, e.g. a scalar or some 3D array with
            # appropriate units
            data = self.data + x
        return EISCube(data=data, header=self.meta.copy(), wavelength=self.wavelength)

    def __radd__(self, x):
        """
        Define reverse addition in the same way as addition.
        """
        return self.__add__(x)

    def __mul__(self, x):
        """
        Allow for multiplication of data in the cube.
        """
        x = u.Quantity(x)
        data = self.data*x
        header = self.meta.copy()
        header['bunit'] = (data.unit).to_string()
        return EISCube(data=data, header=header, wavelength=self.wavelength)

    def __rmul__(self, x):
        """
        Define reverse multiplication in the same way as multiplication.
        """
        return self.__mul__(x)

    def _fix_header(self):
        """
        Set any missing keys, reset any broken ones
        """
        # assuming y is rows, x is columns
        self.meta['naxis1'] = self.data.shape[1]
        self.meta['naxis2'] = self.data.shape[0]
        self.meta['naxis3'] = self.wavelength.shape[0]

    def save(self, filename, use_fits=False, **kwargs):
        """
        Save to FITS or HDF5 file. Default is HDF5 because this is faster and produces smaller
        files.
        """
        if use_fits:
            self._save_to_fits(filename, **kwargs)
        else:
            # change extension for clarity
            filename = '.'.join([os.path.splitext(filename)[0], 'h5'])
            self._save_to_hdf5(filename, **kwargs)

    def _save_to_hdf5(self, filename, **kwargs):
        """
        Save to HDF5 file.
        """
        dset_save_kwargs = kwargs.get(
            'hdf5_save_params', {'compression': 'gzip', 'dtype': np.float32})
        with h5py.File(filename, 'x') as hf:
            meta_group = hf.create_group('meta')
            for key in self.meta:
                meta_group.attrs[key] = self.meta[key]
            dset_wvl = hf.create_dataset('wavelength', data=self.wavelength.value)
            dset_wvl.attrs['unit'] = self.wavelength.unit.to_string()
            dset_intensity = hf.create_dataset('intensity', data=self.data, **dset_save_kwargs)
            dset_intensity.attrs['unit'] = self.data.unit.to_string()

    def _save_to_fits(self, filename, **kwargs):
        """
        Save to FITS file
        """
        # sanitize header
        header = self.meta.copy()
        if 'keycomments' in header:
            del header['keycomments']

        # create table to hold wavelength array
        table_hdu = astropy.io.fits.BinTableHDU.from_columns(
            [astropy.io.fits.Column(name='wavelength',
                                    format='D',
                                    unit=self.wavelength.unit.to_string(),
                                    array=self.wavelength.value)])
        # create image to hold 3D array
        image_hdu = astropy.io.fits.PrimaryHDU(np.swapaxes(self.data.value.T, 1, 2),
                                               header=astropy.io.fits.Header(header))
        # write to file
        hdulist = astropy.io.fits.HDUList([image_hdu, table_hdu])
        hdulist.writeto(filename, output_verify='silentfix')

    def _restore_from_file(self, filename, **kwargs):
        """
        Load from HDF5 or FITS file
        """
        use_fits = kwargs.get('use_fits', os.path.splitext(filename)[-1] == '.fits')
        use_hdf5 = kwargs.get('use_hdf5', os.path.splitext(filename)[-1] == '.h5')
        if use_fits:
            data, header, wavelength = self._restore_from_fits(filename)
        elif use_hdf5:
            data, header, wavelength = self._restore_from_hdf5(filename)
        else:
            raise ValueError('Cube can only be initialized with a FITS or HDF5 file.')

        return data, header, wavelength

    def _restore_from_hdf5(self, filename):
        """
        Helper to load cube from HDF5 file
        """
        header = MetaDict()
        with h5py.File(filename, 'r') as hf:
            for key in hf['meta'].attrs:
                header[key] = hf['meta'].attrs[key]
            wavelength = u.Quantity(hf['wavelength'],
                                    get_keys(hf['wavelength'].attrs, ('unit', 'units')))
            data = u.Quantity(hf['intensity'], get_keys(hf['intensity'].attrs, ('unit', 'units')))

        return data, header, wavelength

    def _restore_from_fits(self, filename):
        """
        Helper to load cube from FITS file
        """
        tmp = astropy.io.fits.open(filename)
        header = MetaDict(get_header(tmp)[0])
        data = tmp[0].data*u.Unit(header['bunit'])
        wavelength = tmp[1].data.field(0)*u.Unit(tmp[1].header['TUNIT1'])
        tmp.close()

        return np.swapaxes(data.T, 0, 1), header, wavelength

    @property
    def integrated_intensity(self):
        """
        Map of the intensity integrated over wavelength.
        """
        tmp = np.dot(self.data, np.gradient(self.wavelength.value))
        tmp_meta = self[0].meta.copy()
        tmp_meta['wavelnth'] = self.meta['wavelnth']
        tmp_meta['bunit'] = (u.Unit(self.meta['bunit'])*self.wavelength.unit).to_string()
        tmp_map = Map(tmp, tmp_meta)
        tmp_map.plot_settings.update({'cmap': self.cmap})

        return tmp_map