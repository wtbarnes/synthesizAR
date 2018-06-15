"""
Containers for multidimensional data
"""

import os

import numpy as np
import h5py
import astropy.io.fits
import astropy.units as u
import sunpy.cm
from sunpy.map import Map, MapCube, GenericMap
from sunpy.util.metadata import MetaDict
from sunpy.io.fits import get_header

__all__ = ['EMCube', 'EISCube']


class EMCube(MapCube):
    """
    Container for the emission measure at each pixel of a map for a range of temperatures.

    Parameters
    ----------
    data : `~astropy.units.Quantity`
        3D array (2D spatial + temperature) cube of emission measure data
    header : `~sunpy.util.metadata.MetaDict`
    temperature_bin_edges : `~astropy.units.Quantity`
        Should have same shape as third dimension of `data`

    Other Parameters
    ----------------
    plot_settings : `dict`
    """

    @u.quantity_input
    def __init__(self, data, header, temperature_bin_edges: u.K, **kwargs):
        self.temperature_bin_edges = temperature_bin_edges
        # sanitize header
        meta_base = header.copy()
        meta_base['temp_unit'] = self.temperature_bin_edges.unit.to_string()
        meta_base['bunit'] = data.unit.to_string()
        # build map list
        map_list = []
        for i in range(self.temperature_bin_edges.shape[0] - 1):
            tmp = GenericMap(data[:, :, i], meta_base)
            tmp.meta['temp_a'] = self.temperature_bin_edges[i].value
            tmp.meta['temp_b'] = self.temperature_bin_edges[i+1].value
            tmp.plot_settings.update(kwargs.get('plot_settings', {}))
            map_list.append(tmp)

        # call super method
        super().__init__(map_list)

    @property
    def total_emission(self):
        """
        Sum the emission measure over all temperatures.
        """
        tmp_meta = self[0].meta.copy()
        tmp_meta['temp_a'] = self.temperature_bin_edges[0]
        tmp_meta['temp_b'] = self.temperature_bin_edges[-1]
        return GenericMap(self.as_array().sum(axis=2), tmp_meta,
                          plot_settings=self[0].plot_settings)

    def get_1d_distribution(self, bottom_left_corner, top_right_corner):
        """
        Mean emission measure distribution over area defined by corners.

        Parameters
        ----------
        bottom_left_corner : `~astropy.coordinates.SkyCoord`
        top_right_corner : `~astropy.coordinates.SkyCoord`
        """
        em_list = []
        for i in range(self.temperature_bin_edges.shape[0] - 1):
            em_list.append(self[i].submap(bottom_left_corner, top_right_corner).data.mean())

        return u.Quantity(em_list, u.Unit(self[0].meta['bunit']))

    def make_slope_map(self, temperature_bounds=None, em_threshold=None, rsquared_tolerance=0.5):
        """
        Calculate emission measure slope :math:`a` in each pixel

        Create map of emission measure slopes by fitting :math:`\mathrm{EM}\sim T^a` for a
        given temperature range. Only those pixels for which the minimum :math:`\mathrm{EM}`
        across all temperature bins is above some threshold value.

        .. warning:: This method provides no measure of the goodness of the fit. Some slope values
                     may not provide an accurate fit to the data.

        Parameters
        ----------
        temperature_bounds : `~astropy.units.Quantity`, optional
        em_threshold : `~astropy.units.Quantity`, optional
            Mask emission measure below this value
        rsquared_tolerance : `float`
            Throw away slopes with :math:`r^2` below this value
        """
        if temperature_bounds is None:
            temperature_bounds = u.Quantity((1e6, 4e6), u.K)
        if em_threshold is None:
            em_threshold = u.Quantity(1e25, u.cm**(-5))
        # cut on temperature
        temperature_bin_centers = (self.temperature_bin_edges[:-1]
                                   + self.temperature_bin_edges[1:])/2.
        index_temperature_bounds = np.where(np.logical_and(
            temperature_bin_centers >= temperature_bounds[0],
            temperature_bin_centers <= temperature_bounds[1]))
        temperature_fit = temperature_bin_centers[index_temperature_bounds].value
        # unwrap to 2D and threshold
        data = self.as_array()*u.Unit(self[0].meta['bunit'])
        flat_data = data.reshape(np.prod(data.shape[:2]), temperature_bin_centers.shape[0])
        index_data_threshold = np.where(np.min(
            flat_data[:, index_temperature_bounds[0]], axis=1) >= em_threshold)
        flat_data_threshold = flat_data.value[index_data_threshold[0], :][:, index_temperature_bounds[0]]
        # very basic but vectorized fitting
        _, rss_flat, _, _, _ = np.polyfit(
            np.log10(temperature_fit), np.log10(flat_data_threshold.T), 0, full=True)
        coefficients, rss, _, _, _ = np.polyfit(
            np.log10(temperature_fit), np.log10(flat_data_threshold.T), 1, full=True)
        rsquared = 1. - rss/rss_flat
        slopes = np.where(rsquared >= rsquared_tolerance, coefficients[0], 0.)
        # rebuild into a map
        slopes_flat = np.zeros(flat_data.shape[0]) * np.nan
        slopes_flat[index_data_threshold[0]] = slopes
        slopes_2d = np.reshape(slopes_flat, data.shape[:2])
        base_meta = self[0].meta.copy()
        base_meta['temp_a'] = temperature_fit[0]
        base_meta['temp_b'] = temperature_fit[-1]
        base_meta['bunit'] = ''
        base_meta['detector'] = 'EM slope'
        base_meta['comment'] = 'Linear fit to log-transformed LOS EM'
        plot_settings = self[0].plot_settings.copy()
        plot_settings['norm'] = None

        return GenericMap(slopes_2d, base_meta, plot_settings=plot_settings)

    def __getitem__(self, key):
        """
        Override the MapCube indexing so that an `EMCube` object is returned.
        """
        if type(self.temperature_bin_edges[key].value) == np.ndarray and \
           len(self.temperature_bin_edges[key].value) > 1:
            tmp_data = u.Quantity(self.as_array()[:, :, key], u.Unit(self.maps[0].meta['bunit']))
            tmp_meta = self.maps[0].meta.copy()
            tmp = EMCube(tmp_data, tmp_meta, self.temperature_bin_edges[key],
                         plot_settings=self.maps[0].plot_settings)
        else:
            tmp = self.maps[key]
            tmp.meta['temp_a'] = self.temperature_bin_edges[key].value
            tmp.meta['temp_b'] = self.temperature_bin_edges[key+1].value

        return tmp

    def save(self, filename):
        """
        Save emission measure cube to an HDF5 file.
        """
        with h5py.File(filename, 'x') as hf:
            dset_data = hf.create_dataset('emission_measure', data=self.as_array())
            dset_data.attrs['unit'] = self[0].meta['bunit']
            dset_temperature_bin_edges = hf.create_dataset(
                'temperature_bin_edges', data=self.temperature_bin_edges.value)
            dset_temperature_bin_edges.attrs['unit'] = self.temperature_bin_edges.unit.to_string()
            meta_group = hf.create_group('meta')
            for key in self[0].meta:
                meta_group.attrs[key] = self[0].meta[key]

    @classmethod
    def restore(cls, filename, **kwargs):
        """
        Restore `EMCube` from an HDF5 file.
        """
        header = MetaDict()
        with h5py.File(filename, 'r') as hf:
            data = u.Quantity(hf['emission_measure'], hf['emission_measure'].attrs.get(
                'unit', hf['emission_measure'].attrs.get('units')))
            temperature_bin_edges = u.Quantity(
                hf['temperature_bin_edges'], hf['temperature_bin_edges'].attrs.get(
                    'unit', hf['temperature_bin_edges'].attrs.get('units')))
            for key in hf['meta'].attrs:
                header[key] = hf['meta'].attrs[key]

        return cls(data, header, temperature_bin_edges, **kwargs)


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
            wavelength = np.array(hf['wavelength'])*u.Unit(hf['wavelength'].attrs.get(
                'unit', hf['intensity'].attrs.get('units')))
            data = np.array(hf['intensity'])*u.Unit(hf['intensity'].attrs.get(
                'unit', hf['intensity'].attrs.get('units')))

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
