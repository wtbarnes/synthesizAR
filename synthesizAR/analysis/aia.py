"""
Analyze AIA data efficiently
"""
import warnings

import numpy as np
import dask.bytes
import dask.array as da
import distributed
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
from astropy.io.fits.hdu.base import BITPIX2DTYPE
from astropy.coordinates import SkyCoord
import sunpy.map
from sunpy.util.metadata import MetaDict

__all__ = ['DistributedAIACube']


def validate_dtype_shape(head):
    naxes = head['NAXIS']
    dtype = BITPIX2DTYPE[head['BITPIX']]
    shape = [head[f'NAXIS{n}'] for n in range(naxes, 0, -1)]
    return dtype, shape


def get_header(fn, hdu=0):
    with fn as fi:
        return MetaDict(sunpy.io.fits.get_header(fi)[hdu])
    

class DelayedFITS:
    def __init__(self, file, shape, dtype, hdu=0):
        self.shape = shape
        self.dtype = dtype
        self.file = file
        self.hdu = hdu
    
    def __getitem__(self, item):
        with self.file as fi:
            with fits.open(fi) as hdul:
                return hdul[self.hdu].section[item]
            

class DistributedAIACube(object):
    """
    Load sequence of AIA images for a single channel and operate on them in a distributed and
    parallelized way.
    
    Object for dealing efficiently with large out-of-core AIA datacubes.
    Nearly all of this is inspired by work done by Stuart Mumford, in particular how to handle
    FITS files in Dask.

    #TODO: Refactor this to use ndcube instead
    """
    def __init__(self, maps):
        if not all([m.data.shape == maps[0].data.shape for m in maps]):
            raise ValueError('All maps must have same dimensions')
        if not all([m.data.dtype == maps[0].data.dtype for m in maps]):
            raise ValueError('All maps must have same dtype')
        self.maps = maps

    @classmethod
    def from_files(cls, read_template):
        openfiles = dask.bytes.open_files(read_template)
        headers = cls._get_headers(openfiles)
        dtype, shape = cls._get_dtype_and_shape(headers)
        maps = cls._get_maps(openfiles, headers, dtype, shape)
        return cls(maps)

    @staticmethod
    def _get_maps(openfiles, headers, dtype, shape):
        arrays = [da.from_array(DelayedFITS(f, shape=shape, dtype=dtype, hdu=0),
                                chunks=shape) for f in openfiles]
        return [sunpy.map.Map(a, h) for a, h in zip(arrays, headers)]

    @staticmethod
    def _get_headers(openfiles):
        client = distributed.get_client()
        futures = client.map(get_header, openfiles)
        return client.gather(futures)

    @staticmethod
    def _get_dtype_and_shape(headers):
        dtypes = [validate_dtype_shape(h) for h in headers]
        if not all([d == dtypes[0] for d in dtypes]):
            raise ValueError('All maps must have same shape and dtype')
        return dtypes[0]

    @property
    def time(self,):
        """
        Create time array
        """
        # Simulations just store the seconds, should probably fix that
        if 'tunit' in self.maps[0].meta:
            return u.Quantity([m.meta['t_obs'] for m in self.maps], self.maps[0].meta['tunit'])
        else:
            return u.Quantity([(Time(m.meta['t_obs']) - Time(self.maps[0].meta['t_obs'])).to(u.s) 
                               for m in self.maps])
            
    @property
    def shape(self,):
        return self.time.shape + self.maps[0].data.shape

    @property
    def dtype(self,):
        return self.maps[0].data.dtype

    @property
    def unstacked_data(self,):
        return [m.data for m in self.maps]

    @property
    def stacked_data(self,):
        return da.stack(self.unstacked_data)

    def rechunk(self, shape):
        return self.stacked_data.rechunk(shape)
    
    def average(self, **kwargs):
        """
        Time-average in each pixel
        """
        chunks = kwargs.get('chunks', (self.shape[0], self.shape[1]//10, self.shape[2]//10))
        cube = self.rechunk(chunks)
        # FIXME: should this be a weighted average? How to calculate the weights?
        # FIXME: should we modify any of the metadata before taking an average?
        return sunpy.map.Map(cube.mean(axis=0,dtype=np.float64), self.maps[0].meta.copy())

    def prep(self,):
        """
        Lazily apply a prep operation to all maps and return a new distributed cube object
        """
        pass

    def derotate(self, reference_date):
        """
        Lazily apply a derotation to all maps and return a new distributed cube object
        """
        pass


class DistributedAIACollection(object):
    """
    A collection of distributed AIA images over multiple wavelengths

    #TODO: refactor this to use ndcube sequence
    #TODO: figure out useful methods to add here
    #TODO: Add a check for data being aligned?
    #NOTE: It is assumed that the data in this container are all aligned, i.e. prepped and derotated
    """

    def __init__(self, *args, **kwargs):
        # Check all spatial and time shapes the same
        if not all([a.shape[1:] == args[0].shape[1:] for a in args]):
            raise ValueError('All spatial dimensions must be the same')
        if not all([a.shape[0] == args[0].shape[0] for a in args]):
            # Not an error because may have missing timesteps in observations
            # Will interpolate later to account for this
            raise warnings.warn('Time dimensions are not all equal length')
        self._cubes = {f"{a.maps[0].meta['wavelnth']}": a for a in args}
        self.channels = list(self._cubes.keys())

    def __getitem__(self, channel):
        channel = f'{channel}'
        return self._cubes[channel]


class AIATimeLags(DistributedAIACollection):
    """
    Compute AIA timelag maps in a distributed way
    """
    @property
    def timelags(self):
        if np.all([np.all(self[c].time == self[self.channels[0]].time) for c in self.channels]):
            delta_t = np.diff(self[self.channels[0]].time.value).cumsum()
            return np.hstack([-delta_t[::-1], np.array([0]), delta_t]) * self[self.channels[0]].time.unit
        else:
            t_interp = self._interpolate_time()
            delta_t = np.diff(t_interp.value).cumsum()
            return np.hstack([-delta_t[::-1], np.array([0]), delta_t]) * t_interp.unit
    
    @property
    def _interpolate_time(self,):
        min_t = min([self[c].time.min() for c in self.channels])
        max_t = max([self[c].time.max() for c in self.channels])
        n_t = max([self[c].time.shape[0] for c in self.channels])
        return np.linspace(min_t, max_t, n_t)
    
    def _interpolate(self, cube, time):
        #t_interp = self._interpolate_time
        #return da.apply_along_axis(self._wrap_numpy_interp, 0, cube, time, t_interp)
        # FIXME: issues with applying np interp
        pass
        
    @staticmethod
    def _wrap_numpy_interp(array, time, new_time):
        print(time.shape)
        print(array.shape)
        return np.interp(new_time, time, array)

    def make_timeseries(self, channel, left_corner, right_corner, **kwargs):
        tmp = self[channel].maps[0]
        x_l, y_l = tmp.world_to_pixel(SkyCoord(*left_corner, frame=tmp.coordinate_frame))
        x_u, y_u = tmp.world_to_pixel(SkyCoord(*right_corner, frame=tmp.coordinate_frame))
        x_l, y_l, x_u, y_u = np.round([x_l.value, y_l.value, x_u.value, y_u.value]).astype(np.int)
        chunks = kwargs.get('chunks', (self[channel].shape[0], self[channel].shape[1]//10, self[channel].shape[2]//10))
        return (self[channel].rechunk(chunks)[:, y_l:y_u, x_l:x_u].mean(axis=(1, 2)))

    def correlation_1d(self, channel_a, channel_b, left_corner, right_corner, **kwargs):
        ts_a = self.make_timeseries(channel_a, left_corner, right_corner, **kwargs)
        ts_b = self.make_timeseries(channel_b, left_corner, right_corner, **kwargs)
        ts_a = (ts_a - ts_a.mean()) / ts_a.std()
        ts_b = (ts_b - ts_b.mean()) / ts_b.std()
        cc = da.fft.irfft(da.fft.rfft(ts_a[::-1], n=self.timelags.shape[0])
                          * da.fft.rfft(ts_b, n=self.timelags.shape[0]), n=self.timelags.shape[0])
        return cc

    def correlation_2d(self, channel_a, channel_b, **kwargs):
        """
        Create Dask task graph to compute cross-correlation using FFT for each pixel in an AIA map
        """
        chunks = kwargs.get('chunks',(self[channel_a].shape[0],
                                      self[channel_a].shape[1]//10,
                                      self[channel_a].shape[2]//10))
        cube_a = self[channel_a].rechunk(chunks)[::-1, :, :]
        cube_b = self[channel_b].rechunk(chunks)
        # Normalize
        std_a = cube_a.std(axis=0)
        std_a = da.where(std_a == 0, 1, std_a)
        v_a = (cube_a - cube_a.mean(axis=0)[np.newaxis, :, :]) / std_a[np.newaxis, :, :]
        std_b = cube_b.std(axis=0)
        std_b = da.where(std_b == 0, 1, std_b)
        v_b = (cube_b - cube_b.mean(axis=0)[np.newaxis, :, :]) / std_b[np.newaxis, :, :]
        # Fast Fourier Transform of both channels
        fft_a = da.fft.rfft(v_a, axis=0, n=self.timelags.shape[0])
        fft_b = da.fft.rfft(v_b, axis=0, n=self.timelags.shape[0])
        # Inverse of product of FFTS to get cross-correlation (by convolution theorem)
        return da.fft.irfft(fft_a * fft_b, axis=0, n=self.timelags.shape[0])

    def make_correlation_map(self, channel_a, channel_b, **kwargs):
        """
        Build map of max correlation value between two AIA channels
        """
        cc = self.correlation_2d(channel_a, channel_b, **kwargs)
        bounds = kwargs.get('timelag_bounds', None)
        if bounds is not None:
            indices, = np.where(np.logical_and(self.timelags >= bounds[0],
                                               self.timelags <= bounds[1]))
            start = indices[0]
            stop = indices[-1] + 1
        else:
            start = 0
            stop = self.timelags.shape[0] + 1
        max_cc = cc[start:stop, :, :].max(axis=0).compute()
        # Metadata
        meta = self[channel_a].maps[0].meta.copy()
        del meta['instrume']
        del meta['t_obs']
        del meta['wavelnth']
        meta['bunit'] = ''
        meta['comment'] = f'{channel_a}-{channel_b} cross-correlation'

        plot_settings = {'cmap': 'plasma'}
        plot_settings.update(kwargs.get('plot_settings', {}))
        correlation_map = sunpy.map.GenericMap(max_cc, meta, plot_settings=plot_settings)

        return correlation_map

    def make_timelag_map(self, channel_a, channel_b, **kwargs):
        """
        Compute map of timelag associated with maximum cross-correlation between
        two channels in each pixel of an AIA map.

        #NOTE: Extra block for computing correlation map is to save time
        """
        cc = self.correlation_2d(channel_a, channel_b, **kwargs)
        bounds = kwargs.get('timelag_bounds', None)
        if bounds is not None:
            indices, = np.where(np.logical_and(self.timelags >= bounds[0],
                                               self.timelags <= bounds[1]))
            start = indices[0]
            stop = indices[-1] + 1
        else:
            start = 0
            stop = self.timelags.shape[0] + 1
        if kwargs.get('return_correlation_map'):
            _cc = cc.compute()
            max_cc = _cc[start:stop, :, :].max(axis=0)
            i_max_cc = _cc[start:stop, :, :].argmax(axis=0)
            del _cc
        else:
            i_max_cc = cc[start:stop, :, :].argmax(axis=0).compute()
        max_timelag = self.timelags[start:stop][i_max_cc]
        # Metadata
        meta = self[channel_a].maps[0].meta.copy()
        del meta['instrume']
        del meta['t_obs']
        del meta['wavelnth']
        meta['bunit'] = 's'
        meta['comment'] = f'{channel_a}-{channel_b} timelag'
        plot_settings = {'cmap': 'RdBu_r', 'vmin': self.timelags[start:stop].value.min(),
                         'vmax': self.timelags[start:stop].value.max()}
        plot_settings.update(kwargs.get('plot_settings', {}))
        timelag_map = sunpy.map.GenericMap(max_timelag, meta.copy(), plot_settings=plot_settings.copy())
        if kwargs.get('return_correlation_map'):
            meta['bunit'] = ''
            meta['comment'] = f'{channel_a}-{channel_b} cross-correlation'
            plot_settings['cmap'] = 'plasma'
            del plot_settings['vmin']
            del plot_settings['vmax']
            correlation_map = sunpy.map.GenericMap(max_cc, meta, plot_settings=plot_settings)
            return timelag_map, correlation_map
        else:
            return timelag_map
