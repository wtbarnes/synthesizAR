"""
Analyze AIA data efficiently
"""
import warnings

import numpy as np
from scipy.interpolate import interp1d
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

__all__ = ['DistributedAIACube', 'DistributedAIACollection', 'AIATimelags']


def validate_dtype_shape(head):
    naxes = head['NAXIS']
    dtype = BITPIX2DTYPE[head['BITPIX']]
    shape = [head[f'NAXIS{n}'] for n in range(naxes, 0, -1)]
    return dtype, shape


def get_header(fn, hdu=0):
    with fn as fi:
        return MetaDict(sunpy.io.fits.get_header(fi)[hdu])


class DelayedFITS:
    def __init__(self, file, shape, dtype, hdu=0, verify=False):
        self.shape = shape
        self.dtype = dtype
        self.file = file
        self.hdu = hdu
        self.verify = verify

    def __getitem__(self, item):
        with self.file as fi:
            with fits.open(fi, memmap=True) as hdul:
                if self.verify:
                    hdul.verify('silentfix+warn')
                return hdul[self.hdu].data[item]


class DistributedAIACube(object):
    """
    Lazy load a sequence of AIA images for a single channel with Dask

    .. warning:: This object is unstable and will likely be moved out of this package in the near future

    .. note:: Nearly all of the early work was  done by Stuart Mumford, in particular how to handle FITS files in Dask.

    Parameters
    ----------
    maps : `list`
        List of `~sunpy.map.Map` objects

    See Also
    --------
    DistributedAIACube.from_files : Create a cube from a list of FITS files
    """
    def __init__(self, maps):
        # TODO: Refactor this to use ndcube instead
        if not all([m.data.shape == maps[0].data.shape for m in maps]):
            raise ValueError('All maps must have same dimensions')
        if not all([m.data.dtype == maps[0].data.dtype for m in maps]):
            raise ValueError('All maps must have same dtype')
        self.maps = maps
        self.time = self._get_time()

    @classmethod
    def from_files(cls, fits_files, **kwargs):
        """
        Create a `DistributedAIACube` object from a list of FITS files

        Parameters
        ----------
        fits_files : `list` or `str`
            Can either be a list of filenames or a glob pattern

        Examples
        --------
        >>> from synthesizAR.analysis import DistributedAIACube
        >>> from sunpy.map import Map
        >>> m1 = Map('/path/to/data/map01.fits') #doctest: +SKIP
        >>> m2 = Map('/path/to/data/map02.fits') #doctest: +SKIP
        # Directly from maps
        >>> c = DistributedAIACube([m1, m2]) #doctest: +SKIP
        # From filenames
        >>> c = DistributedAIACube.from_files(['/path/to/data/map01.fits', '/path/to/data/map02.fits']) #doctest: +SKIP
        # Or from glob pattern
        >>> c = DistributedAIACube.from_files('/path/to/data/map*.fits') #doctest: +SKIP

        See Also
        --------
        dask.bytes.open_files
        """
        openfiles = dask.bytes.open_files(fits_files)
        headers = cls._get_headers(openfiles, **kwargs)
        dtype, shape = cls._get_dtype_and_shape(headers)
        maps = cls._get_maps(openfiles, headers, dtype, shape, **kwargs)
        return cls(maps)

    @staticmethod
    def _get_maps(openfiles, headers, dtype, shape, **kwargs):
        hdu = kwargs.get('hdu', 0)
        verify = kwargs.get('verify', False)
        arrays = [da.from_array(DelayedFITS(f, shape, dtype, hdu=hdu, verify=verify), chunks=shape)
                  for f in openfiles]
        return [sunpy.map.Map(a, h) for a, h in zip(arrays, headers)]

    @staticmethod
    def _get_headers(openfiles, **kwargs):
        client = distributed.get_client()
        futures = client.map(get_header, openfiles, hdu=kwargs.get('hdu', 0))
        return client.gather(futures)

    @staticmethod
    def _get_dtype_and_shape(headers):
        dtypes = [validate_dtype_shape(h) for h in headers]
        if not all([d == dtypes[0] for d in dtypes]):
            raise ValueError('All maps must have same shape and dtype')
        return dtypes[0]

    def _get_time(self,):
        """
        Retrieve time array
        """
        # FIXME: Simulations just store the seconds, should probably fix that
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
        Compute average in time for each pixel
        """
        chunks = kwargs.get('chunks', (self.shape[0], self.shape[1]//10, self.shape[2]//10))
        cube = self.rechunk(chunks)
        # FIXME: should this be a weighted average? How to calculate the weights?
        # FIXME: should we modify any of the metadata before taking an average?
        return sunpy.map.Map(cube.mean(axis=0, dtype=np.float64), self.maps[0].meta.copy())

    def prep(self,):
        """
        Lazily apply a prep operation to all maps and return a new distributed cube object

        See Also
        --------
        sunpy.instr.aia.aia_prep
        """
        raise NotImplementedError('TODO')

    def derotate(self, reference_date):
        """
        Lazily apply a derotation to all maps and return a new distributed cube object

        See Also
        --------
        sunpy.physics.diffrot_map
        """
        raise NotImplementedError('TODO')

    def submap(self, *args, **kwargs):
        """
        Return a new `DistributedAIACube` with lazily-evaluated submap of each map.
        """
        return DistributedAIACube([m.submap(*args, **kwargs) for m in self.maps])


class DistributedAIACollection(object):
    """
    A collection of `~DistributedAIACube` objects for multiple AIA channels

    .. warning:: It is assumed that the data in this container are all aligned, i.e. prepped and derotated
    """

    def __init__(self, *args, **kwargs):
        # TODO: refactor this to use ndcube sequence
        # TODO: Add a check for data being aligned?
        # Check all spatial and time shapes the same
        if not all([a.shape[1:] == args[0].shape[1:] for a in args]):
            raise ValueError('All spatial dimensions must be the same')
        if not all([a.shape[0] == args[0].shape[0] for a in args]):
            # Not an error because may have missing timesteps in observations
            # Will interpolate later to account for this
            warnings.warn('Time dimensions are not all equal length')
        self._cubes = {a.maps[0].meta['wavelnth']: a for a in args}
        self.channels = sorted(list(self._cubes.keys()), key=lambda x: x)

    def __getitem__(self, channel):
        # Index
        if type(channel) is int and channel not in self.channels:
            channel = self.channels[channel]
        # Convert from string
        if type(channel) is str:
            channel = float(channel)
        return self._cubes[channel]


class AIATimelags(DistributedAIACollection):
    """
    Lazily compute cross-correlations and time delays over a `DistributedAIACube`
    """
    @property
    def needs_interpolation(self,):
        """
        Check if observing times for all `DistributedAIACube` are the same
        """
        if not all([c.shape[0] == self[0].shape[0] for c in self]):
            return True
        return ~np.all([u.allclose(c.time, self[0].time) for c in self])

    @property
    def timelags(self):
        time = self._interpolate_time if self.needs_interpolation else self[0].time
        delta_t = np.diff(time.value).cumsum()
        return np.hstack([-delta_t[::-1], np.array([0]), delta_t]) * time.unit
    
    @property
    def _interpolate_time(self,):
        min_t = min([c.time.min() for c in self])
        max_t = max([c.time.max() for c in self])
        n_t = max([c.time.shape[0] for c in self])
        return np.linspace(min_t, max_t, n_t)
    
    def _interpolate(self, time, cube):
        t_interp = self._interpolate_time
        def interp_wrapper(y):
            return interp1d(time, y, axis=0, kind='linear', fill_value='extrapolate')(t_interp)
        return da.map_blocks(interp_wrapper, cube, chunks=t_interp.shape+cube.chunks[1:],
                             dtype=cube.dtype)

    def make_timeseries(self, channel, left_corner, right_corner, **kwargs):
        """
        Return a timeseries for a given channel and spatial selection

        Parameters
        ----------
        channel : `int`, `float`, or `str`
        left_corner : `tuple`
        right_corner : `tuple`
        chunks : `tuple`, optional
        """

        tmp = self[channel].maps[0]
        x_l, y_l = tmp.world_to_pixel(SkyCoord(*left_corner, frame=tmp.coordinate_frame))
        x_u, y_u = tmp.world_to_pixel(SkyCoord(*right_corner, frame=tmp.coordinate_frame))
        x_l, y_l, x_u, y_u = np.round([x_l.value, y_l.value, x_u.value, y_u.value]).astype(np.int)
        chunks = kwargs.get('chunks', (
            self[channel].shape[0], self[channel].shape[1]//10, self[channel].shape[2]//10))
        return (self[channel].rechunk(chunks)[:, y_l:y_u, x_l:x_u].mean(axis=(1, 2)))

    def correlation_1d(self, channel_a, channel_b, left_corner, right_corner, **kwargs):
        """
        Lazily compute cross-correlation between two channels of an average over multiple pixels.
        """

        ts_a = self.make_timeseries(channel_a, left_corner, right_corner, **kwargs)
        ts_b = self.make_timeseries(channel_b, left_corner, right_corner, **kwargs)
        if self.needs_interpolation:
            ts_a = self._interpolate(self[channel_a].time, ts_a)
            ts_b = self._interpolate(self[channel_b].time, ts_b)
        ts_a = (ts_a - ts_a.mean()) / ts_a.std()
        ts_b = (ts_b - ts_b.mean()) / ts_b.std()
        cc = da.fft.irfft(da.fft.rfft(ts_a[::-1], n=self.timelags.shape[0])
                          * da.fft.rfft(ts_b, n=self.timelags.shape[0]), n=self.timelags.shape[0])
        # Normalize by length of timeseries
        return cc / ts_a.shape[0]

    def correlation_2d(self, channel_a, channel_b, **kwargs):
        """
        Lazily compute cross-correlation in each pixel of an AIA map
        """
        # Shape must be the same in spatial direction
        chunks = kwargs.get('chunks', (self[channel_a].shape[1]//10,
                                       self[channel_a].shape[2]//10))
        cube_a = self[channel_a].rechunk(self[channel_a].shape[:1]+chunks)
        cube_b = self[channel_b].rechunk(self[channel_b].shape[:1]+chunks)
        if self.needs_interpolation:
            cube_a = self._interpolate(self[channel_a].time, cube_a)
            cube_b = self._interpolate(self[channel_b].time, cube_b)
        # Reverse the first timeseries
        cube_a = cube_a[::-1, :, :]
        # Normalize by mean and standard deviation
        std_a = cube_a.std(axis=0)
        std_a = da.where(std_a == 0, 1, std_a)
        v_a = (cube_a - cube_a.mean(axis=0)[np.newaxis, :, :]) / std_a[np.newaxis, :, :]
        std_b = cube_b.std(axis=0)
        std_b = da.where(std_b == 0, 1, std_b)
        v_b = (cube_b - cube_b.mean(axis=0)[np.newaxis, :, :]) / std_b[np.newaxis, :, :]
        # FFT of both channels
        fft_a = da.fft.rfft(v_a, axis=0, n=self.timelags.shape[0])
        fft_b = da.fft.rfft(v_b, axis=0, n=self.timelags.shape[0])
        # Inverse of product of FFTS to get cross-correlation (by convolution theorem)
        cc = da.fft.irfft(fft_a * fft_b, axis=0, n=self.timelags.shape[0])
        # Normalize by the length of the timeseries
        return cc / cube_a.shape[0]

    def make_correlation_map(self, channel_a, channel_b, **kwargs):
        """
        Compute map of delay maximum cross-correlation value between two channels in each pixel of
        an AIA map.
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
        client = distributed.get_client()
        max_cc = client.persist(cc[start:stop, :, :].max(axis=0))
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
        Compute map of delay corresponding to maximum cross-correlation value between
        two channels in each pixel of an AIA map.
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
        client = distributed.get_client()
        i_max_cc = client.persist(cc[start:stop, :, :].argmax(axis=0))
        max_timelag = self.timelags[start:stop][i_max_cc]
        meta = self[channel_a].maps[0].meta.copy()
        del meta['instrume']
        del meta['t_obs']
        del meta['wavelnth']
        meta['bunit'] = 's'
        meta['comment'] = f'{channel_a}-{channel_b} timelag'
        plot_settings = {'cmap': 'RdBu_r', 'vmin': self.timelags[start:stop].value.min(),
                         'vmax': self.timelags[start:stop].value.max()}
        plot_settings.update(kwargs.get('plot_settings', {}))
        timelag_map = sunpy.map.GenericMap(max_timelag, meta.copy(),
                                           plot_settings=plot_settings.copy())
        return timelag_map
