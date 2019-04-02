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
import sunpy.map
from sunpy.util.metadata import MetaDict

__all__ = ['DistributedAIACube', 'DistributedAIACollection']


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
