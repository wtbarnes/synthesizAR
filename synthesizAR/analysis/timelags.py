"""
Objects and functions for computing timelags from AIA data
"""
from scipy.interpolate import interp1d
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import dask.array as da
from sunpy.map import GenericMap

from .aia import DistributedAIACollection

__all__ = ['AIATimelags', 'cross_correlation']


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

    def _skycoords_to_indices(self, channel, left_corner, right_corner):
        """
        Convert ~astropy.coordinates.SkyCoord to an array index
        """
        tmp = self[channel].maps[0]
        x_l, y_l = tmp.world_to_pixel(SkyCoord(*left_corner, frame=tmp.coordinate_frame))
        x_u, y_u = tmp.world_to_pixel(SkyCoord(*right_corner, frame=tmp.coordinate_frame))
        x_l, y_l, x_u, y_u = np.round([x_l.value, y_l.value, x_u.value, y_u.value]).astype(np.int)
        return xl, y_l, x_u, y_u

    def get_timeseries(self, channel, left_corner, right_corner, **kwargs):
        """
        Return a timeseries for a given channel and spatial selection

        Parameters
        ----------
        channel : `int`, `float`, or `str`
        left_corner : `tuple`
        right_corner : `tuple`
        chunks : `tuple`, optional
        """
        x_l, y_l, x_u, y_u = self._skycoords_to_indices(channel, left_corner, right_corner)
        chunks = kwargs.get('chunks', (
            self[channel].shape[0], self[channel].shape[1]//10, self[channel].shape[2]//10))
        return (self[channel].rechunk(chunks)[:, y_l:y_u, x_l:x_u].mean(axis=(1, 2)))

    def cross_correlation(self, channel_a, channel_b, **kwargs):
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

    def peak_cross_correlation_map(self, channel_a, channel_b, **kwargs):
        """
        Construct map of peak cross-correlation between two channels in each pixel of
        an AIA map.
        """
        cc = self.cross_correlation(channel_a, channel_b, **kwargs)
        bounds = kwargs.get('timelag_bounds', None)
        if bounds is not None:
            indices, = np.where(np.logical_and(self.timelags >= bounds[0],
                                               self.timelags <= bounds[1]))
            start = indices[0]
            stop = indices[-1] + 1
        else:
            start = 0
            stop = self.timelags.shape[0] + 1
        max_cc = cc[start:stop, :, :].max(axis=0)
        meta = self[channel_a].maps[0].meta.copy()
        del meta['instrume']
        del meta['t_obs']
        del meta['wavelnth']
        meta['bunit'] = ''
        meta['comment'] = f'{channel_a}-{channel_b} cross-correlation'
        plot_settings = {'cmap': 'plasma'}
        plot_settings.update(kwargs.get('plot_settings', {}))
        correlation_map = GenericMap(max_cc, meta, plot_settings=plot_settings)

        return correlation_map

    def timelag_map(self, channel_a, channel_b, **kwargs):
        """
        Construct map of timelag values that maximize the cross-correlation between
        two channels in each pixel of an AIA map.
        """
        cc = self.cross_correlation(channel_a, channel_b, **kwargs)
        bounds = kwargs.get('timelag_bounds', None)
        if bounds is not None:
            indices, = np.where(np.logical_and(self.timelags >= bounds[0],
                                               self.timelags <= bounds[1]))
            start = indices[0]
            stop = indices[-1] + 1
        else:
            start = 0
            stop = self.timelags.shape[0] + 1
        i_max_cc = cc[start:stop, :, :].argmax(axis=0)
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
        timelag_map = GenericMap(max_timelag, meta.copy(), plot_settings=plot_settings.copy())
        return timelag_map


@u.quantity_input
def cross_correlation(time: u.s, ts_a, ts_b,):
    """
    Given two timeseries `ts_a` and `ts_b`, compute the cross-correlation as
    a function of temporal offset.

    Parameters
    ----------
    time : `~astropy.units.Quantity`
    ts_a : array-like
    ts_b : array-like

    Returns
    -------
    timelag : `~astropy.units.Quantity`
        Temporal offset between timeseries
    cc : array-like
        Cross-correlation as a function of `timelag`
    """
    if time.shape != ts_a.shape or time.shape != ts_b.shape:
        raise ValueError('Timeseries and time arrays must be of equal length')
    ts_a = (ts_a - ts_a.mean()) / ts_a.std()
    ts_b = (ts_b - ts_b.mean()) / ts_b.std()
    delta_t = np.diff(time.value).cumsum()
    timelags = u.Quantity(np.hstack([-delta_t[::-1], np.array([0]), delta_t]), time.unit)
    n = timelags.shape[0]
    cc = np.fft.irfft(np.fft.rfft(ts_a[::-1], n=n) * np.fft.rfft(ts_b, n=n), n=n)
    # Normalize by length of timeseries
    return timelags, cc / time.shape[0]
