"""
Base class for instrument objects.
"""
import os
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.util.metadata import MetaDict
from sunpy.coordinates.frames import Helioprojective, HeliographicStonyhurst
from sunpy.map import make_fitswcs_header, Map
import zarr

from synthesizAR.util import is_visible, find_minimum_fov

__all__ = ['ChannelBase', 'InstrumentBase']


@dataclass
class ChannelBase:
    channel: u.Quantity
    name: str


class InstrumentBase(object):
    """
    Base class for instruments. This object is not meant to be instantiated directly. Instead,
    specific instruments should subclass this base object and implement a `calculate_intensity_kernel`
    method for that specific instrument.

    Parameters
    ----------
    observing_time : `~astropy.units.Quantity`
        Tuple of start and end observing times
    observer_coordinate : `~astropy.coordinates.SkyCoord`
        Coordinate of the observing instrument
    cadence : `~astropy.units.Quantity`
    resolution : `~astropy.units.Quantity`
    pad_fov : `~astropy.units.Quantity`, optional
        Two-dimensional array specifying the padding to apply to the field of view of the synthetic
        image in both directions. If None, no padding is applied and the field of view is defined
        by the maximal extent of the loop coordinates in each direction.
    fov_center : `~astropy.coordinates.SkyCoord`, optional
    fov_width : `~astropy.units.Quantity`, optional
    average_over_los : `bool`, optional
    """
    fits_template = MetaDict()

    @u.quantity_input(observing_time=u.s,
                      cadence=u.s,
                      pad_fov=u.arcsec,
                      fov_width=u.arcsec)
    def __init__(self,
                 observing_time: u.s,
                 observer,
                 resolution,
                 cadence=None,
                 pad_fov=None,
                 fov_center=None,
                 fov_width=None,
                 average_over_los=False):
        self.observer = observer
        self.cadence = cadence
        self._observing_time = observing_time
        self.resolution = resolution
        self.pad_fov = (0, 0) * u.arcsec if pad_fov is None else pad_fov
        self.fov_center = fov_center
        self.fov_width = fov_width
        self.average_over_los = average_over_los

    @property
    def observing_time(self) -> u.s:
        if self.cadence is None or len(self._observing_time) > 2:
            return self._observing_time
        else:
            return np.arange(*self._observing_time.to('s').value,
                             self.cadence.to('s').value) * u.s

    @property
    def observer(self):
        return self._observer.transform_to(HeliographicStonyhurst)

    @observer.setter
    def observer(self, value):
        self._observer = value

    @property
    def telescope(self):
        return self.name

    @property
    def detector(self):
        return self.name

    def get_instrument_name(self, channel):
        return self.name

    def calculate_intensity_kernel(self, *args, **kwargs):
        """
        Converts emissivity for a particular transition to counts per detector channel. When writing
        a new instrument class, this method should be overridden.
        """
        raise NotImplementedError('No detect method implemented.')

    @property
    def projected_frame(self):
        return Helioprojective(observer=self.observer, obstime=self.observer.obstime)

    @property
    @u.quantity_input
    def pixel_area(self) -> u.cm**2:
        """
        Pixel area
        """
        w_x, w_y = (1*u.pix * self.resolution).to(u.radian).value * self.observer.radius
        return w_x * w_y

    def convolve_with_psf(self, smap, channel):
        """
        Perform a simple convolution with a Gaussian kernel
        """
        # Specify in order x, y (axis 1, axis 2)
        w = getattr(channel, 'gaussian_width', (1,1)*u.pixel)
        # gaussian filter takes order (row, column)
        return smap._new_instance(gaussian_filter(smap.data, w.value[::-1]), smap.meta)

    def observe(self, skeleton, save_directory=None, channels=None, **kwargs):
        """
        Calculate the time dependent intensity for all loops and project them along
        the line-of-sight as defined by the instrument observer.

        Parameters
        ----------
        skeleton : `~synthesizAR.Skeleton`
        save_directory : `str` or path-like
        """
        check_visible = kwargs.pop('check_visible', False)
        if channels is None:
            channels = self.channels
        try:
            import distributed
            client = distributed.get_client()
        except (ImportError, ValueError):
            client = None
        coordinates = skeleton.all_coordinates
        coordinates_centers = skeleton.all_coordinates_centers
        maps = {}
        for channel in channels:
            # Compute intensity as a function of time and field-aligned coordinate
            if client:
                # Parallel
                kernel_futures = client.map(self.calculate_intensity_kernel,
                                            skeleton.loops,
                                            channel=channel,
                                            **kwargs)
                kernel_interp_futures = client.map(self.interpolate_to_instrument_time,
                                                   kernel_futures,
                                                   skeleton.loops,
                                                   observing_time=self.observing_time)
            else:
                # Seriel
                kernels_interp = []
                for l in skeleton.loops:
                    k = self.calculate_intensity_kernel(l, channel=channel, **kwargs)
                    k = self.interpolate_to_instrument_time(
                        k, l, observing_time=self.observing_time,
                    )
                    kernels_interp.append(k)

            if kwargs.get('save_kernels_to_disk', False):
                files = client.map(self.write_kernel_to_file,
                                   kernel_interp_futures,
                                   skeleton.loops,
                                   channel=channel,
                                   name=self.name)
                # NOTE: block here to avoid pileup of tasks that can overwhelm the scheduler
                distributed.wait(files)
                kernels = self.observing_time.shape[0]*[None]  # placeholder so we know to read from a file
            else:
                # NOTE: this can really blow up your memory if you are not careful
                kernels = np.concatenate(client.gather(kernel_interp_futures) if client else kernels_interp, axis=1)

            maps[channel.name] = []
            for i, t in enumerate(self.observing_time):
                m = self.integrate_los(t, channel, skeleton, coordinates, coordinates_centers,
                                       kernels=kernels[i], check_visible=check_visible)
                m = self.convolve_with_psf(m, channel)
                if save_directory is None:
                    maps[channel.name].append(m)
                else:
                    fname = os.path.join(save_directory, f'm_{channel.name}_t{i}.fits')
                    m.save(fname, overwrite=True)
                    maps[channel.name].append(fname)
        return maps

    @staticmethod
    def write_kernel_to_file(kernel, loop, channel, name):
        root = zarr.open(loop.model_results_filename, 'a')
        if name not in root[loop.name]:
            root[loop.name].create_group(name)
        ds = root[f'{loop.name}/{name}'].create_dataset(
            channel.name,
            data=kernel.value,
            chunks=(None,)+kernel.shape[:1],
            overwrite=True,
        )
        ds.attrs['unit'] = kernel.unit.to_string()

    @staticmethod
    def interpolate_to_instrument_time(kernel, loop, observing_time, axis=0):
        """
        Interpolate the intensity kernel from the simulation time to the cadence
        of the instrument for the desired observing window.
        """
        time = loop.time
        if time.shape == (1,):
            if time != observing_time:
                raise ValueError('Model and observing times are not equal for a single model time step.')
            return kernel
        f_t = interp1d(time.to(observing_time.unit).value,
                       kernel.value,
                       axis=axis,
                       fill_value='extrapolate')
        kernel_interp = u.Quantity(f_t(observing_time.value), kernel.unit)
        return kernel_interp

    def integrate_los(self, time, channel, skeleton, coordinates, coordinates_centers, kernels=None, check_visible=False):
        # Get Coordinates
        coords = coordinates_centers.transform_to(self.projected_frame)
        # Compute weights
        widths = np.concatenate([l.field_aligned_coordinate_width for l in skeleton.loops])
        loop_area = np.concatenate([l.cross_sectional_area_center for l in skeleton.loops])
        if kernels is None:
            import distributed
            i_time = np.where(time == self.observing_time)[0][0]
            client = distributed.get_client()
            root = skeleton.loops[0].zarr_root
            # NOTE: do this outside of the client.map call to make Dask happy
            path = f'{{}}/{self.name}/{channel.name}'
            kernels = np.concatenate(client.gather(client.map(
                lambda l: root[path.format(l.name)][i_time, :],
                skeleton.loops,
            )))
            unit_kernel = u.Unit(
                root[f'{skeleton.loops[0].name}/{self.name}/{channel.name}'].attrs['unit'])
            kernels = kernels * unit_kernel
        # If a volumetric quantity, integrate over the cell and normalize by pixel area.
        # For some quantities (e.g. temperature, velocity), we just want to know the
        # average along the LOS
        if not self.average_over_los:
            kernels *= (loop_area / self.pixel_area).decompose() * widths
        if check_visible:
            visible = is_visible(coords, self.observer)
        else:
            visible = np.ones(kernels.shape)
        # Bin
        bins, (blc, trc) = self.get_detector_array(coordinates)
        hist, _, _ = np.histogram2d(
            coords.Tx.value,
            coords.Ty.value,
            bins=bins,
            range=((blc.Tx.value, trc.Tx.value), (blc.Ty.value, trc.Ty.value)),
            weights=kernels.value * visible,
        )
        # For some quantities, need to average over all components along a given LOS
        if self.average_over_los:
            _hist, _, _ = np.histogram2d(
                coords.Tx.value,
                coords.Ty.value,
                bins=bins,
                range=((blc.Tx.value, trc.Tx.value), (blc.Ty.value, trc.Ty.value)),
                weights=visible,
            )
            hist /= np.where(_hist == 0, 1, _hist)
        header = self.get_header(channel, coordinates)
        header['bunit'] = kernels.unit.to_string()
        # FIXME: not sure we really want to do this...this changes our coordinate
        # frame but maybe we don't want it to change!
        header['date-obs'] = (self.observer.obstime + time).isot

        return Map(hist.T, header)

    def get_header(self, channel, coordinates):
        """
        Create the FITS header for a given channel and set of loop coordinates
        that define the needed FOV.
        """
        bins, bin_range = self.get_detector_array(coordinates)
        center = SkyCoord(Tx=(bin_range[1].Tx + bin_range[0].Tx)/2,
                          Ty=(bin_range[1].Ty + bin_range[0].Ty)/2,
                          frame=bin_range[0].frame)
        # FIXME: reference_pixel should be center of the frame in the pixel
        # coordinate system of the image.
        header = make_fitswcs_header(
            (bins[1], bins[0]),  # swap order because it expects (row,column)
            center,
            reference_pixel=(u.Quantity(bins, 'pix') - 1*u.pix) / 2,  # center of the lower left pixel is (0,0)
            scale=self.resolution,
            instrument=self.get_instrument_name(channel),  # sometimes this depends on the channel
            telescope=self.telescope,
            wavelength=channel.channel,
        )
        return header

    def get_detector_array(self, coordinates):
        """
        Calculate the number of pixels in the detector FOV and the physical coordinates of the
        bottom left and top right corners.
        """
        if self.fov_center is not None and self.fov_width is not None:
            center = self.fov_center.transform_to(self.projected_frame)
            bins_x = int(np.ceil((self.fov_width[0] / self.resolution[0]).decompose()).value)
            bins_y = int(np.ceil((self.fov_width[1] / self.resolution[1]).decompose()).value)
            bottom_left_corner = SkyCoord(
                Tx=center.Tx - self.fov_width[0]/2,
                Ty=center.Ty - self.fov_width[1]/2,
                frame=center.frame,
            )
            top_right_corner = SkyCoord(
                Tx=bottom_left_corner.Tx + self.fov_width[0],
                Ty=bottom_left_corner.Ty + self.fov_width[1],
                frame=bottom_left_corner.frame
            )
        else:
            # If not specified, derive FOV from loop coordinates
            coordinates = coordinates.transform_to(self.projected_frame)
            bottom_left_corner, top_right_corner = find_minimum_fov(
                coordinates, padding=self.pad_fov,
            )
            delta_x = top_right_corner.Tx - bottom_left_corner.Tx
            delta_y = top_right_corner.Ty - bottom_left_corner.Ty
            bins_x = int(np.ceil((delta_x / self.resolution[0]).decompose()).value)
            bins_y = int(np.ceil((delta_y / self.resolution[1]).decompose()).value)
        return (bins_x, bins_y), (bottom_left_corner, top_right_corner)
