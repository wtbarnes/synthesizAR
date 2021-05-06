"""
Base class for instrument objects.
"""
import os
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.util.metadata import MetaDict
from sunpy.coordinates.frames import Helioprojective, HeliographicStonyhurst
from sunpy.map import make_fitswcs_header, Map
import distributed
import zarr

from synthesizAR.util import is_visible

__all__ = ['ChannelBase', 'InstrumentBase']


@dataclass
class ChannelBase:
    telescope_number: int
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
    pad_fov : `~astropy.units.Quantity`, optional
        Two-dimensional array specifying the padding to apply to the field of view of the synthetic
        image in both directions. If None, no padding is applied and the field of view is defined
        by the maximal extent of the loop coordinates in each direction.
    fov_center : `~astropy.coordinates.SkyCoord`, optional
    fov_width : `~astropy.units.Quantity`, optional
    average_over_los : `bool`, optional
    """
    fits_template = MetaDict()

    @u.quantity_input
    def __init__(self,
                 observing_time: u.s,
                 observer, pad_fov=None,
                 fov_center=None,
                 fov_width=None,
                 average_over_los=False):
        self.observing_time = np.arange(*observing_time.to('s').value,
                                        self.cadence.to('s').value)*u.s
        self.observer = observer.transform_to(HeliographicStonyhurst)
        self.pad_fov = (0, 0) * u.arcsec if pad_fov is None else pad_fov
        self.fov_center = fov_center
        self.fov_width = fov_width
        self.average_over_los = average_over_los

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

    def convolve_with_psf(self, data):
        # TODO: do the convolution here!
        return data

    def observe(self, skeleton, save_directory, channels=None, **kwargs):
        """
        Calculate the time dependent intensity for all loops and project them along
        the line-of-sight as defined by the instrument observer.

        Parameters
        ----------

        """
        if channels is None:
            channels = self.channels
        client = distributed.get_client()
        coordinates = skeleton.all_coordinates
        coordinates_centers = skeleton.all_coordinates_centers
        for channel in channels:
            kernels = client.map(self.calculate_intensity_kernel,
                                 skeleton.loops,
                                 channel=channel,
                                 **kwargs)
            kernels_interp = client.map(self.interpolate_to_instrument_time,
                                        kernels,
                                        skeleton.loops,
                                        observing_time=self.observing_time)
            files = client.map(self.write_kernel_to_file,
                               kernels_interp,
                               skeleton.loops,
                               channel=channel,
                               name=self.name)
            # NOTE: block here to avoid pileup of tasks that can overwhelm the scheduler
            distributed.wait(files)
            for i, t in enumerate(self.observing_time):
                m = self.integrate_los(t, channel, skeleton, coordinates, coordinates_centers)
                m = self.convolve_with_psf(m)
                m.save(os.path.join(save_directory, f'm_{channel.name}_t{i}.fits'), overwrite=True)

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
    def interpolate_to_instrument_time(kernel, loop, observing_time):
        """
        Interpolate the intensity kernel from the simulation time to the cadence
        of the instrument for the desired observing window.
        """
        time = loop.time
        if time.shape == (1,):
            if time != observing_time:
                raise ValueError('Model and observing times are not equal for a single model time step.')
            return kernel
        f_t = interp1d(time.to(observing_time.unit).value, kernel.value, axis=0, fill_value='extrapolate')
        return f_t(observing_time.value) * kernel.unit

    def integrate_los(self, time, channel, skeleton, coordinates, coordinates_centers):
        client = distributed.get_client()
        # Get Coordinates
        coords = coordinates_centers.transform_to(self.projected_frame)
        # Compute weights
        i_time = np.where(time == self.observing_time)[0][0]
        widths = np.concatenate([l.field_aligned_coordinate_width for l in skeleton.loops])
        loop_area = np.concatenate([l.cross_sectional_area for l in skeleton.loops])
        root = skeleton.loops[0].zarr_root
        # NOTE: do this outside of the client.map call to make Dask happy
        path = f'{{}}/{self.name}/{channel.name}'
        kernels = np.concatenate(client.gather(client.map(
            lambda l: root[path.format(l.name)][i_time, :],
            skeleton.loops,
        )))
        unit_kernel = u.Unit(
            root[f'{skeleton.loops[0].name}/{self.name}/{channel.name}'].attrs['unit'])
        area_ratio = (loop_area / self.pixel_area).decompose()
        weights = area_ratio * widths * (kernels*unit_kernel)
        visible = is_visible(coords, self.observer)
        # Bin
        bins, (blc, trc) = self.get_detector_array(coordinates)
        hist, _, _ = np.histogram2d(
            coords.Tx.value,
            coords.Ty.value,
            bins=bins,
            range=((blc.Tx.value, trc.Tx.value), (blc.Ty.value, trc.Ty.value)),
            weights=weights.value * visible,
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
        header['bunit'] = weights.unit.decompose().to_string()
        header['date-obs'] = (self.observer.obstime + time).isot

        return Map(hist.T, header)

    def get_header(self, channel, coordinates):
        """
        Create the FITS header for a given channel and set of loop coordinates
        that define the needed FOV.
        """
        bins, bin_range = self.get_detector_array(coordinates)
        header = make_fitswcs_header(
            (bins[1], bins[0]),  # swap order because it expects (row,column)
            bin_range[0],  # align with the lower left corner of the lower left pixel
            reference_pixel=(-0.5, -0.5)*u.pixel,  # center of the lower left pixel is (0,0)
            scale=self.resolution,
            instrument=f'{self.detector}_{channel.telescope_number}',
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
            # NOTE: this is the coordinate of the bottom left corner of the bottom left corner pixel,
            # NOT the coordinate at the center of the pixel!
            bottom_left_corner = SkyCoord(
                Tx=coordinates.Tx.min() - self.pad_fov[0],
                Ty=coordinates.Ty.min() - self.pad_fov[1],
                frame=coordinates.frame
            )
            delta_x = coordinates.Tx.max() + self.pad_fov[0] - bottom_left_corner.Tx
            delta_y = coordinates.Ty.max() + self.pad_fov[1] - bottom_left_corner.Ty
            bins_x = int(np.ceil((delta_x / self.resolution[0]).decompose()).value)
            bins_y = int(np.ceil((delta_y / self.resolution[1]).decompose()).value)
            # Compute right corner after the fact to account for rounding in bin numbers
            # NOTE: this is the coordinate of the top right corner of the top right corner pixel, NOT
            # the coordinate at the center of the pixel!
            top_right_corner = SkyCoord(
                Tx=bottom_left_corner.Tx + self.resolution[0]*bins_x*u.pixel,
                Ty=bottom_left_corner.Ty + self.resolution[1]*bins_y*u.pixel,
                frame=coordinates.frame
            )
        return (bins_x, bins_y), (bottom_left_corner, top_right_corner)
