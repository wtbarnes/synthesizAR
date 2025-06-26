"""
Base class for instrument objects.
"""
import astropy.units as u
import astropy.wcs
import copy
import numpy as np
import pathlib
import tempfile
import zarr

from astropy.coordinates import SkyCoord
from dataclasses import dataclass
from functools import cached_property
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from sunpy.coordinates import HeliographicStonyhurst, Helioprojective
from sunpy.coordinates.utils import solar_angle_equivalency
from sunpy.map import make_fitswcs_header, Map

from synthesizAR.util import find_minimum_fov
from synthesizAR.util.decorators import return_quantity_as_tuple

__all__ = ['ChannelBase', 'InstrumentBase']


@dataclass
class ChannelBase:
    name: str = None
    channel: u.Quantity = None


class InstrumentBase:
    """
    Base class for instruments. This object is not meant to be instantiated directly. Instead,
    specific instruments should subclass this base object and implement a
    `calculate_intensity_kernel` method for that specific instrument.

    Parameters
    ----------
    observing_time : `~astropy.units.Quantity`
        If cadence is also provided and this has a length of 2, this is interpreted as
        the start and end time of the observation period and an observing time is
        constructed based on this interval and the cadence. Otherwise, this is interpreted
        as the times at which the observations should be forward modeled.
    observer : `~astropy.coordinates.SkyCoord`
        Coordinate of the observing instrument
    resolution : `~astropy.units.Quantity`
    cadence : `~astropy.units.Quantity`, optional
        If specified, this is used to construct the observing time.
    pad_fov : `~astropy.units.Quantity`, optional
        Two-dimensional array specifying the padding to apply to the field of view of the synthetic
        image in both directions in pixel space. If None, no padding is applied and the field of
        view is defined by the maximal extent of the loop coordinates in each direction.
        Note that if ``fov_center`` and ``fov_width`` are specified, this is ignored.
    fov_center : `~astropy.coordinates.SkyCoord`, optional
        Reference coordinate coinciding with the center of the field of view.
        For this to have an effect, must also specify ``fov_width``.
    fov_width : `~astropy.units.Quantity`, optional
        Angular extent of the field of the view.
        For this to have an effect, must also specify ``fov_center``.
    average_over_los : `bool`, optional
        Set to true for non-volumetric quantities
    """

    @u.quantity_input
    def __init__(self,
                 observing_time: u.s,
                 observer,
                 resolution: u.Unit('arcsec/pix'),
                 cadence: u.s = None,
                 pad_fov: u.pixel = None,
                 fov_center = None,
                 fov_width: u.arcsec = None,
                 rotation_angle: u.deg = None,
                 average_over_los=False):
        self.observer = observer
        self.cadence = cadence
        self.observing_time = observing_time
        self.resolution = resolution
        self.pad_fov = pad_fov
        self.fov_center = fov_center
        self.fov_width = fov_width
        self.rotation_angle = rotation_angle
        self.average_over_los = average_over_los

    @property
    def observing_time(self) -> u.s:
        return self._observing_time

    @observing_time.setter
    def observing_time(self, value):
        if self.cadence is not None and len(value) == 2:
            self._observing_time = np.arange(*value.to_value('s'),
                                             self.cadence.to_value('s')) * u.s
        else:
            self._observing_time = value

    @property
    def cadence(self):
        return self._cadence

    @cadence.setter
    def cadence(self, value):
        self._cadence = value

    @property
    def resolution(self) -> u.arcsec/u.pix:
        return self._resolution

    @resolution.setter
    def resolution(self, value):
        self._resolution = value

    @property
    def rotation_angle(self) -> u.deg:
        return self._rotation_angle

    @rotation_angle.setter
    def rotation_angle(self, value):
        self._rotation_angle = value

    @property
    def observer(self):
        return self._observer

    @observer.setter
    def observer(self, value):
        self._observer = value.transform_to(HeliographicStonyhurst)

    @property
    def pad_fov(self) -> u.pixel:
        return self._pad_fov

    @pad_fov.setter
    def pad_fov(self, value):
        if value is None:
            value = [0, 0] * u.pixel
        self._pad_fov = value

    @property
    def telescope(self):
        return self.name

    @property
    def detector(self):
        return self.name

    @property
    def observatory(self):
        return self.name

    @property
    def _expected_unit(self):
        raise NotImplementedError

    @property
    def channels(self):
        raise NotImplementedError

    def get_instrument_name(self, channel):
        return self.name

    def calculate_intensity_kernel(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def projected_frame(self):
        return Helioprojective(observer=self.observer)

    @cached_property
    @u.quantity_input
    def pixel_area(self) -> u.cm**2:
        """
        Cartesian area on the surface of the Sun covered by a single pixel.
        """
        sa_equiv = solar_angle_equivalency(self.observer)
        res = (1*u.pix * self.resolution).to('cm', equivalencies=sa_equiv)
        return res[0] * res[1]

    def convolve_with_psf(self, smap, channel):
        """
        Perform a simple convolution with a Gaussian kernel
        """
        w = getattr(channel, 'psf_width', (0, 0)*u.pix)
        # PSF width is specified in order (x-like, y-like) but
        # gaussian_filter expects array index ordering
        w = w.to_value('pixel')[::-1]
        return smap._new_instance(gaussian_filter(smap.data, w), smap.meta)

    def observe(self, skeleton, save_directory=None, channels=None, **kwargs):
        """
        Calculate the time dependent intensity for all loops and project them along
        the line-of-sight as defined by the instrument observer.

        Parameters
        ----------
        skeleton : `~synthesizAR.Skeleton`
        save_directory : `str` or path-like
        """
        if channels is None:
            channels = self.channels
        try:
            import distributed
            client = distributed.get_client()
        except (ImportError, ValueError):
            client = None
        ref_coord, n_pixels = self.get_fov(skeleton.all_coordinates)
        wcs = astropy.wcs.WCS(header=self.get_header(ref_coord, n_pixels))
        coordinates_centers = skeleton.all_coordinates_centers
        pixel_locations = wcs.world_to_pixel(coordinates_centers)
        visibilities = coordinates_centers.transform_to(self.projected_frame).is_visible()
        maps = {}
        for channel in channels:
            # Compute intensity as a function of time and field-aligned coordinate
            if client:
                # Parallel
                kernel_futures = client.map(self.calculate_intensity_kernel,
                                            skeleton.strands,
                                            channel=channel,
                                            **kwargs)
                kernel_interp_futures = client.map(self.interpolate_to_instrument_time,
                                                   kernel_futures,
                                                   skeleton.strands,
                                                   observing_time=(self.observing_time.value, self.observing_time.unit.to_string()))
            else:
                # Serial
                kernels_interp = []
                for l in skeleton.strands:
                    k = self.calculate_intensity_kernel(l, channel=channel, **kwargs)
                    k = self.interpolate_to_instrument_time(
                        k, l, observing_time=(self.observing_time.value, self.observing_time.unit.to_string()),
                    )
                    kernels_interp.append(k)

            if kwargs.get('save_kernels_to_disk', False):
                with tempfile.TemporaryDirectory() as tmpdir:
                    self._make_stacked_kernel_array(tmpdir, skeleton.strands, channel)
                    indices = self._find_loop_array_bounds(skeleton.strands)
                    if client:
                        files = client.map(self.write_kernel_to_file,
                                           kernel_interp_futures,
                                           skeleton.strands,
                                           indices,
                                           channel=channel,
                                           name=self.name,
                                           tmp_store=tmpdir)
                        # NOTE: block here to avoid pileup of tasks that can overwhelm the scheduler
                        distributed.wait(files)
                    else:
                        for k, l, i in zip(kernels_interp, skeleton.strands, indices):
                            self.write_kernel_to_file(k, l, i, channel, self.name, tmpdir)
                    self._rechunk_stacked_kernels(tmpdir, skeleton.strands[0].model_results_filename, channel)
                    kernels = self.observing_time.shape[0]*[None]  # placeholder so we know to read from a file
            else:
                # NOTE: this can really blow up your memory if you are not careful
                if client:
                    kernels_interp = client.gather(kernel_interp_futures)
                kernels = np.concatenate([u.Quantity(*k) for k in kernels_interp], axis=1)

            header = self.get_header(ref_coord, n_pixels, channel=channel)
            # Build a map for each timestep
            maps[channel.name] = []
            for i, time in enumerate(self.observing_time):
                m = self.integrate_los(
                    time,
                    channel,
                    skeleton,
                    pixel_locations,
                    n_pixels,
                    visibilities,
                    header,
                    kernels=kernels[i])
                m = self.convolve_with_psf(m, channel)
                if save_directory is None:
                    maps[channel.name].append(m)
                else:
                    fname = pathlib.Path(save_directory) / f'm_{channel.name}_t{i}.fits'
                    m.save(fname, overwrite=True)
                    maps[channel.name].append(fname)
        return maps

    @staticmethod
    def write_kernel_to_file(kernel, loop, indices, channel, name, tmp_store):
        # NOTE: remove this once https://github.com/dask/distributed/issues/6808 is fixed
        kernel = u.Quantity(*kernel)

        # Save to individual loop dataset
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
        # Map into stacked array
        tmp_root = zarr.open(tmp_store, 'a')
        ds_stacked = tmp_root[f'{name}/{channel.name}_stacked_kernels']
        ds_stacked[:, indices[0]:indices[1]] = kernel.value
        ds_stacked.attrs['unit'] = kernel.unit.to_string()

    def _make_stacked_kernel_array(self, store, loops, channel):
        """
        If it does not already exist, create the stacked array for all
        kernels for each loop
        """
        root = zarr.open(store, 'a')
        if f'{self.name}/{channel.name}_stacked_kernels' not in root:
            n_space = sum([l.electron_temperature.shape[1] for l in loops])
            shape = self.observing_time.shape + (n_space,)
            root.create_dataset(
                f'{self.name}/{channel.name}_stacked_kernels',
                shape=shape,
                chunks=(shape[0], n_space//len(loops)),
                overwrite=True,
            )

    def _rechunk_stacked_kernels(self, tmp_store, final_store, channel):
        """
        Rechunk the stacked kernels array. This is necessary because our write pattern is in chunks
        at all time steps associated with a single loop, but our read pattern is a single time step
        for all loops.
        """
        # NOTE: for large stacked kernel arrays, this may not be possible because this requires
        # reading the whole array into memory. See this section of the Zarr docs:
        # https://zarr.readthedocs.io/en/stable/tutorial.html#changing-chunk-shapes-rechunking
        tmp_root = zarr.open(tmp_store, 'r')
        tmp_ds = tmp_root[f'{self.name}/{channel.name}_stacked_kernels']
        tmp = tmp_ds[...]
        final_root = zarr.open(final_store, 'a')
        ds = final_root.create_dataset(
            f'{self.name}/{channel.name}_stacked_kernels',
            data=tmp,
            chunks=(1, tmp.shape[1]),
            overwrite=True,
        )
        ds.attrs['unit'] = tmp_ds.attrs['unit']

    def _find_loop_array_bounds(self, loops):
        """
        This finds the indices for where each loop maps into the
        stacked kernel array
        """
        root = zarr.open(loops[0].model_results_filename, 'a')
        index_running = 0
        index_bounds = []
        for loop in loops:
            kernel = root[f'{loop.name}/electron_temperature']
            index_bounds.append((index_running, index_running+kernel.shape[1]))
            index_running += kernel.shape[1]
        return index_bounds

    @staticmethod
    @return_quantity_as_tuple
    def interpolate_to_instrument_time(kernel, loop, observing_time, axis=0):
        """
        Interpolate the intensity kernel from the simulation time to the cadence
        of the instrument for the desired observing window.
        """
        # NOTE: remove this once https://github.com/dask/distributed/issues/6808 is fixed
        observing_time = u.Quantity(*observing_time)
        kernel_value, kernel_unit = kernel

        time = loop.time
        if time.shape == (1,):
            if time != observing_time:
                raise ValueError('Model and observing times are not equal for a single model time step.')
            return u.Quantity(*kernel)
        f_t = interp1d(time.to(observing_time.unit).value,
                       kernel_value,
                       axis=axis,
                       fill_value='extrapolate')
        kernel_interp = u.Quantity(f_t(observing_time.value), kernel_unit)
        return kernel_interp

    def integrate_los(self,
                      time,
                      channel,
                      skeleton,
                      pixel_locations,
                      n_pixels,
                      visibilities,
                      header,
                      kernels=None):
        # Compute weights
        if kernels is None:
            i_time = np.where(time == self.observing_time)[0][0]
            root = skeleton.strands[0].zarr_root
            ds = root[f'{self.name}/{channel.name}_stacked_kernels']
            kernels = u.Quantity(ds[i_time, :], ds.attrs['unit'])
        # If a volumetric quantity, integrate over the cell and normalize by pixel area.
        # For some quantities (e.g. temperature, velocity), we just want to know the
        # average along the LOS
        if not self.average_over_los:
            kernels *= (skeleton.all_cross_sectional_areas / self.pixel_area).decompose() * skeleton.all_widths
        # Bin
        # NOTE: Bin order is (y,x) or (row, column)
        bins = n_pixels.to_value('pixel').astype(int)
        bin_edges = (np.linspace(-0.5, bins[1]-0.5, bins[1]+1),
                     np.linspace(-0.5, bins[0]-0.5, bins[0]+1))
        hist, _, _ = np.histogram2d(
            pixel_locations[1],
            pixel_locations[0],
            bins=bin_edges,
            weights=kernels.to_value(self._expected_unit) * visibilities,
        )
        # For some quantities, need to average over all components along a given LOS
        if self.average_over_los:
            _hist, _, _ = np.histogram2d(
                pixel_locations[1],
                pixel_locations[0],
                bins=bin_edges,
                weights=visibilities,
            )
            hist /= np.where(_hist == 0, 1, _hist)
        # NOTE: Purposefully using a nonstandard key to record this time as we do not
        # want this to have the implicit consequence of changing the coordinate frame
        # by changing a more standard time key. However, still want to record this
        # information somewhere in the header.
        # FIXME: Figure out a better way to deal with this.
        new_header = copy.deepcopy(header)
        new_header['date_sim'] = (self.observer.obstime + time).isot

        return Map(hist, new_header)

    def get_header(self, ref_coord, n_pixels: u.pixel, channel=None):
        """
        Create the FITS header for a given channel.

        Parameters
        ----------
        ref_coord: `~astropy.coordinates.SkyCoord`
            Reference coordinate coincident with the center of the field
            of view
        n_pixels: `~astropy.units.Quantity`
            Pixel extent in the x (horizontal) and y (vertical) direction
        channel:  `ChannelBase`, optional
        """
        # NOTE: channel is a kwarg so that a WCS can be computed without specifying
        # a channel as these keywords do not affect the WCS
        if channel is None:
            instrument = None
            wavelength = None
        else:
            instrument = self.get_instrument_name(channel)
            wavelength = channel.channel
        header = make_fitswcs_header(
            tuple(n_pixels[::-1].to_value('pixel')),  # swap order because it expects (row,column)
            ref_coord,
            reference_pixel=(n_pixels - 1*u.pix) / 2,  # center of lower left pixel is (0,0)
            scale=self.resolution,
            rotation_angle=self.rotation_angle,
            observatory=self.observatory,
            instrument=instrument,
            telescope=self.telescope,
            detector=self.detector,
            wavelength=wavelength,
            unit=self._expected_unit,
        )
        return header

    def get_fov(self, coordinates):
        """
        Return the coordinate at the center of the FOV and the width in pixels.
        """
        if self.fov_center is not None and self.fov_width is not None:
            center = self.fov_center.transform_to(self.projected_frame)
            n_pixels = (self.fov_width / self.resolution).decompose().to('pixel')
        else:
            # If not specified, derive FOV from loop coordinates
            coordinates = coordinates.transform_to(self.projected_frame)
            bottom_left_corner, top_right_corner = find_minimum_fov(coordinates)
            delta_x = top_right_corner.Tx - bottom_left_corner.Tx
            delta_y = top_right_corner.Ty - bottom_left_corner.Ty
            center = SkyCoord(Tx=bottom_left_corner.Tx+delta_x/2,
                              Ty=bottom_left_corner.Ty+delta_y/2,
                              frame=bottom_left_corner.frame)
            pixels_x = int(np.ceil((delta_x / self.resolution[0]).decompose()).value)
            pixels_y = int(np.ceil((delta_y / self.resolution[1]).decompose()).value)
            n_pixels = u.Quantity([pixels_x, pixels_y], 'pixel')
            n_pixels += self.pad_fov
        return center, n_pixels
