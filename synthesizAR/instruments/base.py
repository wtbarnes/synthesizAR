"""
Base class for instrument objects.
"""
import astropy.units as u
import copy
import numpy as np
import pathlib
import tempfile
import zarr

from astropy.coordinates import SkyCoord
from dataclasses import dataclass
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

    @u.quantity_input
    def __init__(self,
                 observing_time: u.s,
                 observer,
                 pad_fov: u.arcsec = None,
                 fov_center=None,
                 fov_width: u.arcsec = None,
                 average_over_los=False):
        self.observer = observer
        self.observing_time = observing_time
        self.pad_fov = (0, 0) * u.arcsec if pad_fov is None else pad_fov
        self.fov_center = fov_center
        self.fov_width = fov_width
        self.average_over_los = average_over_los

    @property
    def observing_time(self) -> u.s:
        return self._observing_time

    @observing_time.setter
    def observing_time(self, value):
        if self.cadence is None or len(value) > 2:
            self._observing_time = value
        else:
            self._observing_time = np.arange(*value.to_value('s'),
                                             self.cadence.to_value('s')) * u.s

    @property
    def cadence(self):
        return None

    @property
    def resolution(self) -> u.arcsec/u.pix:
        return (1, 1) * u.arcsec / u.pix

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

    @property
    def observatory(self):
        return self.name

    @property
    def _expected_unit(self):
        return None

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
        bins, bin_range = self.get_detector_array(coordinates)
        coordinates_centers_projected = coordinates_centers.transform_to(self.projected_frame)
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

            header = self.get_header(channel, coordinates)
            # Build a map for each timestep
            maps[channel.name] = []
            for i, time in enumerate(self.observing_time):
                m = self.integrate_los(
                    time,
                    channel,
                    skeleton,
                    coordinates_centers_projected,
                    bins,
                    bin_range,
                    header,
                    kernels=kernels[i],
                    check_visible=check_visible)
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

    def integrate_los(self, time, channel, skeleton, coordinates_centers, bins, bin_range, header,
                      kernels=None, check_visible=False):
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
        if check_visible:
            visible = coordinates_centers.is_visible()
        else:
            visible = np.ones(kernels.shape)
        # Bin
        blc, trc = bin_range
        hist, _, _ = np.histogram2d(
            coordinates_centers.Tx.value,
            coordinates_centers.Ty.value,
            bins=bins,
            range=((blc.Tx.value, trc.Tx.value), (blc.Ty.value, trc.Ty.value)),
            weights=kernels.to_value(self._expected_unit) * visible,
        )
        # For some quantities, need to average over all components along a given LOS
        if self.average_over_los:
            _hist, _, _ = np.histogram2d(
                coordinates_centers.Tx.value,
                coordinates_centers.Ty.value,
                bins=bins,
                range=((blc.Tx.value, trc.Tx.value), (blc.Ty.value, trc.Ty.value)),
                weights=visible,
            )
            hist /= np.where(_hist == 0, 1, _hist)
        # NOTE: Purposefully using a nonstandard key to record this time as we do not
        # want this to have the implicit consequence of changing the coordinate frame
        # by changing a more standard time key. However, still want to record this
        # information somewhere in the header.
        # FIXME: Figure out a better way to deal with this.
        new_header = copy.deepcopy(header)
        new_header['date_sim'] = (self.observer.obstime + time).isot

        return Map(hist.T, new_header)

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
            reference_pixel=(u.Quantity(bins, 'pix') - 1*u.pix) / 2,  # center of lower left pixel is (0,0)
            scale=self.resolution,
            observatory=self.observatory,
            instrument=self.get_instrument_name(channel),
            telescope=self.telescope,
            detector=self.detector,
            wavelength=channel.channel,
            unit=self._expected_unit,
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
