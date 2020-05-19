"""
Base class for instrument objects.
"""
import os

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


# TODO: some sort of base channel object that all instruments can use by default
# should look something like those in aiapy; use data classes?

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
    assumed_cross_section : `~astropy.units.Quantity`, optional
        Approximation of the loop cross-section. This defines the filling factor.
    pad_fov : `~astropy.units.Quantity`, optional
        Two-dimensional array specifying the padding to apply to the field of view of the synthetic
        image in both directions. If None, no padding is applied and the field of view is defined
        by the maximal extent of the loop coordinates in each direction.
    """
    fits_template = MetaDict()

    @u.quantity_input
    def __init__(self, observing_time: u.s, observer, assumed_cross_section=1e14 * u.cm**2,
                 pad_fov=None):
        self.observing_time = np.arange(observing_time[0].to(u.s).value,
                                        observing_time[1].to(u.s).value,
                                        self.cadence.value)*u.s
        self.observer = observer
        self.assumed_cross_section = assumed_cross_section
        self.pad_fov = (0, 0) * u.arcsec if pad_fov is None else pad_fov

    def calculate_intensity_kernel(self, *args, **kwargs):
        """
        Converts emissivity for a particular transition to counts per detector channel. When writing
        a new instrument class, this method should be overridden.
        """
        raise NotImplementedError('No detect method implemented.')

    def los_velocity(self, v_x, v_y, v_z):
        """
        Compute the LOS velocity for the instrument observer
        """
        # NOTE: transform from HEEQ to HCC with respect to the instrument observer
        obs = self.observer.transform_to(HeliographicStonyhurst)
        Phi_0, B_0 = obs.lon.to(u.radian), obs.lat.to(u.radian)
        v_los = v_z*np.sin(B_0) + v_x*np.cos(B_0)*np.cos(Phi_0) + v_y*np.cos(B_0)*np.sin(Phi_0)
        # NOTE: Negative sign to be consistent with convention v_los > 0 away from observer
        return -v_los

    @property
    def projected_frame(self):
        return Helioprojective(observer=self.observer, obstime=self.observer.obstime)

    @property
    def cross_section_ratio(self):
        """
        Ratio between loop cross-sectional area and pixel area. This essentially defines
        our filling factor.
        """
        w_x, w_y = (1*u.pix * self.resolution).to(u.radian).value * self.observer.radius
        return (self.assumed_cross_section / (w_x * w_y)).decompose()

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
            distributed.wait(files)
            # TODO: add step to save to a file, dont keep in memory
            for i, t in enumerate(self.observing_time):
                m = self.integrate_los(t, channel, skeleton)
                m = self.convolve_with_psf(m)
                m.save(os.path.join(save_directory, f'm_{channel.name}_t{i}.fits'))

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
        f_t = interp1d(time.to(observing_time.unit).value, kernel.value, axis=0)
        return f_t(observing_time.value) * kernel.unit

    def integrate_los(self, time, channel, skeleton):
        # Get Coordinates
        coords = skeleton.all_coordinates_centers.transform_to(self.projected_frame)
        # Compute weights
        i_time = np.where(time == self.observing_time)[0][0]
        widths = np.concatenate([l.field_aligned_coordinate_width for l in skeleton.loops])
        root = zarr.open(skeleton.loops[0].model_results_filename, 'r')
        kernels = np.concatenate([root[f'{l.name}/{self.name}/{channel.name}'][i_time, :]
                                  for l in skeleton.loops])
        unit_kernel = u.Unit(
            root[f'{skeleton.loops[0].name}/{self.name}/{channel.name}'].attrs['unit'])
        weights = self.cross_section_ratio * widths * (kernels*unit_kernel)
        visible = is_visible(coords, self.observer)
        # Bin
        bins, (blc, trc) = self.get_detector_array(skeleton.all_coordinates)
        hist, _, _ = np.histogram2d(
            coords.Tx.value,
            coords.Ty.value,
            bins=bins,
            range=((blc.Tx.value, trc.Tx.value), (blc.Ty.value, trc.Ty.value)),
            weights=weights.value * visible,
        )
        header = self.get_header(channel, skeleton.all_coordinates)
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
        coordinates = coordinates.transform_to(self.projected_frame)
        # NOTE: this is the coordinate of the bottom left corner of the bottom left corner pixel,
        # NOT the coordinate at the center of the pixel!
        bottom_left_corner = SkyCoord(
            Tx=coordinates.Tx.min() - self.pad_fov[0],
            Ty=coordinates.Ty.min() - self.pad_fov[1],
            frame=coordinates.frame
        )
        bins_x = int(np.ceil((coordinates.Tx.max() + self.pad_fov[0] - bottom_left_corner.Tx) / self.resolution[0]).value)
        bins_y = int(np.ceil((coordinates.Ty.max() + self.pad_fov[1] - bottom_left_corner.Ty) / self.resolution[1]).value)
        # Compute right corner after the fact to account for rounding in bin numbers
        # NOTE: this is the coordinate of the top right corner of the top right corner pixel, NOT
        # the coordinate at the center of the pixel!
        top_right_corner = SkyCoord(
            Tx=bottom_left_corner.Tx + self.resolution[0]*bins_x*u.pixel,
            Ty=bottom_left_corner.Ty + self.resolution[1]*bins_y*u.pixel,
            frame=coordinates.frame
        )
        return (bins_x, bins_y), (bottom_left_corner, top_right_corner)
