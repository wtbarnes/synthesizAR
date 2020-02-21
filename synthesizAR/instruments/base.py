"""
Base class for instrument objects.
"""
import numpy as np
from scipy.interpolate import interp1d
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.util.metadata import MetaDict
from sunpy.coordinates.frames import Helioprojective, HeliographicStonyhurst
from sunpy.map import make_fitswcs_header
import distributed
import zarr


class InstrumentBase(object):
    """
    Base class for instruments. Need to at least implement a detect() method that is used by the
    `Observer` class to get the detector counts.

    Parameters
    ----------
    observing_time : `~astropy.units.Quantity`
        Tuple of start and end observing times
    observer_coordinate : `~astropy.coordinates.SkyCoord`
        Coordinate of the observing instrument
    """
    fits_template = MetaDict()

    @u.quantity_input
    def __init__(self, observing_time: u.s, observer):
        self.observing_time = np.arange(observing_time[0].to(u.s).value,
                                        observing_time[1].to(u.s).value,
                                        self.cadence.value)*u.s
        self.observer = observer

    def calculate_intensity_kernel(self, *args, **kwargs):
        """
        Converts emissivity for a particular transition to counts per detector channel. When writing
        a new instrument class, this method should be overridden.
        """
        raise NotImplementedError('No detect method implemented.')

    def integrate_los(self, time, channel, skeleton):
        """
        Integrate intensity along a LOS.
        """
        # TODO: decide if this could be generalized for all instruments?
        raise NotImplementedError

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
        loop_cs = 1e14 * u.cm**2
        w_x, w_y = (1*u.pix * self.resolution).to(u.radian).value * self.observer.radius
        return (loop_cs / (w_x * w_y)).decompose()

    def observe(self, skeleton, channels=None, **kwargs):
        # This method can be attached to the base class
        # This is where the actual forward modeling is done
        #
        # 1. Construct detector array for given instrument, observer at t_i
        # 2. Get all loop coordinates in frame of observer at t_i
        # 3. Compute kernel of LOS intensity integral for all coordinates at t_i
        # 4. Bin coordinates, weighted by kernels and grid cell width and x.s. ratio, into detector array
        # 5. Write out as a sunpy map with appropriate metadata
        #
        client = distributed.get_client()
        if channels is None:
            channels = self.channels
        maps = {c.name: [] for c in channels}
        for channel in channels:
            kernels = client.map(self.calculate_intensity_kernel,
                                 skeleton.loops,
                                 channel=channel)
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
            for t in self.observing_time:
                m = self.integrate_los(t, channel, skeleton)
                maps[channel.name].append(m)

        return maps

    @staticmethod
    def write_kernel_to_file(kernel, loop, channel, name):
        # This method can be moved to the base instrument
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
        # This method can be moved to the base instrument
        time = loop.time
        if time.shape == (1,):
            if time != observing_time:
                raise ValueError('Model and observing times are not equal for a single model time step.')
            return kernel
        f_t = interp1d(time.to(observing_time.unit).value, kernel.value, axis=0)
        return f_t(observing_time.value) * k.unit

    def get_header(self, channel, coordinates):
        # This method can be attached to the base object
        bins, bin_range = self.get_detector_array(coordinates)
        header = make_fitswcs_header(
            (bins[1],bins[0]),  # swap order because it expects (row,column)
            bin_range[0],  # align with the lower left corner of the lower left pixel
            reference_pixel=(-0.5, -0.5)*u.pixel,  # center of the lower left pixel is (0,0)
            scale=self.resolution,
            instrument=f'{self.detector}_{channel.telescope_number}',
            telescope=self.telescope,
            wavelength=channel.channel,
        )
        return header
        
    def get_detector_array(self, coordinates):
        # This method can be attached to the base object
        coordinates = coordinates.transform_to(self.projected_frame)
        if self.pad_fov is None:
            pad_x, pad_y = 0*u.arcsec, 0*u.arcsec
        # Note: this is the coordinate of the bottom left corner of the bottom left corner pixel
        # This is NOT the coordinate at the center of the pixel!
        bottom_left_corner = SkyCoord(
            Tx=coordinates.Tx.min() - pad_x,
            Ty=coordinates.Ty.min() - pad_y,
            frame=coordinates.frame
        )
        bins_x = int(np.ceil((coordinates.Tx.max() + pad_x - bottom_left_corner.Tx) / self.resolution[0]).value)
        bins_y = int(np.ceil((coordinates.Ty.max() + pad_y - bottom_left_corner.Ty) / self.resolution[1]).value)
        # Compute right corner after the fact to account for rounding in bin numbers
        top_right_corner = SkyCoord(
            Tx=bottom_left_corner.Tx + self.resolution[0]*bins_x*u.pixel,
            Ty=bottom_left_corner.Ty + self.resolution[1]*bins_y*u.pixel,
            frame=coordinates.frame
        )
        return (bins_x, bins_y), (bottom_left_corner, top_right_corner)
