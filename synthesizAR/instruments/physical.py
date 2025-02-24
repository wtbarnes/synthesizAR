"""
Instruments for calculating physical quantities, rather than
observed counts, projected along a LOS
"""
import astropy.units as u
import astropy.wcs
import ndcube
import numpy as np

from dataclasses import dataclass
from scipy.interpolate import interp1d

from synthesizAR.instruments import ChannelBase, InstrumentBase
from synthesizAR.instruments.util import add_wave_keys_to_header, extend_celestial_wcs
from synthesizAR.util import los_velocity
from synthesizAR.util.decorators import return_quantity_as_tuple

__all__ = [
    'InstrumentDEM',
    'InstrumentVDEM',
    'InstrumentLOSVelocity',
    'InstrumentTemperature'
]


@dataclass
class ChannelDEM(ChannelBase):
    # NOTE: These have to have a default value because of the
    # keyword arguments on the base class.
    bin_edges: u.Quantity[u.K] = None

    def __post_init__(self):
        self.log_bin_edges = np.log10(self.bin_edges.to_value('K'))
        self.name = f'{self.log_bin_edges[0]:.2f}-{self.log_bin_edges[1]:.2f}'

    @property
    @u.quantity_input
    def bin_center(self) -> u.K:
        return self.bin_edges.sum()/2


class InstrumentDEM(InstrumentBase):
    name = 'DEM'

    @u.quantity_input
    def __init__(self, *args, temperature_bin_edges: u.K, **kwargs):
        self.temperature_bin_edges = temperature_bin_edges
        super().__init__(*args, **kwargs)

    @property
    def channels(self):
        n_bins = self.temperature_bin_edges.shape[0]-1
        bin_edges = [self.temperature_bin_edges[[i, i+1]] for i in range(n_bins)]
        return [ChannelDEM(bin_edges=be) for be in bin_edges]

    @property
    @u.quantity_input
    def temperature_bin_centers(self) -> u.K:
        return (self.temperature_bin_edges[1:] + self.temperature_bin_edges[:-1])/2

    def get_instrument_name(self, channel):
        # This ensures that the temperature bin labels are in the header
        return f'{self.name}_{channel.name}'

    @property
    def _expected_unit(self):
        return u.cm**(-5)

    @staticmethod
    @return_quantity_as_tuple
    def calculate_intensity_kernel(loop, channel, **kwargs):
        T = loop.electron_temperature
        n = loop.density
        bin_mask = np.where(
            np.logical_and(T >= channel.bin_edges[0], T < channel.bin_edges[1]), 1, 0)
        kernel = n**2 * bin_mask
        return kernel

    def maps_to_cube(self, dem, time_index):
        """
        Transform a set of DEM maps at a single time step to a `~ndcube.NDCube`
        """
        # NOTE: this is the format that .observe returns
        return type(self).dem_maps_list_to_cube(
            [dem[c.name][time_index] for c in self.channels],
            self.temperature_bin_centers,
        )

    @staticmethod
    def dem_maps_list_to_cube(dem_maps, temperature_bin_centers):
        compound_wcs = extend_celestial_wcs(
            dem_maps[0].wcs,
            (temperature_bin_centers, 'temperature', 'phys.temperature'),
        )
        dem_array = u.Quantity([d.quantity for d in dem_maps])
        return ndcube.NDCube(dem_array, wcs=compound_wcs, meta=dem_maps[0].meta)

    @staticmethod
    def calculate_intensity(dem, spectra, header, meta=None):
        """
        Compute intensity from a DEM and a temperature-dependent spectra

        Parameters
        ----------
        dem: `~ndcube.NDCube`
            The first axis should correspond to temperature
        spectra: `~ndcube.NDCube`
        header: `dict` or header-like
            Header information corresponding to the spatial axes of the DEM cube
        meta: `dict`, optional
            Additional metadata
        wavelength
        """
        temperature_bin_centers = dem.axis_world_coords(0)[0]
        wavelength_spectra = spectra.axis_world_coords(1)[0]
        temperature_spectra = spectra.axis_world_coords(0)[0].to(temperature_bin_centers.unit)
        f_interp = interp1d(temperature_spectra.value,
                            spectra.data,
                            axis=0,
                            bounds_error=False,
                            fill_value=0.0)
        spectra_interp = f_interp(temperature_bin_centers.value)
        spectra_interp = spectra_interp * spectra.unit
        intensity = np.tensordot(spectra_interp, u.Quantity(dem.data, dem.unit), axes=([0], [0]))
        wave_header = add_wave_keys_to_header(wavelength_spectra, header)
        wave_header['BUNIT'] = intensity.unit.to_string()
        wave_header['NAXIS'] = len(intensity.shape)
        wave_header['WCSAXES'] = len(intensity.shape)
        meta = {} if meta is None else meta
        meta = {**meta, **wave_header}
        wcs = astropy.wcs.WCS(header=wave_header)
        return ndcube.NDCube(intensity, wcs=wcs, meta=meta)


@dataclass
class ChannelVDEM(ChannelBase):
    # NOTE: These have to have a default value because of the
    # keyword arguments on the base class.
    temperature_bin_edges: u.Quantity[u.K] = None
    velocity_bin_edges: u.Quantity[u.km/u.s] = None

    def __post_init__(self):
        self.name = self._make_channel_label(self.temperature_bin_edges,
                                             self.velocity_bin_edges)

    @staticmethod
    def _make_channel_label(temperature_bin_edges, velocity_bin_edges):
        # This is a static method so it can be used externally
        log_tbin_edges = np.log10(temperature_bin_edges.to_value('K'))
        vbin_edges = velocity_bin_edges.to_value('km/s')
        tname = f'{log_tbin_edges[0]:.2f}-{log_tbin_edges[1]:.2f}'
        vname = f'{vbin_edges[0]:.2f}-{vbin_edges[1]:.2f}'
        return f'logT:{tname}_v:{vname}'

    @property
    @u.quantity_input
    def temperature_bin_center(self) -> u.K:
        return self.temperature_bin_edges.sum()/2

    @property
    @u.quantity_input
    def velocity_bin_center(self) -> u.Unit('km/s'):
        return self.velocity_bin_edges.sum()/2


class InstrumentVDEM(InstrumentBase):
    name = 'VDEM'

    @u.quantity_input
    def __init__(self,
                 *args,
                 temperature_bin_edges: u.K,
                 velocity_bin_edges: u.Unit('km/s'),
                 **kwargs):
        self.temperature_bin_edges = temperature_bin_edges
        self.velocity_bin_edges = velocity_bin_edges
        super().__init__(*args, **kwargs)

    @property
    def channels(self):
        channels = []
        for i in range(self.temperature_bin_centers.shape[0]):
            for j in range(self.velocity_bin_centers.shape[0]):
                channels.append(
                    ChannelVDEM(temperature_bin_edges=self.temperature_bin_edges[[i,i+1]],
                                velocity_bin_edges=self.velocity_bin_edges[[j,j+1]])
                )
        return channels

    @property
    @u.quantity_input
    def temperature_bin_centers(self) -> u.K:
        return (self.temperature_bin_edges[1:] + self.temperature_bin_edges[:-1])/2

    @property
    @u.quantity_input
    def velocity_bin_centers(self) -> u.Unit('km/s'):
        return (self.velocity_bin_edges[1:] + self.velocity_bin_edges[:-1])/2

    def get_instrument_name(self, channel):
        # This ensures that the temperature bin labels are in the header
        return f'{self.name}_{channel.name}'

    @property
    def _expected_unit(self):
        return u.cm**(-5)

    def observe(self, *args, **kwargs):
        kwargs['observer'] = self.observer
        return super().observe(*args, **kwargs)

    @staticmethod
    @return_quantity_as_tuple
    def calculate_intensity_kernel(loop, channel, **kwargs):
        observer = kwargs.get('observer')
        T = loop.electron_temperature
        n = loop.density
        v_los = los_velocity(loop.velocity_xyz, observer)
        in_temperature_bin = np.logical_and(T>=channel.temperature_bin_edges[0],
                                            T<channel.temperature_bin_edges[1])
        in_velocity_bin = np.logical_and(v_los>=channel.velocity_bin_edges[0],
                                         v_los<channel.velocity_bin_edges[1])
        bin_mask = np.where(np.logical_and(in_temperature_bin, in_velocity_bin), 1, 0)
        kernel = n**2 * bin_mask
        return kernel

    def maps_to_cube(self, vdem, time_index):
        """
        Transform a set of DEM maps at a single time step to a `~ndcube.NDCube`
        """
        arrays = []
        for j in range(self.velocity_bin_centers.shape[0]):
            _arrays = []
            for i in range(self.temperature_bin_centers.shape[0]):
                key = self.channels[0]._make_channel_label(
                    self.temperature_bin_edges[[i,i+1]],
                    self.velocity_bin_edges[[j,j+1]]
                )
                _map = vdem[key][time_index]
                _arrays.append(_map.data)
            arrays.append(_arrays)
        arrays = np.array(arrays)
        compound_wcs = extend_celestial_wcs(
            _map.wcs,
            (self.temperature_bin_centers, 'temperature', 'phys.temperature'),
            (self.velocity_bin_centers, 'velocity', 'phys.velocity'),
        )
        return ndcube.NDCube(arrays, wcs=compound_wcs, meta=_map.meta, unit=_map.unit)


class InstrumentQuantityBase(InstrumentBase):

    @u.quantity_input
    def __init__(self, *args, **kwargs):
        super().__init__(*args, average_over_los=True, **kwargs)

    @property
    def channels(self):
        return [ChannelBase(name=self.name)]


class InstrumentLOSVelocity(InstrumentQuantityBase):
    name = 'los_velocity'

    @staticmethod
    @return_quantity_as_tuple
    def calculate_intensity_kernel(loop, *args, **kwargs):
        observer = kwargs.get('observer')
        return los_velocity(loop.velocity_xyz, observer)

    def observe(self, *args, **kwargs):
        kwargs['observer'] = self.observer
        return super().observe(*args, **kwargs)

    @property
    def _expected_unit(self):
        return u.km / u.s


class InstrumentTemperature(InstrumentQuantityBase):
    name = 'temperature'

    @staticmethod
    @return_quantity_as_tuple
    def calculate_intensity_kernel(loop, *args, **kwargs):
        return loop.electron_temperature

    @property
    def _expected_unit(self):
        return u.K
