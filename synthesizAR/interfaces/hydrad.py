"""
Model interface for the HYDrodynamics and RADiation (HYDRAD) code
"""
import os

import numpy as np
from scipy.interpolate import splrep, splprep, splev
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
import sunpy.sun.constants as sun_const
from hydrad_tools.configure import Configure
from hydrad_tools.parse import Strand

__all__ = ['HYDRADInterface']


class HYDRADInterface(object):
    """
    Interface to the HYDrodynamics and RADiation (HYDRAD) code

    Configure, interpolate, and load results for HYDRAD simulations
    for each loop in the magnetic skeleton. Note that the load function
    will also reinterpolate the loop coordinates and magnetic field
    strength for each loop such that the spatial resolution is
    `delta_s_uniform`.

    Parameters
    ----------
    base_config: `dict`
        Dictionary of configuration options for the HYDRAD model.
    hydrad_dir: `str`
        Path to a "clean copy" of HYDRAD.
    output_dir: `str`
        Root directory to place all of the HYDRAD results in. Subdirectories
        will be named according to the name of each loop.
    heating_model: object
        Instance of a heating model class that describes when and
        where the heating should occur along the loop
    delta_s_uniform: `~astropy.units.Quantity`
        Grid spacing of the uniform spatial grid to which all
        quantities will be interpreted
    """
    name = 'HYDRAD'

    @u.quantity_input
    def __init__(self, base_config, hydrad_dir, output_dir, heating_model, delta_s_uniform: u.cm):
        self.base_config = base_config
        self.hydrad_dir = hydrad_dir
        self.output_dir = output_dir
        self.heating_model = heating_model
        self.delta_s_uniform = delta_s_uniform
        self.max_grid_cell = 1e8*u.cm
    
    def configure_input(self, loop):
        config = self.base_config.copy()
        config['general']['loop_length'] = loop.length
        # This makes sure that the chromosphere does not take up the whole loop
        config['general']['footpoint_height'] = 0.5 * min(10*u.Mm, 0.5*loop.length)
        config['initial_conditions']['heating_location'] = loop.length / 2.
        config['grid']['minimum_cells'] = int(loop.length / self.max_grid_cell)
        # Gravity and cross-section coefficients
        config['general']['tabulated_gravity_profile'] = self.get_gravity_coefficients(loop)
        config['general']['tabulated_cross_section_profile'] = self.get_cross_section_coefficients(loop)
        # Heating configuration
        config['heating']['events'] = self.heating_model.calculate_event_properties(loop)
        # Setup configuration and generate initial conditions
        c = Configure(config)
        c.setup_simulation(self.output_dir, base_path=self.hydrad_dir, name=loop.name,
                           verbose=False)

    def load_results(self, loop):
        # Create the strand and uniform coordinate
        s = Strand(os.path.join(self.output_dir, loop.name), read_amr=False)
        s_uniform = np.arange(
            0, s.loop_length.to(u.cm).value, self.delta_s_uniform.to(u.cm).value)*u.cm
        # Preallocate space for arrays
        shape = s.time.shape + s_uniform.shape
        electron_temperature = np.zeros(shape)
        ion_temperature = np.zeros(shape)
        density = np.zeros(shape)
        velocity = np.zeros(shape)

        # Interpolate each quantity at each timestep
        for i, _ in enumerate(s.time):
            p = s[i]
            coord = p.coordinate.to(u.cm).value
            tsk = splrep(coord, p.electron_temperature.to(u.K).value,)
            electron_temperature[i, :] = splev(s_uniform.value, tsk, ext=0)
            tsk = splrep(coord, p.ion_temperature.to(u.K).value,)
            ion_temperature[i, :] = splev(s_uniform.value, tsk, ext=0)
            tsk = splrep(coord, p.electron_density.to(u.cm**(-3)).value)
            density[i, :] = splev(s_uniform.value, tsk, ext=0)
            tsk = splrep(coord, p.velocity.to(u.cm/u.s).value,)
            velocity[i, :] = splev(s_uniform.value, tsk, ext=0)

        # Interpolate loop coordinates
        tsk, _ = splprep(loop.coordinates.cartesian.xyz.value)
        coord_xyz = splev((s_uniform/s.loop_length).decompose().value, tsk)
        loop._coordinates = SkyCoord(
            x=coord_xyz[0, :]*loop.coordinates.cartesian.xyz.unit,
            y=coord_xyz[1, :]*loop.coordinates.cartesian.xyz.unit,
            z=coord_xyz[2, :]*loop.coordinates.cartesian.xyz.unit,
            frame=loop._coordinates.frame,
            representation='cartesian',
        )
        # Interpolate magnetic field strength
        tsk = splrep(loop.field_aligned_coordinate.value, loop.field_strength.value)
        field_strength = splev(s_uniform, tsk)
        loop._field_strength = u.Quantity(
            np.where(field_strength < 0, 0., field_strength), loop.field_strength.unit)

        return (
            s.time,
            electron_temperature*u.K,
            ion_temperature*u.K,
            density*u.cm**(-3),
            velocity*u.cm/u.s,
        )

    def get_cross_section_coefficients(self, loop):
        s_norm = loop.field_aligned_coordinate / loop.length
        return np.polyfit(s_norm, loop.field_strength, 6)[::-1]

    def get_gravity_coefficients(self, loop):
        s_norm = loop.field_aligned_coordinate / loop.length
        s_hat = (np.gradient(loop.coordinates.cartesian.xyz, axis=1)
                 / np.linalg.norm(np.gradient(loop.coordinates.cartesian.xyz, axis=1), axis=0))
        r_hat = u.Quantity(np.stack([
            np.sin(loop.coordinates.spherical.lat)*np.cos(loop.coordinates.spherical.lon),
            np.sin(loop.coordinates.spherical.lat)*np.sin(loop.coordinates.spherical.lon),
            np.cos(loop.coordinates.spherical.lat)
        ]))
        g_parallel = -sun_const.surface_gravity.cgs * ((
            const.R_sun.cgs / loop.coordinates.spherical.distance)**2) * (r_hat * s_hat).sum(axis=0)
        return np.polyfit(s_norm, g_parallel, 6)[::-1]
