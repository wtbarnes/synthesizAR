"""
Model interface for the HYDrodynamics and RADiation (HYDRAD) code
"""
import os

import numpy as np
from scipy.interpolate import splrep, splev
import astropy.units as u
import astropy.constants as const
import sunpy.sun.constants as sun_const

from pydrad.configure import Configure
from pydrad.parse import Strand


__all__ = ['HYDRADInterface']


class HYDRADInterface(object):
    """
    Interface to the HYDrodynamics and RADiation (HYDRAD) code

    Configure, interpolate, and load results for HYDRAD simulations
    for each loop in the magnetic skeleton.

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
    use_gravity: `bool`, optional
        If true, use the loop coordinates to determine the gravitational stratification
    use_magnetic_field: `bool`, optional
        If true, use the magnetic field profile to determine the loop expansion factor
    use_initial_conditions: `bool`, optional
        If true, use only the hydrostatic initial conditions as the model
    maximum_chromosphere_ratio: `float`, optional
        Maximum allowed ratio between the loop length and the total chromsphere depth.
        If specified, `general.footpoint_height` will be set to this ratio times the
        loop length if ``2 * general.footpoint_height / length`` is greater than this
        ratio.
    """
    name = 'HYDRAD'

    @u.quantity_input
    def __init__(self,
                 base_config,
                 hydrad_dir,
                 output_dir,
                 heating_model,
                 use_gravity=True,
                 use_magnetic_field=True,
                 use_initial_conditions=False,
                 maximum_chromosphere_ratio=None):
        self.base_config = base_config
        self.hydrad_dir = hydrad_dir
        self.output_dir = output_dir
        self.heating_model = heating_model
        self.use_gravity = use_gravity
        self.use_magnetic_field = use_magnetic_field
        self.use_initial_conditions = use_initial_conditions
        self.maximum_chromosphere_ratio = maximum_chromosphere_ratio

    def configure_input(self, loop):
        config = self.base_config.copy()
        # NOTE: Truncate precision in loop length here as passing a loop length
        # with too many significant figures to HYDRAD can cause issues with the
        # initical conditions calculation. Here, we truncate the loop length such
        # that it has 1 significant figure when expressed in Mm
        length = float(f'{loop.length.to("Mm").value:.1f}') * u.Mm
        config['general']['loop_length'] = length
        config['initial_conditions']['heating_location'] = length / 2.
        if self.maximum_chromosphere_ratio:
            config['general']['footpoint_height'] = min(config['general']['footpoint_height'],
                                                        self.maximum_chromosphere_ratio * length / 2)
        if self.use_gravity:
            config['general']['poly_fit_gravity'] = self.configure_gravity_fit(loop)
        if self.use_magnetic_field:
            config['general']['poly_fit_magnetic_field'] = self.configure_magnetic_field_fit(loop)
        config = self.heating_model.calculate_event_properties(config, loop)
        c = Configure(config)
        c.setup_simulation(os.path.join(self.output_dir, loop.name),
                           base_path=self.hydrad_dir,
                           verbose=False)

    def load_results(self, loop):
        loop_coord_center = loop.field_aligned_coordinate_center.to(u.cm).value
        s = Strand(os.path.join(self.output_dir, loop.name))
        if self.use_initial_conditions:
            time = s.initial_conditions.time.reshape((1,))
            s = [s.initial_conditions]
        else:
            time = s.time
        shape = time.shape + loop_coord_center.shape
        electron_temperature = np.zeros(shape)
        ion_temperature = np.zeros(shape)
        density = np.zeros(shape)
        velocity = np.zeros(shape)
        for i, p in enumerate(s):
            coord = p.coordinate.to(u.cm).value
            tsk = splrep(coord, p.electron_temperature.to(u.K).value,)
            electron_temperature[i, :] = splev(loop_coord_center, tsk, ext=0)
            tsk = splrep(coord, p.ion_temperature.to(u.K).value,)
            ion_temperature[i, :] = splev(loop_coord_center, tsk, ext=0)
            tsk = splrep(coord, p.electron_density.to(u.cm**(-3)).value)
            density[i, :] = splev(loop_coord_center, tsk, ext=0)
            tsk = splrep(coord, p.velocity.to(u.cm/u.s).value,)
            velocity[i, :] = splev(loop_coord_center, tsk, ext=0)

        return (
            time,
            electron_temperature*u.K,
            ion_temperature*u.K,
            density*u.cm**(-3),
            velocity*u.cm/u.s,
        )

    def configure_gravity_fit(self, loop):
        return {
            'order': self.base_config['general']['poly_fit_gravity']['order'],
            'domains': self.base_config['general']['poly_fit_gravity']['domains'],
            'x': loop.field_aligned_coordinate_norm.decompose().value,
            'y': loop.gravity,
        }

    def configure_magnetic_field_fit(self, loop):
        return {
            'order': self.base_config['general']['poly_fit_magnetic_field']['order'],
            'domains': self.base_config['general']['poly_fit_magnetic_field']['domains'],
            'x': loop.field_aligned_coordinate_norm.decompose().value,
            'y': loop.field_strength,
        }
