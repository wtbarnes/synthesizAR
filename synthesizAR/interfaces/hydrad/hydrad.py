"""
Model interface for the HYDrodynamics and RADiation (HYDRAD) code
"""
import os

import numpy as np
from scipy.interpolate import splrep, splev
import astropy.units as u
from pydrad.configure import Configure
from pydrad.parse import Strand

from .helpers import cross_section_coefficients, gravity_coefficients

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
    """
    name = 'HYDRAD'

    @u.quantity_input
    def __init__(self, base_config, hydrad_dir, output_dir, heating_model):
        self.base_config = base_config
        self.hydrad_dir = hydrad_dir
        self.output_dir = output_dir
        self.heating_model = heating_model

    def configure_input(self, loop):
        config = self.base_config.copy()
        config['general']['loop_length'] = loop.length
        config['initial_conditions']['heating_location'] = loop.length / 2.
        config['general']['poly_fit_gravity'] = gravity_coefficients(loop)
        config['general']['poly_fit_magnetic_field'] = cross_section_coefficients(loop)
        config = self.heating_model.calculate_event_properties(config, loop)
        c = Configure(config)
        c.setup_simulation(os.path.join(self.output_dir, loop.name),
                           base_path=self.hydrad_dir,
                           verbose=False)

    def load_results(self, loop):
        loop_coord_center = loop.field_aligned_coordinate_center.to(u.cm).value
        s = Strand(os.path.join(self.output_dir, loop.name))
        shape = s.time.shape + loop_coord_center.shape
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
            s.time,
            electron_temperature*u.K,
            ion_temperature*u.K,
            density*u.cm**(-3),
            velocity*u.cm/u.s,
        )
