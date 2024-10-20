"""
Model interface for the HYDrodynamics and RADiation (HYDRAD) code
"""
import astropy.units as u
import numpy as np
import os
import pathlib
import pydrad.parse

from pydrad.configure import Configure
from scipy.interpolate import splev, splrep

__all__ = ['HYDRADInterface']


class HYDRADInterface:
    """
    Interface to the HYDrodynamics and RADiation (HYDRAD) code

    Configure, interpolate, and load results for HYDRAD simulations
    for each loop in the magnetic skeleton. Note that if you want to
    just use a preexisting HYDRAD simulation, the simulations should
    just be placed in the `output_dir` directory with the name of
    each subdirectory corresponding to the appropriate loop. If you do
    this, all other input parameters besides `interpolate_to_norm` can
    be omitted.

    Parameters
    ----------
    output_dir: pathlike
        Root directory to place all of the HYDRAD results in. Subdirectories
        will be named according to the name of each loop.
    base_config: `dict`, optional
        Dictionary of configuration options for the HYDRAD model.
    hydrad_dir: pathlike, optional
        Path to a "clean copy" of HYDRAD.
    heating_model: object, optional
        Instance of a heating model class that describes when and
        where the heating should occur along the loop
    use_gravity: `bool`, optional
        If true, use the loop coordinates to determine the gravitational stratification
    use_magnetic_field: `bool`, optional
        If true, use the magnetic field profile to determine the loop expansion factor
    use_initial_conditions: `bool`, optional
        If true, use only the hydrostatic initial conditions as the model
    maximum_chromosphere_ratio: `float`, optional
        Maximum allowed ratio between the loop length and the total chromosphere depth.
        If specified, `general.footpoint_height` will be set to this ratio times the
        loop length if ``2 * general.footpoint_height / length`` is greater than this
        ratio.
    interpolate_to_norm: `bool`, optional
        If True, the loop quantities are interpolated using the coordinates normalized
        by the loop length. In cases where the length of the simulated loop does not
        match that of the geometric loop model, using this option will "stretch" or
        "squash" the simulated solution appropriately
    """
    name = 'HYDRAD'

    @u.quantity_input
    def __init__(self,
                 output_dir,
                 base_config=None,
                 hydrad_dir=None,
                 heating_model=None,
                 use_gravity=True,
                 use_magnetic_field=True,
                 use_initial_conditions=False,
                 maximum_chromosphere_ratio=None,
                 interpolate_to_norm=False):
        self.output_dir = pathlib.Path(output_dir)
        self.base_config = base_config
        self.hydrad_dir = hydrad_dir if hydrad_dir is None else pathlib.Path(hydrad_dir)
        self.heating_model = heating_model
        self.use_gravity = use_gravity
        self.use_magnetic_field = use_magnetic_field
        self.use_initial_conditions = use_initial_conditions
        self.maximum_chromosphere_ratio = maximum_chromosphere_ratio
        self.interpolate_to_norm = interpolate_to_norm

    def configure_input(self, loop):
        config = self.base_config.copy()
        config['general']['loop_length'] = loop.length
        config['initial_conditions']['heating_location'] = loop.length / 2.
        if self.maximum_chromosphere_ratio:
            config['general']['footpoint_height'] = min(
                config['general']['footpoint_height'], self.maximum_chromosphere_ratio * loop.length / 2)
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
        strand = pydrad.parse.Strand(self.output_dir / loop.name)
        return self._load_results_from_strand(
            loop,
            strand,
            use_initial_conditions=self.use_initial_conditions,
            interpolate_to_norm=self.interpolate_to_norm,
        )

    @staticmethod
    def _load_results_from_strand(loop,
                                  strand,
                                  use_initial_conditions=False,
                                  interpolate_to_norm=False):
        loop_coord_center = loop.field_aligned_coordinate_center.to_value('cm')
        if interpolate_to_norm:
            loop_coord_center = loop.field_aligned_coordinate_center_norm.value
        if use_initial_conditions:
            time = strand.initial_conditions.time.reshape((1,))
            strand = [strand.initial_conditions]
        else:
            time = strand.time
        shape = time.shape + loop_coord_center.shape
        quantities = {
            'electron_temperature': np.zeros(shape) * u.K,
            'ion_temperature': np.zeros(shape) * u.K,
            'density': np.zeros(shape) * u.cm**(-3),
            'velocity': np.zeros(shape) * u.cm/u.s,
        }
        for i, p in enumerate(strand):
            coord = p.coordinate.to('cm').value
            if interpolate_to_norm:
                coord /= strand.loop_length.to('cm').value
            for k in quantities:
                q = getattr(p, 'electron_density' if k == 'density' else k)
                tsk = splrep(coord, q.to_value(quantities[k].unit))
                q_interp = splev(loop_coord_center, tsk, ext=0)
                quantities[k][i, :] = u.Quantity(q_interp, quantities[k].unit)

        return (
            time,
            quantities['electron_temperature'],
            quantities['ion_temperature'],
            quantities['density'],
            quantities['velocity'],
        )

    def configure_gravity_fit(self, loop):
        return {
            'order': self.base_config['general']['poly_fit_gravity']['order'],
            'domains': self.base_config['general']['poly_fit_gravity']['domains'],
            'x': loop.field_aligned_coordinate,
            'y': loop.gravity,
        }

    def configure_magnetic_field_fit(self, loop):
        return {
            'order': self.base_config['general']['poly_fit_magnetic_field']['order'],
            'domains': self.base_config['general']['poly_fit_magnetic_field']['domains'],
            'x': loop.field_aligned_coordinate,
            'y': loop.field_strength,
        }
