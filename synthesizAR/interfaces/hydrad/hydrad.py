"""
Model interface for the HYDrodynamics and RADiation (HYDRAD) code
"""
import astropy.units as u
import copy
import pathlib
import pydrad.configure
import pydrad.parse

from synthesizAR import log

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
        self.hydrad_dir = hydrad_dir
        self.heating_model = heating_model
        self.use_gravity = use_gravity
        self.use_magnetic_field = use_magnetic_field
        self.use_initial_conditions = use_initial_conditions
        self.maximum_chromosphere_ratio = maximum_chromosphere_ratio
        self.interpolate_to_norm = interpolate_to_norm

    @property
    def hydrad_dir(self):
        return self._hydrad_dir

    @hydrad_dir.setter
    def hydrad_dir(self, value):
        if value is not None:
            value = pathlib.Path(value)
        self._hydrad_dir = value

    def _map_strand_to_config_dict(self, loop):
        # NOTE: This is a separate function for ease of debugging
        config = copy.deepcopy(self.base_config)
        config['general']['loop_length'] = loop.length
        config['initial_conditions']['heating_location'] = loop.length / 2
        if self.maximum_chromosphere_ratio:
            config['general']['footpoint_height'] = min(
                config['general']['footpoint_height'], self.maximum_chromosphere_ratio * loop.length / 2)
        if self.use_gravity:
            config['general']['poly_fit_gravity'] = self.configure_gravity_fit(loop)
        if self.use_magnetic_field:
            config['general']['poly_fit_magnetic_field'] = self.configure_magnetic_field_fit(loop)
        config = self.heating_model.calculate_event_properties(config, loop)
        return config

    def configure_input(self, loop, **kwargs):
        # Import here to avoid circular imports
        from synthesizAR import log
        log.debug(f'Configuring HYDRAD for {loop.name}')
        config_dict = self._map_strand_to_config_dict(loop)
        c = pydrad.configure.Configure(config_dict)
        c.setup_simulation(self.output_dir / loop.name,
                           base_path=self.hydrad_dir,
                           **kwargs)

    def load_results(self, loop, emission_model=None):
        read_ine = emission_model is not None
        strand = pydrad.parse.Strand(self.output_dir / loop.name,
                                     read_ine=read_ine,
                                     read_phy=False,
                                     read_hstate=False,
                                     read_trm=False,
                                     read_scl=False)
        return self._load_results_from_strand(
            loop,
            strand,
            use_initial_conditions=self.use_initial_conditions,
            interpolate_to_norm=self.interpolate_to_norm,
            emission_model=emission_model,
        )

    @staticmethod
    def _load_results_from_strand(loop,
                                  strand,
                                  use_initial_conditions=False,
                                  interpolate_to_norm=False,
                                  emission_model=None):
        if interpolate_to_norm:
            loop_coord_center = loop.field_aligned_coordinate_center_norm
        else:
            loop_coord_center = loop.field_aligned_coordinate_center
        if use_initial_conditions:
            time = strand.initial_conditions.time.reshape((1,))
            strand = strand.initial_conditions
        else:
            time = strand.time
        quantity_names = [
            'electron_temperature',
            'hydrogen_temperature',
            'electron_density',
            'velocity',
        ]
        quantity_name_mapping = {
            'hydrogen_temperature': 'ion_temperature',
            'electron_density': 'density',
        }
        if emission_model:
            quantity_names += [f'{ion.element_name}_{ion.ionization_stage}' for ion in emission_model]
        quantities = {quantity_name_mapping.get(qn, qn): [] for qn in quantity_names}
        # NOTE: Purposefully not using strand.to_constant_grid to avoid re-instantiating
        # profiles and reading files multiple times.
        for profile in strand:
            for name in quantity_names:
                # Need this try/except as the HYDRAD simulations may not have ine files
                # for every ion in the emission model.
                try:
                    quantity = profile.to_constant_grid(name, loop_coord_center)
                except AttributeError:
                    log.warning(f'No quantity {name} for HYDRAD snapshot {profile.time}.')
                else:
                    quantities[quantity_name_mapping.get(name, name)].append(quantity)
        quantities = {k: u.Quantity(v) for k, v in quantities.items()}
        return {'time': time, **quantities}

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
