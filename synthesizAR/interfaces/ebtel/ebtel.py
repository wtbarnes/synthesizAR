"""
Interface between loop object and ebtel++ simulation
"""
import astropy.units as u
import ebtelplusplus
import ebtelplusplus.models
import numpy as np


class EbtelInterface:
    """
    Interface to the Enthalpy-Based Thermal Evolution of Loops (EBTEL) model

    This interface uses `ebtelplusplus` to run an EBTEL simulation based on
    the properties of a particular `synthesizAR.Strand`.

    Parameters
    ----------
    total_time : `~astropy.units.Quantity`
        The total time of the simulation. This will be the same for all strands
    event_builder : `synthesizAR.interfaces.ebtel.AbstractEventBuilder`, optional
        Mapping between strand properties and heating event properties
    heating_model : `ebtelplusplus.models.HeatingModel`, optional
        A model that specifies the background and energy partition. Events are attached
        to this model per strand by ``event_builder``.
    physics : `ebtelplusplus.models.PhysicsModel`, optional
    solver : `ebtelplusplus.models.SolverModel`, optional
    """
    name = 'EBTEL'

    @u.quantity_input
    def __init__(self,
                 total_time: u.s,
                 event_builder=None,
                 heating_model=None,
                 physics_model=None,
                 solver_model=None):
        self.total_time = total_time
        self.event_builder = event_builder
        self.heating_model = heating_model
        self.physics_model = physics_model
        self.solver_model = solver_model

    @property
    def heating_model(self):
        return self._heating_model

    @heating_model.setter
    def heating_model(self, val):
        # NOTE: unlike the other models, this cannot be None at this level because
        # we optionally attach events to it.
        if val is None:
            val = ebtelplusplus.models.HeatingModel()
        if val.events and self.event_builder is not None:
            raise ValueError(
                'Specifying an event_builder will override existing events on heating model.'
                'Either specify events explicitly or provide an event_builder but not both.'
            )
        self._heating_model = val

    def load_results(self, strand):
        """
        Load EBTEL output for a given particular strand.

        Parameters
        ----------
        strand : `synthesizAR.Strand`
        """
        if self.event_builder is not None:
            self.heating_model.events = self.event_builder(strand)
        results = ebtelplusplus.run(self.total_time,
                                    strand.length/2,
                                    heating=self.heating_model,
                                    physics=self.physics_model,
                                    solver=self.solver_model,
                                    dem=None)
        electron_temperature = self._map_quantity_to_strand(strand, results.electron_temperature)
        ion_temperature = self._map_quantity_to_strand(strand, results.ion_temperature)
        density = self._map_quantity_to_strand(strand, results.density)
        velocity = self._map_velocity_to_strand(strand, results.velocity)
        return results.time, electron_temperature, ion_temperature, density, velocity

    def _map_quantity_to_strand(self, strand, quantity):
        return np.outer(quantity, np.ones(strand.n_s))

    def _map_velocity_to_strand(self, strand, quantity):
        velocity = self._map_quantity_to_strand(strand, quantity)
        # flip sign of velocity where the radial distance from center is maximum
        # FIXME: this is probably not the best way to do this...
        r = np.sqrt(np.sum(strand.coordinate_center.cartesian.xyz.value**2, axis=0))
        i_mirror = np.where(np.diff(np.sign(np.gradient(r))))[0]
        if i_mirror.shape[0] > 0:
            i_mirror = i_mirror[0] + 1
        else:
            # If the first method fails, just set it at the midpoint
            i_mirror = int(strand.n_s / 2) if strand.n_s % 2 == 0 else int((strand.n_s - 1) / 2)
        velocity[:, i_mirror:] = -velocity[:, i_mirror:]
        return velocity
