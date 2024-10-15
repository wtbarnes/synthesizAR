"""
Interface between loop object and ebtel++ simulation
"""
import astropy.units as u
import ebtelplusplus
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
    heating_model : `ebtelplusplus.models.HeatingModel`
        A model that specifies the background
    physics : `ebtelplusplus.models.PhysicsModel`
    solver : `ebtelplusplus.models.SolverModel`
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
        self.physics = physics_model
        self.solver = solver_model

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
                                    physics=self.physics,
                                    solver=self.solver,
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
