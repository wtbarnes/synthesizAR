"""
Class for an individual loop structure that is part of a larger active region.
"""

import os

import numpy as np
import astropy.units as u
import h5py


class Loop(object):
    """
    Container for geometric and thermodynamic properties of a coronal loop

    Parameters
    ----------
    name : `str`
    coordinates : `astropy.Quantity`
        HEEQ Cartesian coordinates of the loop
    field_strength : `astropy.Quantity`
        Scalar magnetic field strength along the loop

    Examples
    --------
    >>> import astropy.units as u
    >>> import synthesizAR
    >>> coordinate = u.Quantity([[1,2,3],[4,5,6]], 'Mm')
    >>> field_strength = u.Quantity([100,200], 'gauss')
    >>> loop = synthesizAR.Loop('coronal_loop', coordinate, field_strength)
    >>> loop
    Name : coronal_loop
    Loop full-length, 2L : 5.196 Mm
    Footpoints : (1e+08,2e+08,3e+08),(4e+08,5e+08,6e+08) cm
    Maximum field strength : 200.00 G
    """

    @u.quantity_input
    def __init__(self, name, coordinates: u.cm, field_strength: u.gauss):
        self.name = name
        self.coordinates = coordinates.to(u.cm)
        self.field_strength = field_strength.to(u.gauss)

    def __repr__(self):
        fp0 = ','.join(['{:.3g}'.format(l) for l in self.coordinates[0, :].value])
        fp1 = ','.join(['{:.3g}'.format(l) for l in self.coordinates[-1, :].value])
        return f'''Name : {self.name}
Loop full-length, 2L : {self.full_length.to(u.Mm):.3f}
Footpoints : ({fp0}),({fp1}) {self.coordinates.unit.to_string()}
Maximum field strength : {np.max(self.field_strength):.2f}'''

    @property
    def field_aligned_coordinate(self):
        """
        Field-aligned coordinate :math:`s` such that :math:`0<s<L`
        """
        return np.append(0., np.linalg.norm(np.diff(self.coordinates.value, axis=0),
                                            axis=1).cumsum()) * self.coordinates.unit

    @property
    def full_length(self):
        """
        Loop full-length :math:`2L`, from footpoint to footpoint
        """
        return np.sum(np.linalg.norm(np.diff(self.coordinates.value, axis=0),
                                     axis=1)) * self.coordinates.unit

    @property
    def time(self):
        """
        Simulation time
        """
        with h5py.File(self.parameters_savefile, 'r') as hf:
            dset = hf['/'.join([self.name, 'time'])]
            time = u.Quantity(dset, dset.attrs['units'])
        return time

    @property
    def electron_temperature(self):
        """
        Loop electron temperature as function of coordinate and time.
        """
        with h5py.File(self.parameters_savefile, 'r') as hf:
            dset = hf['/'.join([self.name, 'electron_temperature'])]
            temperature = u.Quantity(dset, dset.attrs['units'])
        return temperature

    @property
    def ion_temperature(self):
        """
        Loop ion temperature as function of coordinate and time.
        """
        with h5py.File(self.parameters_savefile, 'r') as hf:
            dset = hf['/'.join([self.name, 'ion_temperature'])]
            temperature = u.Quantity(dset, dset.attrs['units'])
        return temperature

    @property
    def density(self):
        """
        Loop density as a function of coordinate and time.
        """
        with h5py.File(self.parameters_savefile, 'r') as hf:
            dset = hf['/'.join([self.name, 'density'])]
            density = u.Quantity(dset, dset.attrs['units'])
        return density

    @property
    def velocity(self):
        """
        Velcoity in the field-aligned direction of the loop as a function of loop coordinate and
        time.
        """
        with h5py.File(self.parameters_savefile, 'r') as hf:
            dset = hf['/'.join([self.name, 'velocity'])]
            velocity = u.Quantity(dset, dset.attrs['units'])
        return velocity

    @property
    def velocity_x(self):
        """
        X-component of velocity in the HEEQ Cartesian coordinate system as a function of time.
        """
        with h5py.File(self.parameters_savefile, 'r') as hf:
            dset = hf['/'.join([self.name, 'velocity_x'])]
            velocity = u.Quantity(dset, dset.attrs['units'])
        return velocity

    @property
    def velocity_y(self):
        """
        Y-component of velocity in the HEEQ Cartesian coordinate system as a function of time.
        """
        with h5py.File(self.parameters_savefile, 'r') as hf:
            dset = hf['/'.join([self.name, 'velocity_y'])]
            velocity = u.Quantity(dset, dset.attrs['units'])
        return velocity

    @property
    def velocity_z(self):
        """
        Z-component of velocity in the HEEQ Cartesian coordinate system as a function of time.
        """
        with h5py.File(self.parameters_savefile, 'r') as hf:
            dset = hf['/'.join([self.name, 'velocity_z'])]
            velocity = u.Quantity(dset, dset.attrs['units'])
        return velocity
