"""
Class for an individual loop structure that is part of a larger active region.
"""

import os

import numpy as np
import astropy.units as u
import h5py


class Loop(object):
    """
    Coronal loop object for easily handling all of the properties associated with a loop in
    an active region.

    Parameters
    ----------
    Attributes
    ----------
    Notes
    -----
    """


    def __init__(self,name,coordinates,field_strength):
        #set unique label for loop object
        self.name = name
        #Load in cartesian coordinates, assign units as centimeters
        self.coordinates = coordinates*u.cm
        #Load in field strength along the field line; convert from Tesla to Gauss
        self.field_strength = (np.array(field_strength)*u.T).to(u.Gauss)

    def __repr__(self):
        return '''
        Name : {name}
        Loop full-length, 2L : {loop_length:.3f}
        Footpoints : ({fp0}),({fp1}) {fpu}
        maximum field strength : {max_b:.2f}
        '''.format(name=self.name,
                   loop_length=self.full_length.to(u.Mm),
                   fp0=','.join(['{:.3g}'.format(l) for l in self.coordinates[0,:].value]),
                   fp1=','.join(['{:.3g}'.format(l) for l in self.coordinates[-1,:].value]),
                   fpu=self.coordinates.unit.to_string(),
                   max_b=np.max(self.field_strength))

    @property
    def field_aligned_coordinate(self):
        """
        Field-aligned coordinate :math:`s`. This will have the same units the original coordinates.
        """
        return np.append(0., np.linalg.norm(np.diff(self.coordinates.value,axis=0),
                                            axis=1).cumsum())*self.coordinates.unit

    @property
    def full_length(self):
        """
        Loop full-length :math:`2L`. This will have the same units as the original coordinates.
        """
        return np.sum(np.linalg.norm(np.diff(self.coordinates.value,axis=0),
                                    axis=1))*self.coordinates.unit

    @property
    def temperature(self):
        """
        Loop temperature as function of coordinate and time. Can be stored in memory or pulled from
        an HDF5 file.
        """
        if hasattr(self,'parameters_savefile'):
            with h5py.File(self.parameters_savefile,'r') as hf:
                dset = hf[os.path.join(self.name,'temperature')]
                temperature = np.array(dset)*u.Unit(dset.attrs['units'])
            return temperature
        else:
            return self._temperature

    @property
    def density(self):
        """
        Loop density as a function of coordinate and time. Can be stored in memory or pulled from an
        HDF5 file.
        """
        if hasattr(self,'parameters_savefile'):
            with h5py.File(self.parameters_savefile,'r') as hf:
                dset = hf[os.path.join(self.name,'density')]
                density = np.array(dset)*u.Unit(dset.attrs['units'])
            return density
        else:
            return self._density

    @property
    def velocity(self):
        """
        Velcoity in the field-aligned direction of the loop as a function of loop coordinate and
        time. Can be stored in memory or pulled from an HDF5 file.
        """
        if hasattr(self,'parameters_savefile'):
            with h5py.File(self.parameters_savefile,'r') as hf:
                dset = hf[os.path.join(self.name,'velocity')]
                velocity = np.array(dset)*u.Unit(dset.attrs['units'])
            return velocity
        else:
            return self._velocity

    @u.quantity_input(wavelength=u.angstrom)
    def get_emission(self,wavelength):
        """
        Get the calculated emission (energy per unit volume per unit time per unit solid angle) for
        a particular wavelength. Can be stored in memory or pulled from an HDF5 file.
        """
        if hasattr(self,'emission_savefile'):
            with h5py.File(self.emission_savefile,'r') as hf:
                dset = hf[os.path.join(self.name,str(wavelength.to(u.angstrom).value))]
                emiss = np.array(dset)*u.Unit(dset.attrs['units'])
        else:
            emiss = self.emission[wavelength]

        return emiss
