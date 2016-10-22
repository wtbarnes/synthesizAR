"""
Class for an individual loop structure that is part of a larger active region.
"""

import numpy as np
import astropy.units as u
import h5py

class Loop(object):
    """
    Coronal loop object for easily handling all of the properties associated with a loop in an active region.

    Parameters
    ----------
    Attributes
    ----------
    Notes
    -----
    """

    def __init__(self,name,coordinates,field_strength):
        """
        Constructor
        """
        #set unique label for loop object
        self.name = name
        #Load in cartesian coordinates, assign units as centimeters
        self.coordinates = coordinates*u.cm
        #Load in field strength along the field line; convert from Tesla to Gauss
        self.field_strength = (np.array(field_strength)*u.T).to(u.Gauss)


    @property
    def field_aligned_coordinate(self):
        """
        Field-aligned coordinate s. This will have the same units the original coordinates.
        """
        return np.append(0., np.linalg.norm(np.diff(self.coordinates.value,axis=0), axis=1).cumsum())*self.coordinates.unit


    @property
    def full_length(self):
        """
        Loop full-length 2L. This will have the same units as the original coordinates.
        """
        return np.sum(np.linalg.norm(np.diff(self.coordinates.value,axis=0),axis=1))*self.coordinates.unit

    def get_emissivity(self,wavelength):
        """
        Get the calculated emissivity that was either set or saved to disk
        """
        if self.emissivity_savefile:
            with h5py.File(self.emissivity_savefile,'r') as hf:
                dset = hf.get(wavelength)
                emiss = np.array(dset)*u.Unit(dset.attr['units'])
        else:
            emiss = self.emissivity

        return emiss
