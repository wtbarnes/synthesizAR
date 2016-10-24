"""
Create data products loop simulations
"""

import os

import numpy as np
import astropy.units as u


class Observer(object):
    """
    Class for assembling AR from loops and creating data products from 2D
    projections.

    Parameters
    ----------
    Examples
    --------
    Notes
    -----
    """

    def __init__(self):
        """
        Constructor
        """
        #set global time here based on instrument cadence
        pass


    def _resample_loop(self,loop,ds=None,global_time=None):
        """
        Resample loops in space and align in time. Only for the
        global AR picture

        Parameters
        ----------
        field : synthesizAR.Skeleton
        ds :
            Resolution to resample all loops at; should be much less than the
            instrument resolution
        global_time : array-like
            Aligned time vector for all loops; should be at instrument
            cadence.
        """
        new_coordinates,new_time = loop.coordinates.copy(),loop.time.copy()
        if ds is not None:
            #interpolate loop lengths to higher resolution with a B-spline
            N_interp = int(np.ceil(loop.full_length/self.ds.to(loop.full_length.unit)))
            nots,_ = splprep(loop.coordinates.value.T)
            _tmp = splev(np.linspace(0,1,N_interp),nots)
            new_coordinates = [(x,y,z) for x,y,z in zip(_tmp[0],_tmp[1],_tmp[2])]*loop.coordinates.unit
        if self.global_time is not None:
            new_time = self.global_time
        else:
            self.logger.warning('Global time not set. Evolution of loops may not be synchronized. Make sure instrument cadence is set.')

        if ds is not None or global_time is not None:
            #interpolate density and temperature in 2D
            pass
