"""
Interface between loop object and ebtel++ simulation
"""

import os
import logging
import copy

import numpy as np
from scipy.interpolate import splprep,splev
import astropy.units as u

from synthesizAR.util import InputHandler,OutputHandler


class EbtelInterface(object):
    """
    Interface between field/loop model for the EBTEL model

    Parameters
    ----------
    base_config : `dict`
        Config dictionary with default parameters for all loops.
    heating_model
    """

    def __init__(self,base_config,heating_model,dt=None,ds=None):
        """
        Create EBTEL interface
        """
        self.logger = logging.getLogger(name=type(self).__name__)
        self.base_config = base_config
        self.heating_model = heating_model
        self.heating_model.base_config = base_config
        if dt is None:
            self.global_time = None
            self.logger.warning('Global time not set. Evolution of loops may not be synchronized. Set global time before importing loops to change this.')
        else:
            self.global_time = np.linspace(0.0,self.base_config['total_time'],
                int(np.ceil(self.base_config['total_time']/dt)))*u.s
        self.ds = ds
        if self.ds is None:
            self.logger.warning('Interpolated loop spacing set to None. You will not be able to load loop results until this is set.')


    def configure_input(self,loop,parent_config_dir,parent_results_dir):
        """
        Configure EBTEL input for a given loop object.

        Parameters
        ----------
        loop
        parent_config_dir : `string`
        parent_results_dir : `string`
        """
        oh = OutputHandler(os.path.join(parent_config_dir,loop.name+'.xml'), copy.deepcopy(self.base_config))
        oh.output_dict['output_filename'] = os.path.join(parent_results_dir,loop.name)
        oh.output_dict['loop_length'] = loop.full_length.value/2.0
        event_properties = self.heating_model.calculate_event_properties(loop)
        events = []
        for i in range(self.heating_model.number_events):
            events.append({'event':{
            'magnitude':event_properties['magnitude'][i],
            'rise_start':event_properties['rise_start'][i],
            'rise_end':event_properties['rise_end'][i],
            'decay_start':event_properties['decay_start'][i],
            'decay_end':event_properties['decay_end'][i]}})
        oh.output_dict['heating']['events'] = events
        oh.print_to_xml()
        oh.output_dict['config_filename'] = oh.output_filename
        loop.hydro_configuration = oh.output_dict


    def load_results(self,loop):
        """
        Load EBTEL output for a given loop object.

        Parameters
        ----------
        loop
        """
        if self.ds is not None:
            #interpolate loop lengths to higher resolution with a B-spline
            N_interp = int(np.ceil(loop.full_length/self.ds.to(loop.full_length.unit)))
            nots,_ = splprep(loop.coordinates.value.T)
            _tmp = splev(np.linspace(0,1,N_interp),nots)
            loop.coordinates = [(x,y,z) for x,y,z in zip(_tmp[0],_tmp[1],_tmp[2])]*loop.coordinates.unit

        #load in data and interpolate to universal time
        N_s = len(loop.field_aligned_coordinate)
        _tmp = np.loadtxt(loop.hydro_configuration['output_filename'])

        if self.global_time is not None:
            loop.time = self.global_time
        else:
            loop.time = _tmp[:,0]

        loop.temperature = np.outer(np.interp(loop.time, _tmp[:,0], _tmp[:,1]), np.ones(N_s))*u.K
        loop.density = np.outer(np.interp(loop.time, _tmp[:,0], _tmp[:,3]), np.ones(N_s))*(u.cm**(-3))
