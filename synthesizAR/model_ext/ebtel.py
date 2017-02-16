"""
Interface between loop object and ebtel++ simulation
"""

import os
import itertools
import logging
import copy

import numpy as np
import astropy.units as u

from synthesizAR.util import InputHandler,OutputHandler
from synthesizAR.atomic import get_ion_data,solve_nei_populations


class EbtelInterface(object):
    """
    Interface between field/loop model for the EBTEL model

    Parameters
    ----------
    base_config : `dict`
        Config dictionary with default parameters for all loops.
    heating_model
    """


    def __init__(self,base_config,heating_model):
        """
        Create EBTEL interface
        """
        self.logger = logging.getLogger(name=type(self).__name__)
        self.name = 'EBTEL'
        self.base_config = base_config
        self.heating_model = heating_model
        self.heating_model.base_config = base_config

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
        loop : `synthesizAR.Loop` object
        """
        # load text
        N_s = len(loop.field_aligned_coordinate)
        _tmp = np.loadtxt(loop.hydro_configuration['output_filename'])

        # reshape into a 1D loop structure with units
        time = _tmp[:,0]*u.s
        temperature = np.outer(_tmp[:,1],np.ones(N_s))*u.K
        density = np.outer(_tmp[:,3],np.ones(N_s))*(u.cm**(-3))
        velocity = np.outer(_tmp[:,-2],np.ones(N_s))*u.cm/u.s
        # flip sign of velocity at apex
        i_mirror = np.where(np.diff(loop.coordinates.value[:,2])>0)[0][-1] + 2
        velocity[:,i_mirror:] = -velocity[:,i_mirror:]

        return time,temperature,density,velocity

    def get_fractional_ionization(self,ion_list,loop,**kwargs):
        """
        Solve the ionization balance equation for a particular loop and ion.
        """
        ion_data_options = kwargs.get('ion_data_options',{})
        nei_solver_options = kwargs.get('nei_solver_options',{})

        fractional_ionization = {}
        #group ions by element and remove any duplicates
        grouped_ions = {key:list(set(sorted([g[1] for g in group]))) \
                        for key,group in itertools.groupby(sorted(ion_list),lambda x:x[0])}
        #iterate over elements
        for element in grouped_ions:
            #only get data once for each element
            if not hasattr(self,'_rate_data'):
                self._rate_data = {}
            if element not in self._rate_data:
                self._rate_data[element] = {}
                self.logger.info('Retrieving rate information for {}'.format(element))
                irate,rrate,eq_pop,temperature = get_ion_data(element,
                                                            zrange=[np.min(grouped_ions[element]),
                                                                    np.max(grouped_ions[element])],
                                                            **ion_data_options)
                self._rate_data[element]['ionization_rate'] = irate
                self._rate_data[element]['recombination_rate'] = rrate
                self._rate_data[element]['equilibrium_populations'] = eq_pop
                self._rate_data[element]['temperature'] = temperature

            #calculate the NEI populations
            self.logger.debug('Calculating NEI populations for {}'.format(element))
            nei_populations = solve_nei_populations(loop.time.value,
                                                loop.temperature.value[:,0],
                                                loop.density.value[:,0],
                                                self._rate_data[element]['ionization_rate'],
                                                self._rate_data[element]['recombination_rate'],
                                                self._rate_data[element['equilibrium_populations']],self._rate_data[element]['temperature'],
                                                **nei_solver_options)
            for ion in grouped_ions[element]:
                ion_index = ion - np.min(grouped_ions[element])
                fractional_ionization['{}_{}'.format(element,ion)] = np.repeat(
                                                            nei_populations[:,ion_index,np.newaxis],
                                                            loop.temperature.shape[1],axis=1)

        return fractional_ionization
