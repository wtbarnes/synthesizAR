"""
Interface between loop object and ebtel++ simulation
"""

import os
import logging
import copy
import warnings

import numpy as np
import h5py
import astropy.units as u
try:
    import dask
except ImportError:
    warnings.warn('''Dask library not found. You will not be able to compute nonequilibrium ion
                     populations for EBTEL simulations''')

from synthesizAR.util import InputHandler, OutputHandler
from synthesizAR.atomic import Element


class EbtelInterface(object):
    """
    Interface to the EBTEL model

    Interface between synthesizAR and the Enthalpy-Based Thermal Evolution of Loops code for
    computing time-dependent solutions of spatially-averaged loops.

    Parameters
    ----------
    base_config : `dict`
        Config dictionary with default parameters for all loops.
    heating_model : object
        Heating model class for configuring event times and rates
    parent_config_dir : `str`
        Path to configuration file directory
    parent_results_dir : `str`
        Path to results file directory
    """

    def __init__(self, base_config, heating_model, parent_config_dir, parent_results_dir):
        """
        Create EBTEL interface
        """
        self.logger = logging.getLogger(name=type(self).__name__)
        self.name = 'EBTEL'
        self.base_config = base_config
        self.heating_model = heating_model
        self.heating_model.base_config = base_config
        self.parent_config_dir = parent_config_dir
        self.parent_results_dir = parent_results_dir

    def configure_input(self, loop):
        """
        Configure EBTEL input for a given loop object.
        """
        oh = OutputHandler(os.path.join(self.parent_config_dir, loop.name+'.xml'),
                           copy.deepcopy(self.base_config))
        oh.output_dict['output_filename'] = os.path.join(self.parent_results_dir, loop.name)
        oh.output_dict['loop_length'] = loop.full_length.value/2.0
        event_properties = self.heating_model.calculate_event_properties(loop)
        events = []
        for i in range(self.heating_model.number_events):
            events.append({'event': {
                                    'magnitude': event_properties['magnitude'][i],
                                    'rise_start': event_properties['rise_start'][i],
                                    'rise_end': event_properties['rise_end'][i],
                                    'decay_start': event_properties['decay_start'][i],
                                    'decay_end': event_properties['decay_end'][i]}})
        oh.output_dict['heating']['events'] = events
        oh.print_to_xml()
        oh.output_dict['config_filename'] = oh.output_filename
        loop.hydro_configuration = oh.output_dict

    def load_results(self, loop):
        """
        Load EBTEL output for a given loop object.

        Parameters
        ----------
        loop : `synthesizAR.Loop` object
        """
        # load text
        N_s = loop.field_aligned_coordinate.shape[0]
        _tmp = np.loadtxt(loop.hydro_configuration['output_filename'])

        # reshape into a 1D loop structure with units
        time = _tmp[:, 0]*u.s
        electron_temperature = np.outer(_tmp[:, 1], np.ones(N_s))*u.K
        ion_temperature = np.outer(_tmp[:, 2], np.ones(N_s))*u.K
        density = np.outer(_tmp[:, 3], np.ones(N_s))*(u.cm**(-3))
        velocity = np.outer(_tmp[:, -2], np.ones(N_s))*u.cm/u.s
        # flip sign of velocity where the radial distance from center is maximum
        # FIXME: this is probably not the best way to do this...
        r = np.sqrt(np.sum(loop.coordinates.value**2, axis=1))
        i_mirror = np.where(np.diff(np.sign(np.gradient(r))))[0]
        if i_mirror.shape[0] > 0:
            i_mirror = i_mirror[0] + 1
        else:
            # If the first method fails, just set it at the midpoint
            i_mirror = int(N_s / 2) if N_s % 2 == 0 else int((N_s - 1) / 2)
        velocity[:, i_mirror:] = -velocity[:, i_mirror:]

        return time, electron_temperature, ion_temperature, density, velocity

    @staticmethod
    def calculate_ionization_fraction(field, emission_model, **kwargs):
        """
        Solve the time-dependent ionization balance equation for a particular loop and ion.
        """
        tmpdir = os.path.join(os.path.dirname(emission_model.ionization_fraction_savefile),
                              'tmp_nei')
        unique_elements = list(set([ion.element_name for ion in emission_model]))
        temperature = kwargs.get('temperature', emission_model.temperature)

        # Task wrappers
        @dask.delayed
        def compute_rate_matrix(element):
            return element._rate_matrix()
 
        @dask.delayed
        def compute_ionization_equilibrium(element, rate_matrix):
            return element.equilibrium_ionization(rate_matrix=rate_matrix)
        
        @dask.delayed
        def compute_and_save_nei(loop, element, rate_matrix, initial_condition, save_root_path):
            y_nei = element.non_equilibrium_ionization(loop.time, loop.electron_temperature[:, 0],
                                                       loop.density[:, 0], rate_matrix=rate_matrix,
                                                       initial_condition=initial_condition)
            save_path = os.path.join(save_root_path, f'{element.element_name}_{loop.name}.npy')
            np.save(save_path, y_nei.value)
            return save_path, loop.field_aligned_coordinate.shape[0]
        
        @dask.delayed
        def slice_and_store(nei_matrices):
            with h5py.File(emission_model.ionization_fraction_savefile, 'a') as hf:
                for fn, n_s in nei_matrices:
                    element_name, loop_name = os.path.splitext(os.path.basename(fn))[0].split('_')
                    grp = hf.create_group(loop_name) if loop_name not in hf else hf[loop_name]
                    y_nei = np.load(fn)
                    data = np.repeat(y_nei[:, np.newaxis, :], n_s, axis=1)
                    if element_name not in grp:
                        dset = grp.create_dataset(element_name, data=data)
                    else:
                        dset = grp[element_name]
                        dset[:, :, :] = data
                    dset.attrs['units'] = ''
                    dset.attrs['description'] = 'non-equilibrium ionization fractions'
                    os.remove(fn)
            os.rmdir(tmpdir)
        
        # Build task list
        tasks = []
        for el_name in unique_elements:
            element = Element(el_name, temperature)
            rate_matrix = compute_rate_matrix(element)
            initial_condition = compute_ionization_equilibrium(element, rate_matrix)
            for loop in field.loops:
                tasks.append(compute_and_save_nei(loop, element, rate_matrix, initial_condition,
                                                  tmpdir))

        # Execute tasks and compile to single file
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)
        return slice_and_store(tasks)
