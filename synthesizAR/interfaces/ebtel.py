"""
Interface between loop object and ebtel++ simulation
"""

import os
import logging
import copy
import warnings
import itertools

import numpy as np
import h5py
import astropy.units as u
try:
    import dask
    import distributed
except ImportError:
    warnings.warn('Dask library required for NEI calculation')

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

        Build a Dask task graph to solve the time-dependent ion population equations for all
        loops in the field and all elements in our emission model.
        """
        tmpdir = os.path.join(os.path.dirname(emission_model.ionization_fraction_savefile),
                              'tmp_nei')
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)
        # Create lock for writing HDF5 file
        lock = distributed.Lock()
        unique_elements = list(set([ion.element_name for ion in emission_model]))
        temperature = kwargs.get('temperature', emission_model.temperature)
   
        tasks = {}
        for el_name in unique_elements:
            el = Element(el_name, temperature)
            rate_matrix = dask.delayed(el._rate_matrix)()
            ioneq = dask.delayed(el.equilibrium_ionization)(rate_matrix)
            _tasks = []
            for loop in field.loops:
                _tasks.append(dask.delayed(EbtelInterface.compute_and_save_nei)(
                    el, loop, rate_matrix, ioneq, tmpdir))
            tasks[f'{el.element_name}'] = dask.delayed(EbtelInterface.slice_and_store)(
                _tasks, emission_model.ionization_fraction_savefile, lock)

        return tasks

    @staticmethod
    def compute_and_save_nei(element, loop, rate_matrix, initial_condition, save_path_root):
        """
        Compute and save NEI populations for a given element and loop
        """
        y_nei = element.non_equilibrium_ionization(loop.time, loop.electron_temperature[:, 0],
                                                   loop.density[:, 0], rate_matrix,
                                                   initial_condition)
        save_path = os.path.join(save_path_root, f'{element.element_name}_{loop.name}.npz')
        np.savez(save_path, array=y_nei.value, n_s=loop.field_aligned_coordinate.shape[0],
                 element=element.element_name, loop=loop.name)
        return save_path

    @staticmethod
    def slice_and_store(filenames, savefile, lock):
        """
        Collecting and storing all NEI populations in a single HDF5 file
        """
        with lock:
            with h5py.File(savefile, 'a') as hf:
                for fn in filenames:
                    tmp = np.load(fn)
                    element_name, loop_name = str(tmp['element']), str(tmp['loop'])
                    grp = hf.create_group(loop_name) if loop_name not in hf else hf[loop_name]
                    y_nei, n_s = tmp['array'], int(tmp['n_s'])
                    data = np.repeat(y_nei[:, np.newaxis, :], n_s, axis=1)
                    if element_name not in grp:
                        dset = grp.create_dataset(element_name, data=data)
                    else:
                        dset = grp[element_name]
                        dset[:, :, :] = data
                    dset.attrs['units'] = ''
                    dset.attrs['description'] = 'non-equilibrium ionization fractions'

        return filenames

    @staticmethod
    def _cleanup(filenames):
        for f in itertools.chain.from_iterable(filenames):
            os.remove(f)
        os.rmdir(os.path.dirname(f))
