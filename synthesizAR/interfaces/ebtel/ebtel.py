"""
Interface between loop object and ebtel++ simulation
"""
import astropy.units as u
import copy
import h5py
import numpy as np
import toolz
import warnings

from fiasco import Element

try:
    import distributed
except ImportError:
    warnings.warn('Dask library required for NEI calculation')

from synthesizAR.atomic import non_equilibrium_ionization

from .util import run_ebtel


class EbtelInterface:
    """
    Interface to the Enthalpy-Based Thermal Evolution of Loops (EBTEL) model

    Parameters
    ----------
    base_config : `dict`
        Config dictionary with default parameters for all loops.
    heating_model : object
        Heating model class for configuring event times and rates
    config_dir : `str`
        Path to configuration file directory
    results_dir : `str`
        Path to results file directory
    """
    name = 'EBTEL'

    def __init__(self, base_config, heating_model, ebtel_dir):
        """
        Create EBTEL interface
        """
        self.base_config = base_config
        self.heating_model = heating_model
        self.ebtel_dir = ebtel_dir

    def load_results(self, loop):
        """
        Load EBTEL output for a given loop object.

        Parameters
        ----------
        loop : `synthesizAR.Strand` object
        """
        # Configure run
        output_dict = copy.deepcopy(self.base_config)
        output_dict['loop_length'] = loop.length.to(u.cm).value / 2.0
        event_properties = self.heating_model.calculate_event_properties(loop)
        events = []
        for i in range(event_properties['magnitude'].shape[0]):
            events.append({'event': {'magnitude': event_properties['magnitude'][i],
                                     'rise_start': event_properties['rise_start'][i],
                                     'rise_end': event_properties['rise_end'][i],
                                     'decay_start': event_properties['decay_start'][i],
                                     'decay_end': event_properties['decay_end'][i]}})
        output_dict['heating']['events'] = events
        # Run model
        _tmp = run_ebtel(output_dict, self.ebtel_dir)

        # reshape into a 1D loop structure with units
        N_s = loop.field_aligned_coordinate_center.shape[0]
        time = _tmp['time']*u.s
        electron_temperature = np.outer(_tmp['electron_temperature'], np.ones(N_s))*u.K
        ion_temperature = np.outer(_tmp['ion_temperature'], np.ones(N_s))*u.K
        density = np.outer(_tmp['density'], np.ones(N_s))*(u.cm**(-3))
        velocity = np.outer(_tmp['velocity'], np.ones(N_s))*u.cm/u.s
        # flip sign of velocity where the radial distance from center is maximum
        # FIXME: this is probably not the best way to do this...
        r = np.sqrt(np.sum(loop.coordinate_center.cartesian.xyz.value**2, axis=0))
        i_mirror = np.where(np.diff(np.sign(np.gradient(r))))[0]
        if i_mirror.shape[0] > 0:
            i_mirror = i_mirror[0] + 1
        else:
            # If the first method fails, just set it at the midpoint
            i_mirror = int(N_s / 2) if N_s % 2 == 0 else int((N_s - 1) / 2)
        velocity[:, i_mirror:] = -velocity[:, i_mirror:]

        return time, electron_temperature, ion_temperature, density, velocity

    @staticmethod
    def calculate_ionization_fraction(skeleton, emission_model, **kwargs):
        """
        Solve the time-dependent ionization balance equation for all loops and all elements

        This method computes the time dependent ion population fractions for each element in
        the emission model and each loop in the active region and compiles the results to a single
        HDF5 file. To do this efficiently, it uses the dask.distributed library to take advantage of
        multiple processes/cores/machines and compute the population fractions in parallel. It returns
        an asynchronous `~distributed.client.Future` object which holds the state of the submitted
        tasks.

        Parameters
        ----------
        skeleton : `~synthesizAR.Skeleton`
        emission_model : `~synthesizAR.atomic.EmissionModel`

        Other Parameters
        ---------------------
        temperature : `~astropy.units.Quantity`

        Returns
        --------
        future : `~distributed.client.Future`
        """
        client = distributed.get_client()
        unique_elements = list(set([ion.element_name for ion in emission_model]))
        temperature = kwargs.get('temperature', emission_model.temperature)

        futures = {}
        for el_name in unique_elements:
            el = Element(el_name, temperature)
            partial_nei = toolz.curry(EbtelInterface.compute_nei)(el)
            partial_write = toolz.curry(EbtelInterface.write_to_hdf5)(
                element_name=el_name, savefile=emission_model.ionization_fraction_savefile)
            y = client.map(partial_nei, skeleton.strands, pure=False)
            write_y = client.map(partial_write, y, skeleton.strands, pure=False)
            distributed.client.wait(write_y)
            futures[el_name] = write_y

        return futures

    @staticmethod
    def compute_nei(element, loop):
        """
        Compute NEI populations for a given element and loop
        """
        y = non_equilibrium_ionization(element,
                                       loop.time,
                                       loop.electron_temperature[:, 0],
                                       loop.density[:, 0])
        # Fake a spatial axis by tiling the same result at each s coordinate
        return np.repeat(y.value[:, np.newaxis, :], loop.field_aligned_coordinate.shape[0], axis=1)

    @staticmethod
    def write_to_hdf5(data, loop, element_name, savefile):
        """
        Collect and store all NEI populations in a single HDF5 file
        """
        lock = distributed.Lock('hdf5_ebtel_nei')
        with lock:
            with h5py.File(savefile, 'a') as hf:
                grp = hf.create_group(loop.name) if loop.name not in hf else hf[loop.name]
                if element_name not in grp:
                    dset = grp.create_dataset(element_name, data=data)
                else:
                    dset = grp[element_name]
                    dset[:, :, :] = data
                dset.attrs['unit'] = ''
                dset.attrs['description'] = 'non-equilibrium ionization fractions'
