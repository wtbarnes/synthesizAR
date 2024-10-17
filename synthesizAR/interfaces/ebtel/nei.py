"""
Functions for calculating time-dependent ionization fractions from an EBTEL simulation
"""
import distributed
import h5py
import numpy as np
import toolz

from fiasco import Element

from synthesizAR.atomic import non_equilibrium_ionization

__all__ = ['calculate_ionization_fraction', 'compute_nei', 'write_to_hdf5']


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
    temperature : `~astropy.units.Quantity`, optional
        If not specified, defaults to the temperature array of ``emission_model``

    Returns
    --------
    future : `dict` of lists of `~distributed.client.Future`
        Asynchronous computation of time-dependent ionization per element per strand.
    """
    client = distributed.get_client()
    unique_elements = list(set([ion.element_name for ion in emission_model]))
    temperature = kwargs.get('temperature', emission_model.temperature)

    futures = {}
    for el_name in unique_elements:
        el = Element(el_name, temperature)
        partial_nei = toolz.curry(compute_nei)(el)
        partial_write = toolz.curry(write_to_hdf5)(
            element_name=el_name,
            savefile=emission_model.ionization_fraction_savefile
        )
        y = client.map(partial_nei, skeleton.strands, pure=False)
        write_y = client.map(partial_write, y, skeleton.strands, pure=False)
        distributed.client.wait(write_y)
        futures[el_name] = write_y

    return futures


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
