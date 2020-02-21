"""
Class for the SDO/AIA instrument. Holds information about the cadence and
spatial and spectroscopic resolution.
"""

import pkg_resources

import numpy as np
import zarr
import asdf
import astropy.units as u
from sunpy.map import Map
from aiapy.response import Channel

from synthesizAR.util import is_visible
from synthesizAR.instruments import InstrumentBase

with asdf.open(pkg_resources.resource_filename('synthesizAR', 'instruments/aia_temperature_response.asdf'), 'r') as af:
    _TEMPERATURE_RESPONSE = af.tree


class InstrumentSDOAIA(InstrumentBase):
    """
    Instrument object for the Atmospheric Imaging Assembly on the Solar Dynamics Observatory

    Parameters
    ----------
    observing_time : `tuple`
        start and end of observing time
    observer : `~astropy.coordinates.SkyCoord`
    apply_psf : `bool`, optional
        If True (default), apply AIA point-spread function to images

    Examples
    --------
    """

    def __init__(self, observing_time, observer, pad_fov=None):
        super().__init__(observing_time, observer)
        self.telescope = 'SDO/AIA'
        self.detector = 'AIA'
        self.name = 'SDO_AIA'
        self.channels = [
            Channel(94*u.angstrom),
            Channel(131*u.angstrom),
            Channel(171*u.angstrom),
            Channel(193*u.angstrom),
            Channel(211*u.angstrom),
            Channel(335*u.angstrom),
        ]
        self.cadence = 12.0*u.s
        self.resolution = [0.600698, 0.600698]*u.arcsec/u.pixel
        self.pad_fov = pad_fov

    def convolve_with_psf(self, data, channel):
        # TODO: do the convolution here!
        return data

    @staticmethod
    def calculate_intensity_kernel(loop, channel, **kwargs):
        # This method should be implemented for each specific instrument
        #
        # Compute the kernel of the LOS intensity integral for a given timestep
        # for every grid cell in the magnetic skeleton
        # If the original model results, interpolated to the field-line resolution,
        # are stored in a Zarr file, this is trivially parallelized over loop
        # Minor hangup: need to interpolate to the observing time such that we need to inerpolate
        # our n_s-by-n_t array over the time-axis
        #
        em_model = kwargs.get('emission_model', None)
        if em_model:
            raise NotImplementedError('Full intensity calculation not yet implemented.')
        else:
            T, K = _TEMPERATURE_RESPONSE['temperature'], _TEMPERATURE_RESPONSE[channel.name]
            K_interp = np.interp(loop.electron_temperature, T, K)
            kernel = K_interp * loop.density**2
        return kernel

    def integrate_los(self, time, channel, skeleton):
        # Get Coordinates
        coords = skeleton.all_coordinates_centers.transform_to(self.projected_frame)
        # Compute weights
        i_time = np.where(time == self.observing_time)[0][0]
        widths = np.concatenate([l.field_aligned_coordinate_width for l in skeleton.loops])
        root = zarr.open(skeleton.loops[0].model_results_filename, 'r')
        kernels = np.concatenate([root[f'{l.name}/{self.name}/{channel.name}'][i_time,:]
                                  for l in skeleton.loops])
        unit_kernel = u.Unit(
            root[f'{skeleton.loops[0].name}/{self.name}/{channel.name}'].attrs['unit'])
        weights = self.cross_section_ratio * widths * (kernels*unit_kernel)
        # TODO: apply is_visible mask!
        # Bin
        bins, (blc, trc) = self.get_detector_array(skeleton.all_coordinates)
        hist, _, _ = np.histogram2d(
            coords.Tx.value,
            coords.Ty.value,
            bins=bins,
            range=((blc.Tx.value, trc.Tx.value), (blc.Ty.value, trc.Ty.value)),
            weights=weights.value,
        )
        header = self.get_header(channel, skeleton.all_coordinates)
        header['bunit'] = weights.unit.decompose().to_string()
        header['date-obs'] = (self.observer.obstime + time).isot

        return Map(hist.T, header)  

# TODO: pull out full emiss calc and then remove these
#@staticmethod
#def flatten_emissivities(channel, emission_model):
#    """
#    Compute product between wavelength response and emissivity for all ions
#    """
#    flattened_emissivities = []
#    for ion in emission_model:
#        wavelength, emissivity = emission_model.get_emissivity(ion)
#        if wavelength is None or emissivity is None:
#            flattened_emissivities.append(None)
#            continue
#        interpolated_response = splev(wavelength.value, channel['wavelength_response_spline'],
#                                        ext=1)
#        em_summed = np.dot(emissivity.value, interpolated_response)
#        unit = emissivity.unit*u.count/u.photon*u.steradian/u.pixel*u.cm**2
#        flattened_emissivities.append(u.Quantity(em_summed, unit))
#
#    return flattened_emissivities
#
#@staticmethod
#def calculate_counts_full(channel, loop, emission_model, flattened_emissivities):
#    """
#    Calculate the AIA intensity using the wavelength response functions and a
#    full emission model.
#    """
#    density = loop.density
#    electron_temperature = loop.electron_temperature
#    counts = np.zeros(electron_temperature.shape)
#    itemperature, idensity = emission_model.interpolate_to_mesh_indices(loop)
#    for ion, flat_emiss in zip(emission_model, flattened_emissivities):
#        if flat_emiss is None:
#            continue
#        ionization_fraction = emission_model.get_ionization_fraction(loop, ion)
#        tmp = np.reshape(map_coordinates(flat_emiss.value, np.vstack([itemperature, idensity])),
#                            electron_temperature.shape)
#        tmp = u.Quantity(np.where(tmp < 0., 0., tmp), flat_emiss.unit)
#        counts_tmp = ion.abundance*0.83/(4*np.pi*u.steradian)*ionization_fraction*density*tmp
#        if not hasattr(counts, 'unit'):
#            counts = counts*counts_tmp.unit
#        counts += counts_tmp
#
#    return counts
