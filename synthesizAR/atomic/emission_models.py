"""
Various models for calculating emission from multiple ions
"""
import json

import numpy as np
from scipy.interpolate import splrep, splev, interp1d
import astropy.units as u
from astropy.utils.console import ProgressBar
import h5py
import fiasco

from .chianti import Ion, Element


class EmissionModel(fiasco.IonCollection):
    """
    Model for how atomic data is used to calculate emission from
    coronal plasma.
    """
    
    @u.quantity_input
    def __init__(self, density: u.cm**(-3), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = self[0].temperature
        self.density = density
        self.resolved_wavelengths = kwargs.get('resolved_wavelengths', {})
        # Cannot have empty abundances so replace them as needed
        default_abundance = kwargs.get('default_abundance_dataset', 'sun_photospheric_2009_asplund')
        for ion in self._ion_list:
            if ion.abundance is None:
                ion._dset_names['abundance_filename'] = default_abundance

    def save(self, savefile):
        """
        Save a JSON representation of the emission model.
        """
        save_dict = {
            'temperature': self.temperature.value.tolist(),
            'temperature_unit': self.temperature.unit.to_string(),
            'density': self.density.value.tolist(),
            'density_unit': self.density.unit.to_string(),
            'ion_list': ['_'.join([ion.ion_name.split()[0].lower(), ion.ion_name.split()[1]]) for ion in self],
            'dset_names': [ion._dset_names for ion in self]
        }
        if hasattr(self, 'emissivity_savefile'):
            save_dict['emissivity_savefile'] = self.emissivity_savefile
        if hasattr(self, 'ionization_fraction_savefile'):
            save_dict['ionization_fraction_savefile'] = self.ionization_fraction_savefile
        with open(savefile, 'w') as f:
            json.dump(save_dict, f, indent=4, sort_keys=True)

    @classmethod
    def restore(cls, savefile):
        """
        Restore the emission model from a JSON representation.
        """
        with open(savefile, 'r') as f:
            restore_dict = json.load(f)
        temperature = u.Quantity(restore_dict['temperature'], restore_dict['temperature_unit'])
        density = u.Quantity(restore_dict['density'], restore_dict['density_unit'])
        ion_list = [Ion(ion, temperature, **ds) for ion, ds in zip(restore_dict['ion_list'],
                                                                   restore_dict['dset_names'])]
        emission_model = cls(density, *ion_list)
        if 'emissivity_savefile' in restore_dict:
            emission_model.emissivity_savefile = restore_dict['emissivity_savefile']
        if 'ionization_fraction_savefile' in restore_dict:
            emission_model.ionization_fraction_savefile = restore_dict['ionization_fraction_savefile']

        return emission_model
        
    def interpolate_to_mesh_indices(self, loop):
        """
        Return interpolated loop indices to the temperature and density meshes defined for
        the atomic data. For use with `~scipy.ndimage.map_coordinates`.
        """
        nots_itemperature = splrep(self.temperature.value, np.arange(self.temperature.shape[0]))
        nots_idensity = splrep(self.density.value, np.arange(self.density.shape[0]))
        itemperature = splev(np.ravel(loop.electron_temperature.value), nots_itemperature)
        idensity = splev(np.ravel(loop.density.value), nots_idensity)

        return itemperature, idensity
        
    def calculate_emissivity(self, savefile, **kwargs):
        """
        Calculate and store emissivity for every ion in the model
        """
        notebook = kwargs.get('notebook', True)
        self.emissivity_savefile = savefile
        with h5py.File(savefile, 'w') as hf:
            with ProgressBar(len(self._ion_list), ipython_widget=notebook) as progress:
                for ion in self:
                    wavelength, emissivity = ion.emissivity(self.density, include_energy=False)
                    if wavelength is None or emissivity is None:
                        continue
                    emissivity = emissivity[:, :, np.argsort(wavelength)]
                    wavelength = np.sort(wavelength)
                    grp = hf.create_group(ion.ion_name)
                    ds = grp.create_dataset('wavelength', data=wavelength.value)
                    ds.attrs['units'] = wavelength.unit.to_string()
                    ds = grp.create_dataset('emissivity', data=emissivity.data)
                    ds.attrs['units'] = emissivity.unit.to_string()
                    progress.update()
    
    def get_emissivity(self, ion):
        """
        Get emissivity for a particular ion
        """
        with h5py.File(self.emissivity_savefile, 'r') as hf:
            if ion.ion_name not in hf:
                return (None, None)
            ds = hf['/'.join([ion.ion_name, 'wavelength'])]
            wavelength = u.Quantity(ds, ds.attrs['units'])
            ds = hf['/'.join([ion.ion_name, 'emissivity'])]
            emissivity = u.Quantity(ds,  ds.attrs['units'])
            
        return wavelength, emissivity

    def calculate_ionization_fraction(self, field, savefile, interface=None, **kwargs):
        """
        Compute population fractions for each ion and for each loop.

        Find the fractional ionization for each loop in the model as defined by the loop
        model interface. If no interface is provided, the ionization fractions are calculated
        assuming ionization equilibrium.

        Parameters
        ----------
        field : `~synthesizAR.Field`
        savefile : `str`
        interface : optional
            Hydrodynamic model interface

        Other Parameters
        ----------------
        log_temperature_dex : `float`, optional
        """
        self.ionization_fraction_savefile = savefile
        # Create sufficiently fine temperature grid
        dex = kwargs.get('log_temperature_dex', 0.01)
        logTmin = np.log10(self.temperature.value.min())
        logTmax = np.log10(self.temperature.value.max())
        temperature = u.Quantity(10.**(np.arange(logTmin, logTmax+dex, dex)), self.temperature.unit)
        
        if interface is not None:
            return interface.calculate_ionization_fraction(field, self, temperature=temperature,
                                                           **kwargs)
        unique_elements = list(set([ion.element_name for ion in self]))
        # Calculate ionization equilibrium for each element and interpolate to each loop
        notebook = kwargs.get('notebook', True)
        with h5py.File(self.ionization_fraction_savefile, 'a') as hf:
            with ProgressBar(len(unique_elements) * len(field.loops), ipython_widget=notebook) as progress:
                for el_name in unique_elements:
                    element = Element(el_name, temperature)
                    ioneq = element.equilibrium_ionization()
                    f_ioneq = interp1d(temperature, ioneq, axis=0, kind='linear',
                                       fill_value='extrapolate')
                    for loop in field.loops:
                        grp = hf.create_group(loop.name) if loop.name not in hf else hf[loop.name]
                        tmp = f_ioneq(loop.electron_temperature)
                        data = u.Quantity(np.where(tmp < 0., 0., tmp), ioneq.unit)
                        if element.element_name not in grp:
                            dset = grp.create_dataset(element.element_name, data=data.value)
                        else:
                            dset = grp[element.element_name]
                            dset[:, :, :] = data.value
                        dset.attrs['units'] = data.unit.to_string()
                        dset.attrs['description'] = 'equilibrium ionization fractions'
                        progress.update()

    def get_ionization_fraction(self, loop, ion):
        """
        Get ionization state from the ionization balance equations.

        Get ion population fractions for a particular loop and element. This can be either the
        equilibrium or the non-equilibrium ionization fraction, depending on which was calculated.

        Parameters
        ----------
        loop : `~synthesizAR.Loop`
        ion : `~synthesizAR.atomic.Ion`
        """
        with h5py.File(self.ionization_fraction_savefile, 'r') as hf:
            dset = hf['/'.join([loop.name, ion.element_name])]
            ionization_fraction = u.Quantity(dset[:, :, ion.charge_state], dset.attrs['units'])

        return ionization_fraction

    def calculate_emission(self, loop, **kwargs):
        """
        Calculate power per unit volume for a given temperature and density for every transition,
        :math:`\lambda`, in every ion :math:`X^{+m}`, as given by the equation,

        .. math::
           P_{\lambda}(n,T) = \\frac{1}{4\pi}0.83\mathrm{Ab}(X)\\varepsilon_{\lambda}(n,T)\\frac{N(X^{+m})}{N(X)}n

        where :math:`\\mathrm{Ab}(X)` is the abundance of element :math:`X`,
        :math:`\\varepsilon_{\lambda}` is the emissivity for transition :math:`\lambda`,
        and :math:`N(X^{+m})/N(X)` is the ionization fraction of ion :math:`X^{+m}`.
        :math:`P_{\lambda}` is in units of erg cm\ :sup:`-3` s\ :sup:`-1` sr\ :sup:`-1` if
        `energy_unit` is set to `erg` and in units of photons
        cm\ :sup:`-3` s\ :sup:`-1` sr\ :sup:`-1` if `energy_unit` is set to `photon`.
        """
        raise NotImplementedError('Have not yet reimplemented emission calculation.')