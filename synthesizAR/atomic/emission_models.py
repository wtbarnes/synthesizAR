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

    def save(self, savefile):
        """
        Save a JSON representation of the emission model.
        """
        save_dict = {
            'temperature': self.temperature.value.tolist(),
            'temperature_unit': self.temperature.unit.to_string(),
            'density': self.density.value.tolist(),
            'density_unit': self.density.unit.to_string(),
            'ion_list': ['_'.join([ion.ion_name.split()[0].lower(), ion.ion_name.split()[1]]) for ion in self]
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
        ion_list = [Ion(ion, temperature) for ion in restore_dict['ion_list']]
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
        self.emissivity_savefile = savefile
        with h5py.File(savefile, 'w') as hf:
            with ProgressBar(len(self._ion_list), ipython_widget=kwargs.get('notebook', True)) as progress:
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
        Find the fractional ionization for each loop in the model as defined by the loop
        model interface. If no interface is provided, the ionization fractions are calculated
        assuming ionization equilibrium.
        """
        self.ionization_fraction_savefile = savefile
        if interface is not None:
            interface.calculate_ionization_fraction(field, self, **kwargs)
        else:
            # Group ions by element
            unique_elements = list(set([ion.element_name for ion in self]))
            grouped_ions = {el: [ion for ion in self if ion.element_name == el] for el in unique_elements}
            
            # Create sufficiently fine temperature grid
            dex = kwargs.get('log_temperature_dex', 0.01)
            logTmin, logTmax = np.log10(self.temperature.value.min()), np.log10(self.temperature.value.max())
            temperature = u.Quantity(10.**(np.arange(logTmin, logTmax+dex, dex)), self.temperature.unit)
            
            # Calculate ionization equilibrium for each ion and interpolate to each loop
            with h5py.File(self.ionization_fraction_savefile, 'a') as hf:
                for el_name in grouped_ions:
                    element = Element(el_name, temperature)
                    ioneq = element.equilibrium_ionization()
                    for ion in grouped_ions[el_name]:
                        f_ioneq = interp1d(temperature, ioneq[:, ion.charge_state], kind='linear', 
                                           fill_value='extrapolate')
                        for loop in field.loops:
                            if loop.name not in hf:
                                grp = hf.create_group(loop.name)
                            else:
                                grp = hf[loop.name]
                            tmp = f_ioneq(loop.electron_temperature)
                            data = u.Quantity(np.where(tmp < 0., 0., tmp), ioneq.unit)
                            if ion.ion_name not in grp:
                                dset = grp.create_dataset(ion.ion_name, data=data.value)
                            else:
                                dset = grp[ion.ion_name]
                                dset[:, :] = data.value
                            dset.attrs['units'] = data.unit.to_string()
                            dset.attrs['description'] = 'equilibrium ionization fractions'

    def get_ionization_fraction(self, loop, ion):
        """
        Get ionization state from the ionization balance equations.

        Note
        ----
        This can be either the equilibrium or the non-equilibrium ionization 
        fraction, depending on which was calculated.
        """
        with h5py.File(self.ionization_fraction_savefile, 'r') as hf:
            dset = hf['/'.join([loop.name, ion.ion_name])]
            ionization_fraction = u.Quantity(dset, dset.attrs['units'])

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