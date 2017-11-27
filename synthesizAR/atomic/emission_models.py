"""
Various models for calculating emission from multiple ions
"""
import numpy as np
from scipy.interpolate import splrep, splev, interp1d
import astropy.units as u
from astropy.utils.console import ProgressBar
import h5py
import fiasco


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
        model interface.
        """
        self.ionization_fraction_savefile = savefile
        if interface is not None:
            interface.calculate_ionization_fraction(field, self, **kwargs)
        else:
            with h5py.File(self.ionization_fraction_savefile, 'a') as hf:
                for ion in self:
                    f_ioneq = interp1d(ion.temperature, ion.ioneq, fill_value='extrapolate')
                    for loop in field.loops:
                        if loop.name not in hf:
                            grp = hf.create_group(loop.name)
                        else:
                            grp = hf[loop.name]
                        tmp = f_ioneq(loop.electron_temperature)
                        data = u.Quantity(np.where(tmp < 0., 0., tmp), ion.ioneq.unit)
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