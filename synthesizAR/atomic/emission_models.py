"""
Various models for calculating emission from multiple ions
"""
import warnings

import numpy as np
import astropy.units as u
import zarr
import fiasco
import asdf


class EmissionModel(fiasco.IonCollection):
    """
    Model for how atomic data is used to calculate emission from coronal plasma.
    """
    
    @u.quantity_input
    def __init__(self, density: u.cm**(-3), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.density = density
        self.emissivity_table_filename = None
        self.resolved_wavelengths = kwargs.get('resolved_wavelengths', {})
        # Cannot have empty abundances so replace them as needed
        default_abundance = kwargs.get('default_abundance_dataset', 'sun_photospheric_2009_asplund')
        for ion in self._ion_list:
            if ion.abundance is None:
                warnings.warn(f'Replacing abundance in {ion.ion_name} with {default_abundance}')
                ion._dset_names['abundance_filename'] = default_abundance
                # If the abundance is still None, throw an error
                if ion.abundance is None:
                    raise ValueError(
                        f'No {ion.element_name} abundance available for {default_abundance}'
                    )

    def to_asdf(self, filename):
        """
        Serialize an `EmissionModel` to an ASDF file
        """
        tree = {
            'temperature': self.temperature,
            'density': self.density,
            'ions': [ion.ion_name for ion in self],
            'dset_names': [ion._dset_names for ion in self]
        }
        tree['emissivity_table_filename'] = self.emissivity_table_filename
        with asdf.AsdfFile(tree) as asdf_file:
            asdf_file.write_to(filename)

    @classmethod
    def from_asdf(cls, filename):
        """
        Restore `EmissionModel` instance from an ASDF file
        """
        with asdf.open(filename, mode='r', copy_arrays=True) as af:
            temperature = af.tree['temperature']
            density = af.tree['density']
            ions = af.tree['ions']
            dset_names = af.tree['dset_names']
            emissivity_table_filename = af.tree['emissivity_table_filename']

        ions = [fiasco.Ion(ion, temperature, **ds) for ion, ds in zip(ions, dset_names)]
        em_model = cls(density, *ions)
        em_model.emissivity_table_filename = emissivity_table_filename
        return em_model
        
    def calculate_emissivity_table(self, filename, **kwargs):
        """
        Calculate and store emissivity for every ion in the model.

        In this case, the emissivity, as a function of density :math:`n` and temperature :math:`T`,
        for a transition :math:`ij` is defined as,

        .. math::

            \epsilon_{ij}(n,T) = N_j(n,T) A_{ij}

        where :math:`N_j` is the level population of :math:`j` and :math:`
        """
        self.emissivity_table_filename = filename
        root = zarr.open(store=filename, mode='w', **kwargs)
        for ion in self:
            # NOTE: Purpusefully not using the contribution_function or emissivity methods on
            # fiasco.Ion because (i) ionization fraction may be loop dependent, (ii) don't want
            # to assume any abundance at this stage so that we can change it later without having
            # to recalculate the level populations, and (iii) we want to exclude the hc/lambda
            # factor.
            pop = ion.level_populations(self.density)
            # NOTE: populations not available for every ion
            if pop is None:
                warnings.warn(f'Cannot compute level populations for {ion.ion_name}')
                continue
            upper_level = ion.transitions.upper_level[~ion.transitions.is_twophoton]
            wavelength = ion.transitions.wavelength[~ion.transitions.is_twophoton]
            A = ion.transitions.A[~ion.transitions.is_twophoton]
            i_upper = fiasco.util.vectorize_where(ion._elvlc['level'], upper_level)
            emissivity = pop[:, :, i_upper] * A * u.photon
            emissivity = emissivity[:, :, np.argsort(wavelength)]
            wavelength = np.sort(wavelength)
            # Save as lookup table
            grp = root.create_group(ion.ion_name)
            ds = grp.create_dataset('wavelength', data=wavelength.value)
            ds.attrs['unit'] = wavelength.unit.to_string()
            ds = grp.create_dataset('emissivity', data=emissivity.data)
            ds.attrs['unit'] = emissivity.unit.to_string()

    def get_emissivity(self, ion):
        """
        Get emissivity for a particular ion
        """
        root = zarr.open(self.emissivity_table_filename, 'r')
        if ion.ion_name not in root:
            return (None, None)
        ds = root[f'{ion.ion_name}/wavelength']
        wavelength = u.Quantity(ds, ds.attrs['unit'])
        ds = root[f'{ion.ion_name}/emissivity']
        emissivity = u.Quantity(ds, ds.attrs['unit'])

        return wavelength, emissivity

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
        raise NotImplementedError
