"""
Various models for calculating emission from multiple ions
"""
import asdf
import astropy.units as u
import fiasco
import numpy as np
import pathlib
import zarr

from fiasco.util.exceptions import MissingDatasetException
from functools import cache
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d

__all__ = ['EmissionModel']


class EmissionModel(fiasco.IonCollection):
    """
    Model for what atomic data is used to calculate emissivity.

    This model calculates line and continuum emissivity as a function
    of temperature, density (for line emission), and wavelength. In the
    case of line emission, emissivity is calculated once and then saved
    to a table to avoid repeating expensive level populations calculations.

    Parameters
    ----------
    density: `~astropy.units.Quantity`
        Array of densities over which line emissivity is computed
    args: `~fiasco.Ion`
        All remaining positional arguments are the ions comprising the
        emission model. These can also be elements or collections.
    line_emissivity_table_filename: `str` or pathlike, optional
        Path to Zarr store for saving the line emissivity calculations.
        If this file already exists, care should be taken to ensure that
        all input parameters are the same as the model that originally
        generated this file. If not specified, the emissivity table will
        be saved to the current directory using the `id` of this object.
    kwargs : `dict`, optional
        Additional keyword arguments to be passed to `fiasco.Ion.level_populations`.
    """

    @u.quantity_input
    def __init__(self, density: u.cm**(-3), *args, emissivity_table_filename=None, **kwargs):
        super().__init__(*args)
        self.density = density
        self.emissivity_table_filename = emissivity_table_filename
        self._level_pops_kwargs = kwargs

    @property
    def emissivity_table_filename(self):
        return self._emissivity_table_filename

    @emissivity_table_filename.setter
    def emissivity_table_filename(self, value):
        if value is None:
            value = pathlib.Path.cwd() / f'emissivity_table_{id(self)}.zarr'
        self._emissivity_table_filename = pathlib.Path(value)

    def to_asdf(self, filename):
        """
        Serialize an `EmissionModel` to an ASDF file
        """
        tree = {
            'temperature': self.temperature,
            'density': self.density,
            'emissivity_table_filename': self.emissivity_table_filename,
            'level_pops_kwargs': self._level_pops_kwargs,
            'ions': [ion.ion_name for ion in self],
            'ion_kwargs': [ion._instance_kwargs for ion in self]
        }
        with asdf.AsdfFile(tree) as asdf_file:
            asdf_file.write_to(filename)

    @classmethod
    def from_asdf(cls, filename):
        """
        Restore `EmissionModel` instance from an ASDF file
        """
        with asdf.open(filename, mode='r', memmap=False, lazy_load=False) as af:
            temperature = af.tree['temperature']
            density = af.tree['density']
            emissivity_table_filename = af.tree['emissivity_table_filename']
            level_pops_kwargs = af.tree['level_pops_kwargs']
            ions = af.tree['ions']
            ion_kwargs = af.tree['ion_kwargs']

        ions = [fiasco.Ion(ion, temperature, **kwargs) for ion, kwargs in zip(ions, ion_kwargs)]
        em_model = cls(density, emissivity_table_filename, *ions, **level_pops_kwargs)
        return em_model

    def _get_quantity_from_emissivity_table(self, ion, path):
        "Convenience method for retrieving a unitful quantity for a given ion from the Zarr file"
        root = zarr.open(store=self.emissivity_table_filename, mode='r')
        ds = root[f'{ion.ion_name}/{path}']
        return u.Quantity(ds, ds.attrs.get('unit', ''))

    def _calculate_line_emissivity(self, ion):
        # NOTE: Purposefully not using the contribution_function or emissivity methods on
        # fiasco.Ion because (i) ionization fraction may be loop dependent, (ii) don't want
        # to assume any abundance at this stage so that we can change it later without having
        # to recalculate the level populations, and (iii) we want to exclude the hc/lambda
        # factor.
        try:
            pop = ion.level_populations(self.density, **self._level_pops_kwargs)
        except MissingDatasetException:
            # NOTE: populations not available for every ion
            self.log.warning(f'Cannot compute level populations for {ion.ion_name}')
            wavelength = [0]*u.AA
            emissivity = u.Quantity(np.zeros(self.temperature.shape+self.density.shape+(1,)), 'ph s-1')
        else:
            upper_level = ion.transitions.upper_level[ion.transitions.is_bound_bound]
            wavelength = ion.transitions.wavelength[ion.transitions.is_bound_bound]
            A = ion.transitions.A[ion.transitions.is_bound_bound]
            i_upper = fiasco.util.vectorize_where(ion._elvlc['level'], upper_level)
            emissivity = pop[:, :, i_upper] * A * u.photon
            emissivity = emissivity[:, :, np.argsort(wavelength)]
            wavelength = np.sort(wavelength)
            # This is the factor of n_H/n_e * 1/n_e which replaces 0.83 / n_e
            nH_ne2 = np.outer(ion.proton_electron_ratio, 1/self.density)[..., np.newaxis]
            emissivity *= ion.abundance * nH_ne2
        return wavelength, emissivity

    def get_line_emissivity(self, ion):
        r"""
        Get bound-bound emissivity for all lines of a particular ion.

        .. note:: This first searches the Zarr store at ``line_emissivity_table_filename`` and if
                  no emissivity is found, it is calculated. As such, the first time this is run for
                  a given ion it may be slow, but will cache the result on subsequent calls.

        In this case, the emissivity, as a function of density :math:`n` and temperature :math:`T`,
        for a transition :math:`ij` is defined as,

        .. math::

            G_{ij}(n,T) = \frac{n_H}{n_e}\frac{1}{n_e}\mathrm{Ab}_X N_j(n,T) A_{ij}

        where :math:`N_j` is the level population of level :math:`j` for a given ion of element :math:`X`.
        Note that this has units of :math:`\mathrm{s}^{-1}`. Note that this is effectively the contribution
        function without the ionization fraction. The goal of this function is to compute, per ion and as a
        function of wavelength, every quantity that is dependent on the atomic physics. The ionization fraction
        is not included here because it can, in principle, be time-dependent and thus must be calculated later
        using the results of the field-aligned model.

        Parameters
        ----------
        ion: `fiasco.Ion`
            Ion instance for which to compute bound-bound line emission.
        """
        root = zarr.open(store=self.emissivity_table_filename, mode='a')
        if root.get(f'{ion.ion_name}/line') is None:
            wavelength, emissivity = self._calculate_line_emissivity(ion)
            grp = root.create_group(f'{ion.ion_name}/line')
            ds = grp.create_array('wavelength', data=wavelength.value)
            ds.attrs['unit'] = wavelength.unit.to_string()
            ds = grp.create_array('emissivity', data=emissivity.value)
            ds.attrs['unit'] = emissivity.unit.to_string()
        wavelength = self._get_quantity_from_emissivity_table(ion, 'line/wavelength')
        emissivity = self._get_quantity_from_emissivity_table(ion, 'line/emissivity')
        return wavelength, emissivity

    def _calculate_continuum_emissivity(self, ion):
        wavelength = np.arange(1,1000,0.1) * u.AA
        ph_per_erg = u.photon / wavelength.to('erg', equivalencies=u.equivalencies.spectral())
        # NOTE: It may seem silly to be reimplementing the existing free-free and free-bound
        # methods on the collection object, but this is necessary to avoid multiplying by the
        # ionization fraction.
        try:
            ff = ion.free_free(wavelength)
        except MissingDatasetException:
            self.log.warning(f'Cannot compute free-free emission for {ion.ion_name}')
            ff = u.Quantity(np.zeros(self.temperature.shape+wavelength.shape), 'ph cm3 s-1 Angstrom-1')
        try:
            fb = ion.free_bound(wavelength)
        except MissingDatasetException:
            self.log.warning(f'Cannot compute free-bound emission for {ion.ion_name}')
            fb = u.Quantity(np.zeros(self.temperature.shape+wavelength.shape), 'ph cm3 s-1 Angstrom-1')
        emissivity = ion.abundance*ion.proton_electron_ratio[..., np.newaxis]*(ff + fb)*ph_per_erg
        return wavelength, emissivity

    def get_continuum_emissivity(self, ion):
        r"""
        Get continuum (free-free and free-bound) emissivity of a particular ion.

        The continuum emissivity in this case is given by,

        .. math::

            C(T,\lambda) = \frac{n_H}{n_e}\mathrm{Ab}_X(C_{ff}(T,\lambda) + C_{fb}(T,\lambda))\frac{\lambda}{hc}

        where :math:`C_{ff},C_{fb}` are the free-free and free-bound emissivities of the ion as
        computed by `fiasco.Ion.free_free` and `fiasco.Ion.free_bound`, respectively.

        Parameters
        ----------
        ion: `fiasco.Ion`
            Ion for which to calculate the continuum emissivity
        """
        root = zarr.open(store=self.emissivity_table_filename, mode='a')
        if root.get(f'{ion.ion_name}/continuum') is None:
            wavelength, emissivity = self._calculate_continuum_emissivity(ion)
            grp = root.create_group(f'{ion.ion_name}/continuum')
            ds = grp.create_array('wavelength', data=wavelength.value)
            ds.attrs['unit'] = wavelength.unit.to_string()
            ds = grp.create_array('emissivity', data=emissivity.value)
            ds.attrs['unit'] = emissivity.unit.to_string()
        wavelength = self._get_quantity_from_emissivity_table(ion, 'continuum/wavelength')
        emissivity = self._get_quantity_from_emissivity_table(ion, 'continuum/emissivity')
        return wavelength, emissivity

    @cache
    @u.quantity_input
    def calculate_narrowband_emissivity(
            self,
            ion,
            channel,
        ) -> u.Unit('cm5 DN sr s-1 pix-1'):
        r"""
        Calculate ion emissivity integrated over an instrument wavelength response function.

        Compute product between wavelength response :math:`R_c` for ``channel`` (:math:`c`)
        and emissivity for an ion of ionization stage :math:`k` from element :math:`X` in
        this emission model,

        .. math::

            \epsilon_{c,X,k} = \sum_{\{ij\}_{X,k}}\epsilon_{ij}R_c(\lambda_{ij}) + \int\mathrm{d}\lambda\,R_c(\lambda)C(T,\lambda)

        This emissivity includes bound-bound, free-free, and free-bound emission as a function
        of temperature and density.

        .. note:: This explicitly does not include the temperature-dependent ionization fraction
                  for each ion. The reason for excluding this here is that this quantity can,
                  in principle, be time-dependent and thus must be evaluated for each strand.

        Parameters
        ----------
        ion : `~fiasco.Ion`
        channel : Compatible with `sunkit_instruments.response.AbstractChannel`

        Returns
        -------
        emissivity: `~astropy.units.Quantity`
            Total emissivity as a function of temperature and density.
        """
        wavelength = channel.wavelength
        response = channel.wavelength_response()
        f_interp = interp1d(wavelength, response, bounds_error=False, fill_value=0.0)
        # Bound-bound line emissivity
        transition_wavelengths, line_emissivity = self.get_line_emissivity(ion)
        response_interp = f_interp(transition_wavelengths.to_value(wavelength.unit)) * response.unit
        emissivity = np.dot(line_emissivity, response_interp)
        # Continuum emissivity
        continuum_wavelength, continuum_emissivity = self.get_continuum_emissivity(ion)
        response_interp = f_interp(continuum_wavelength.to_value(wavelength.unit)) * response.unit
        integrand = continuum_emissivity * response_interp
        em_continuum = trapezoid(y=integrand.value,
                                 x=continuum_wavelength.value,
                                 axis=-1) * integrand.unit*continuum_wavelength.unit
        emissivity += em_continuum[:, np.newaxis]
        return emissivity
