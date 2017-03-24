"""
Specific intensity for a variety of atomic transitions, assuming ionization equilibrium.
"""
import os
import sys
import logging
import pickle

import numpy as np
from scipy.interpolate import splrep, splev
from scipy.ndimage import map_coordinates
import astropy.units as u
import h5py
from ChiantiPy.tools.io import elvlcRead, wgfaRead, scupsRead, splupsRead

from synthesizAR.atomic import ChIon


class EmissionModel(object):
    """
    Calculate emission for given set of transitions, assuming ionization equilibrium.

    Notes
    -----
    This should eventually be changed to a base emission model to be used by different (e.g.
    equilibrium versus non-equilibrium) models.
    """

    def __init__(self, ions, temperature=np.logspace(5, 8, 50)*u.K,
                 density=np.logspace(8, 11, 50)/(u.cm**3), energy_unit='erg',
                 chianti_db_filename=None):
        self.density_mesh, self.temperature_mesh = np.meshgrid(density, temperature)
        self.resolved_wavelengths = np.sort(np.hstack([ion['resolved_wavelengths'].value
                                                       for ion in ions])
                                            * ions[0]['resolved_wavelengths'].unit)
        self.logger = logging.getLogger(name=type(self).__name__)
        # build CHIANTI database
        if chianti_db_filename is None:
            chianti_db_filename = 'tmp_chianti_db.h5'
        self.logger.info('Using CHIANTI HDF5 database in {}'.format(chianti_db_filename))
        self._build_chianti_db_h5(ions, chianti_db_filename)
        # build ion objects
        self.ions = []
        for ion in ions:
            self.logger.info('Creating ion {}'.format(ion['name']))
            tmp_ion = ChIon(ion['name'], np.ravel(self.temperature_mesh),
                            np.ravel(self.density_mesh), chianti_db_filename)
            tmp_ion.meta['rcparams']['flux'] = energy_unit
            self.ions.append({'ion': tmp_ion,
                              'resolved_wavelengths': u.Quantity(sorted(ion['resolved_wavelengths']))})

    def save(self, savedir=None):
        """
        Save model object to be used later.
        """
        if savedir is None:
            savedir = 'synthesizAR-{}-save_{}'.format(type(self).__name__,
                                                      datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        self.logger.info('Saving emission model information in {}'.format(savedir))
        # save temperature and density
        with open(os.path.join(savedir, 'temperature_density.pickle'), 'wb') as f:
            pickle.dump([self.temperature_mesh[:, 0], self.density_mesh[0, :]], f)
        # save some basic info
        ions_info = []
        for ion in self.ions:
            ions_info.append({'name': ion['ion'].meta['name'],
                              'resolved_wavelengths': ion['resolved_wavelengths']})
        db_filename = self.ions[0]['ion']._chianti_db_h5
        energy_unit = self.ions[0]['ion'].meta['rcparams']['flux']
        with open(os.path.join(savedir, 'ion_info.pickle'), 'wb') as f:
            pickle.dump([ions_info, db_filename, energy_unit], f)
        # save emissivity and fractional ionization
        if not os.path.exists(os.path.join(savedir, 'emissivity')):
            os.makedirs(os.path.join(savedir, 'emissivity'))
        if not os.path.exists(os.path.join(savedir, 'fractional_ionization')):
            os.makedirs(os.path.join(savedir, 'fractional_ionization'))
        for ion in self.ions:
            if 'emissivity' in ion:
                with open(os.path.join(savedir, 'emissivity',
                                       '{}.pickle'.format(ion['ion'].meta['name'])), 'wb') as f:
                    pickle.dump((ion['wavelength'], (ion['emissivity'])), f)
            if 'fractional_ionization' in ion:
                with open(os.path.join(savedir, 'fractional_ionization',
                                       '{}.pickle'.format(ion['ion'].meta['name'])), 'wb') as f:
                    pickle.dump(ion['fractional_ionization'], f)

    @classmethod
    def restore(cls, savedir):
        """
        Restore emission model from savefile
        """
        with open(os.path.join(savedir, 'temperature_density.pickle'), 'rb') as f:
            temperature, density = pickle.load(f)
        with open(os.path.join(savedir, 'ion_info.pickle'), 'rb') as f:
            ion_info, db_filename, energy_unit = pickle.load(f)
        emiss_model = cls(ion_info, temperature=temperature, density=density,
                          energy_unit=energy_unit, chianti_db_filename=db_filename)
        emiss_model.logger.info('Restoring emission model from {}'.format(savedir))
        for ion in emiss_model.ions:
            tmp_ion_file = os.path.join(savedir, '{}', '{}.pickle'.format(ion['ion'].meta['name']))
            if os.path.isfile(tmp_ion_file.format('emissivity')):
                with open(tmp_ion_file.format('emissivity'), 'rb') as f:
                    ion['wavelength'], ion['emissivity'] = pickle.load(f)
            if os.path.isfile(tmp_ion_file.format('fractional_ionization')):
                with open(tmp_ion_file.format('fractional_ionization'), 'rb') as f:
                    ion['fractional_ionization'] = pickle.load(f)

        return emiss_model

    def _build_chianti_db_h5(self, ions, filename):
        """
        Construct temporary HDF5 CHIANTI database
        """
        # create custom datatype for ragged scups arrays
        self._ragged_scups_dt = h5py.special_dtype(vlen=np.dtype('float64'))
        with h5py.File(filename, 'a') as hf:
            for ion in ions:
                el = ion['name'].split('_')[0]
                ion_label = ion['name'].split('_')[-1]
                if '/'.join([el, ion_label]) in hf:
                    continue
                self.logger.info('Building entry for {}'.format(ion['name']))
                # elvlc
                self.logger.info('Building elvlc entry for {}'.format(ion['name']))
                _tmp = elvlcRead(ion['name'])
                if _tmp['status'] > 0:
                    grp = hf.create_group('/'.join([el, ion_label, 'elvlc']))
                    self._check_keys(_tmp, grp)
                # wgfa
                self.logger.info('Building wgfa entry for {}'.format(ion['name']))
                try:
                    _tmp = wgfaRead(ion['name'])
                    grp = hf.create_group('/'.join([el, ion_label, 'wgfa']))
                    self._check_keys(_tmp, grp)
                except IOError:
                    pass
                # scups
                self.logger.info('Building scups entry for {}'.format(ion['name']))
                _tmp = scupsRead(ion['name'])
                if 'status' not in _tmp:
                    grp = hf.create_group('/'.join([el, ion_label, 'scups']))
                    self._check_keys(_tmp, grp)
                # psplups
                self.logger.info('Building psplups entry for {}'.format(ion['name']))
                _tmp = splupsRead(ion['name'], filetype='psplups')
                if 'file not found' not in _tmp:
                    grp = hf.create_group('/'.join([el, ion_label, 'psplups']))
                    self._check_keys(_tmp, grp)

    def _check_keys(self, chianti_dict, h5_group):
        """
        Clean CHIANTI data dictionaries before reading into HDF5 file
        """
        for key in chianti_dict:
            self.logger.debug('Reading in key {}'.format(key))
            if key == 'ref':
                h5_group.attrs[key] = '\n'.join(chianti_dict[key])
            elif type(chianti_dict[key]) is list or type(chianti_dict[key]) is np.ndarray:
                data = np.array(chianti_dict[key])
                # stupid unicode stuff
                if '<U' in data.dtype.str:
                    data = data.astype('|S1')
                # rows might not all be the same length, some spline fits have more points
                if data.dtype is np.dtype('O'):
                    ds = h5_group.create_dataset(key, (data.size,), dtype=self._ragged_scups_dt)
                    ds[:] = data
                else:
                    h5_group.create_dataset(key, data=data)
            else:
                h5_group.attrs[key] = chianti_dict[key]

    def calculate_emissivity(self):
        """
        Calculate emissivity (energy or photons per unit time) for all ions for the desired and 
        transitions and reshape the data.
        """
        for ion in self.ions:
            self.logger.info('Calculating emissivity for ion {}'.format(ion['ion'].meta['name']))
            wvl, emiss = ion['ion'].calculate_emissivity()
            ion['emissivity'] = np.reshape(np.transpose(emiss)[:, np.argsort(wvl)],
                                           self.temperature_mesh.shape+(len(wvl),))
            ion['wavelength'] = np.sort(wvl)

    def calculate_equilibrium_fractional_ionization(self):
        """
        Calculate fractional ionization as a function of temperature for each ion, assuming
        ionization equilibrium and reshape the data.

        Notes
        -----
        For a full non-equilibrium treatment, this method needs to be overridden.
        """
        for ion in self.ions:
            ioneq = ion['ion'].equilibrium_fractional_ionization
            ion['equilibrium_fractional_ionization'] = np.reshape(ioneq, self.temperature_mesh.shape)

    def interpolate_to_mesh_indices(self, loop):
        """
        Return interpolated loop indices to the temperature and density meshes defined for 
        the atomic data. For use with `~scipy.ndimage.map_coordinates`.
        """
        nots_itemperature = splrep(self.temperature_mesh[:, 0].value,
                                   np.arange(self.temperature_mesh.shape[0]))
        nots_idensity = splrep(self.density_mesh[0, :].value, np.arange(self.density_mesh.shape[1]))
        itemperature = splev(np.ravel(loop.temperature.value), nots_itemperature)
        idensity = splev(np.ravel(loop.density.value), nots_idensity)

        return itemperature, idensity

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
        # get interpolated mesh-to-loop indices
        itemperature, idensity = self.interpolate_to_mesh_indices(loop)

        emiss, meta = {}, {}
        # calculate emissivity
        for ion in self.ions:
            self.logger.debug('Calculating emissivity for ion {}'.format(ion['ion'].meta['name']))
            fractional_ionization = loop.get_fractional_ionization(ion['ion'].meta['Element'],
                                                                   ion['ion'].meta['Ion'])
            if 'emissivity' not in ion:
                self.calculate_emissivity()
            em_ion = fractional_ionization*loop.density*ion['ion'].abundance*0.83/(4*np.pi*u.steradian)

            # wavelength resolved emission
            for t in ion['resolved_wavelengths']:
                transition_key = '{} {} {}'.format(ion['ion'].meta['spectroscopic_name'],
                                                   t.value, t.unit.to_string())
                self.logger.debug('Calculating emission for {}'.format(transition_key))
                i = np.argwhere(np.isclose(ion['wavelength'].value, t.value, rtol=0.0, atol=1.e-5))[0][0]
                _tmp = np.reshape(map_coordinates(ion['emissivity'][:, :, i].value,
                                                  np.vstack([itemperature, idensity])),
                                  loop.temperature.shape)
                _tmp = np.where(_tmp > 0.0, _tmp, 0.0)
                emiss['{}'.format(t.value)] = _tmp*ion['emissivity'].unit*em_ion
                meta['{}'.format(t.value)] = {'ion_name': ion['ion'].meta['spectroscopic_name'],
                                              'wavelength_units': t.unit.to_string()}

        return emiss, meta
