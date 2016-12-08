"""
Specific intensity for a variety of atomic transitions, assuming ionization equilibrium.
"""
import os
import sys

import numpy as np
from scipy.interpolate import splrep,splev
from scipy.ndimage import map_coordinates
import astropy.units as u
import h5py

from synthesizAR.atomic import ChIon


class EquilibriumEmissionModel(object):
    """
    Calculate emission for given set of transitions, assuming ionization equilibrium.

    Notes
    -----
    This should eventually be changed to a base emission model to be used by different (e.g.
    equilibrium versus non-equilibrium) models.
    """


    def __init__(self,ions,temperature=np.logspace(5,8,50)*u.K,
                 density=np.logspace(8,11,50)/(u.cm**3),energy_unit='erg',chianti_db_filename=None):
        self.density_mesh,self.temperature_mesh = np.meshgrid(density,temperature)
        self.wavelengths = np.array(sorted([w.value for ion in ions \
                                        for w in ion['wavelengths']]))*ions[0]['wavelengths'].unit
        self.logger = logging.getLogger(name=type(self).__name__)
        # build CHIANTI database
        if chianti_db_filename is None:
            chianti_db_filename = 'tmp_chianti_db.h5'
        self.logger.info('Creating CHIANTI HDF5 database in {}'.format(chianti_db_filename))
        self._build_chianti_db_h5(ions,chianti_db_filename)
        # build ion objects
        self.ions = []
        for ion in ions:
            self.logger.info('Creating ion {}'.format(ion['name']))
            tmp_ion = ChIon(ion['name'],np.ravel(self.temperature_mesh),np.ravel(self.density_mesh),
                            chianti_db_filename)
            tmp_ion.meta['rcparams']['flux'] = energy_unit
            self.ions.append({'ion':tmp_ion,'transitions':ion['wavelengths']})

    def _build_chianti_db_h5(self,ions,filename):
        """
        Construct temporary HDF5 CHIANTI database
        """
        #create custom datatype for ragged scups arrays
        self._ragged_scups_dt = h5py.special_dtype(vlen=np.dtype('float64'))
        with h5py.File(filename,'a') as hf:
            for ion in ions:
                el = ion['name'].split('_')[0]
                ion_label = ion['name'].split('_')[-1]
                if os.path.join('/',el,ion_label) in hf:
                    continue
                self.logger.info('Building entry for {}'.format(ion['name']))
                #elvlc
                self.logger.info('Building elvlc entry for {}'.format(ion['name']))
                _tmp = ch_tools.io.elvlcRead(ion['name'])
                if _tmp['status']>0:
                    grp = hf.create_group(os.path.join('/',el,ion_label,'elvlc'))
                    self._check_keys(_tmp,grp)
                #wgfa
                self.logger.info('Building wgfa entry for {}'.format(ion['name']))
                try:
                    _tmp = ch_tools.io.wgfaRead(ion['name'])
                    grp = hf.create_group(os.path.join('/',el,ion_label,'wgfa'))
                    self._check_keys(_tmp,grp)
                except IOError:
                    pass
                #scups
                self.logger.info('Building scups entry for {}'.format(ion['name']))
                _tmp = ch_tools.io.scupsRead(ion['name'])
                if 'status' not in _tmp:
                    grp = hf.create_group(os.path.join('/',el,ion_label,'scups'))
                    self._check_keys(_tmp,grp)
                #psplups
                self.logger.info('Building psplups entry for {}'.format(ion['name']))
                _tmp = ch_tools.io.splupsRead(ion['name'],filetype='psplups')
                if 'file not found' not in _tmp:
                    grp = hf.create_group(os.path.join('/',el,ion_label,'psplups'))
                    self._check_keys(_tmp,grp)

    def _check_keys(self,chianti_dict,h5_group):
        """
        Clean CHIANTI data dictionaries before reading into HDF5 file
        """
        for key in chianti_dict:
            self.logger.debug('Reading in key {}'.format(key))
            if key=='ref':
                h5_group.attrs[key] = '\n'.join(chianti_dict[key])
            elif type(chianti_dict[key]) is list or type(chianti_dict[key]) is type(np.array([])):
                data = np.array(chianti_dict[key])
                #stupid unicode stuff
                if '<U' in data.dtype.str:
                    data = data.astype('|S1')
                #rows might not all be the same length, some spline fits have more points
                if data.dtype is np.dtype('O'):
                    ds = h5_group.create_dataset(key,(data.size,),dtype=self._ragged_scups_dt)
                    ds[:] = data
                else:
                    h5_group.create_dataset(key,data=data)
            else:
                h5_group.attrs[key] = chianti_dict[key]

    def _calculate_emissivity(self):
        """
        Calculate emissivity (energy or photons per unit time) for all ions for the desired and transitions and reshape the data.
        """
        for ion in self.ions:
            self.logger.info('Calculating emissivity for ion {}'.format(ion['ion'].meta['name']))
            wvl,emiss = ion['ion'].calculate_emissivity()
            transition_indices = [np.argwhere(wvl==t)[0][0] for t in ion['transitions']]
            ion['emissivity'] = [np.reshape(emiss[ti,:],self.temperature_mesh.shape) \
                                for ti in transition_indices]

    def _calculate_fractional_ionization(self):
        """
        Calculate fractional ionization as a function of temperature for each ion, assuming
        ionization equilibrium and reshape the data.

        Notes
        -----
        For a full non-equilibrium treatment, this method needs to be overridden.
        """
        for ion in self.ions:
            ioneq = ion['ion'].calculate_ionization_equilibrium()
            ion['ionization_fraction'] = np.reshape(ioneq,self.temperature_mesh.shape)

    def calculate_emission(self,temperature,density):
        """
        Calculate power per unit volume for a given temperature and density for every transition,
        :math:`\lambda`, in every ion :math:`X^{+m}`, as given by the equation,

        .. math::
            P_{\lambda}(n,T) = \\frac{1}{4\pi}0.83\mathrm{Ab}(X)\\varepsilon_{\lambda}(n,T)\\frac{N(X^{+m})}{N(X)}n

        where :math:`\\mathrm{Ab}(X)` is the abundance of element :math:`X`, :math:`\\varepsilon_{\lambda}` is the emissivity for transition :math:`\lambda`, and :math:`N(X^{+m})/N(X)` is the ionization fraction of ion :math:`X^{+m}`. :math:`P_{\lambda}` is in units of erg cm\ :sup:`-3` s\ :sup:`-1` sr\ :sup:`-1` if `energy_unit` is set to `erg` and in units of photons cm\ :sup:`-3` s\ :sup:`-1` sr\ :sup:`-1` if `energy_unit` is set to `photon`.
        """
        # interpolate indices
        nots_itemperature =splrep(self.temperature_mesh[:,0].value,
                                np.arange(self.temperature_mesh.shape[0]))
        nots_idensity = splrep(self.density_mesh[0,:].value,np.arange(self.density_mesh.shape[1]))
        itemperature = splev(np.ravel(temperature.value),nots_itemperature)
        idensity = splev(np.ravel(density.value),nots_idensity)

        emiss = {}
        # calculate emissivity
        for ion in self.ions:
            self.logger.debug('Calculating emissivity for ion {}'.format(ion['ion'].meta['name']))
            if 'ionization_fraction' not in ion.keys():
                self._calculate_fractional_ionization()
            if 'emissivity' not in ion.keys():
                self._calculate_emissivity()
            for t,em in zip(ion['transitions'],ion['emissivity']):
                ion_key = '{} {} {}'.format(ion['ion'].meta['spectroscopic_name'],
                                            t.value,t.unit.to_string())
                self.logger.debug('Calculating emission for {}'.format(ion_key))
                _tmp_emiss = em*ion['ionization_fraction']
                _tmp = np.reshape(map_coordinates(_tmp_emiss.value,
                                                np.vstack([itemperature,idensity])),
                                temperature.shape)
                _tmp = np.where(_tmp>0.0,_tmp,0.0)
                emiss[ion_key] = _tmp*em.unit*density*ion['ion'].abundance*0.83/(4*np.pi*u.steradian)

        return emiss
