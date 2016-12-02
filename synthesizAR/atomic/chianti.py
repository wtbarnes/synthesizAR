"""
Wrappers for aggregating CHIANTI data and doing fundamental atomic physics calculations.
This should eventually be deprecated in favor of using ChiantiPy.
"""

import os
import logging
import copy
import itertools

import numpy as np
import h5py
from scipy.interpolate import interp1d,splrep,splev
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import ChiantiPy.tools as ch_tools

from synthesizAR.util import collect_points

class ChIon(object):
    """
    This is a stripped down version of the ChiantiPy ion object. It speeds up and
    streamlines the emissivity calculation.

    Parameters
    ----------
    temperature : `astropy.Quantity`
    density : `astropy.Quantity`
    setup : `bool`

    .. warning:: This object should eventually be deprecated in favor of using the ion
                object in ChiantiPy.
    """


    @u.quantity_input(temperature=u.K,electron_density=u.cm**(-3))
    def __init__(self,ion_name,temperature,electron_density,chianti_db_h5,setup=True,**kwargs):
        self.logger = logging.getLogger(name=type(self).__name__)
        if ion_name not in ch_tools.data.MasterList:
            raise ValueError('{} not in CHIANTI database'.format(ion_name))
        self.meta = ch_tools.util.convertName(ion_name)
        self.meta['name'] = ion_name
        self.meta['spectroscopic_name'] = ch_tools.util.zion2spectroscopic(self.meta['Z'],
                                                                            self.meta['Ion'])
        self.meta['rcparams'] = ch_tools.data.Defaults.copy()
        # set location of CHIANTI database
        self._chianti_db_h5 = chianti_db_h5
        # read ion data from CHIANTI database
        if setup:
            self._setup_ion(**kwargs)
        # check and set temperature and density
        if temperature.size != electron_density.size:
            if temperature.size == 1:
                temperature = np.tile(temperature,len(electron_density))
            elif electron_density.size == 1:
                electron_density = np.tile(electron_density,len(temperature))
            else:
                raise ValueError('''Temperature and density must be equal-sized arrays
                                    if neither is a scalar.''')
        self.temperature = temperature
        self.electron_density = electron_density
        self.proton_density = electron_density*self._calculate_proton_density()

    def _setup_ion(self,**kwargs):
        """
        Read files from CHIANTI database for specified ion
        """
        _tmp = ch_tools.io.abundanceRead(abundancename=self.meta['rcparams']['abundfile'])
        self.abundance = _tmp['abundance'][self.meta['Z']-1]*u.s/u.s
        self.meta['abundance_filename'] = _tmp['abundancename']
        elvlc_lvl = self._read_chianti_db_h5('elvlc','lvl')
        wgfa_lvl2 = self._read_chianti_db_h5('wgfa','lvl2')
        #FIXME: These warnings should really go in the respective reader functions
        try:
            self._has_scups = True
            scups_lvl2 = self._read_chianti_db_h5('scups','lvl2')
            n_levels_scups = np.max(scups_lvl2)
        except ValueError:
            self._has_scups = False
            self.logger.warning('{} scups file not found'.format(self.meta['spectroscopic_name']))
            n_levels_scups = 1e+300
        try:
            psplups_tmp = self._read_chianti_db_h5('psplups','lvl2')
            self._has_psplups = True
        except ValueError:
            self._has_psplups = False
            self.logger.warning('{} psplups file not found'.format(self.meta['spectroscopic_name']))
        self.n_levels = np.min([np.max(elvlc_lvl),np.max(wgfa_lvl2),n_levels_scups])

    def _read_chianti_db_h5(self,filetype,data):
        """
        Reader function for CHIANTI data from HDF5 file
        """
        with h5py.File(self._chianti_db_h5,'r') as hf:
            _tmp_grp = hf[os.path.join('/',self.meta['Element'],str(self.meta['Ion']))]
            if filetype in _tmp_grp:
                _tmp = np.array(_tmp_grp[os.path.join(filetype,data)])
            else:
                raise ValueError('{} file does not exist for {}'.format(filetype,
                                                                self.meta['spectroscopic_name']))
        return _tmp

    def _calculate_proton_density(self):
        """
        Calculate proton density to electron density ratio from Eq. 7 of Young et al. (2003)
        """
        _tmp_ioneq = ch_tools.io.ioneqRead(ioneqname=self.meta['rcparams']['ioneqfile'])
        _tmp_abundance = ch_tools.io.abundanceRead(abundancename=self.meta['rcparams']['abundfile'])
        abundance = _tmp_abundance['abundance'][_tmp_abundance['abundance']>0]

        denominator = np.zeros(len(_tmp_ioneq['ioneqTemperature']))
        for i in range(len(abundance)):
            for z in range(1,i+2):
                denominator += z*_tmp_ioneq['ioneqAll'][i,z,:]*abundance[i]

        p2eratio = abundance[0]*_tmp_ioneq['ioneqAll'][0,1,:]/denominator
        nots = splrep(np.log10(_tmp_ioneq['ioneqTemperature']),p2eratio,s=0)

        return splev(np.log10(self.temperature.value),nots,der=0)

    def _descale_collision_strengths(self,x,y,energy_ratio,c,bt_type):
        """
        Apply descaling procedure of BT92 to scaled thermally averaged collision strengths.
        """
        nots = splrep(x,y,s=0)
        if bt_type==1:
            x_new = 1.0 - np.log(c)/np.log(energy_ratio + c)
            upsilon = splev(x_new,nots,der=0)*np.log(energy_ratio + np.e)
        elif bt_type==2:
            x_new = energy_ratio/(energy_ratio + c)
            upsilon = splev(x_new,nots,der=0)
        elif bt_type==3:
            x_new = energy_ratio/(energy_ratio + c)
            upsilon = splev(x_new,nots,der=0)/(energy_ratio + 1.0)
        elif bt_type==4:
            x_new = 1.0 - np.log(c)/np.log(energy_ratio + c)
            upsilon = splev(x_new,nots,der=0)*np.log(energy_ratio + c)
        elif bt_type==6:
            x_new = energy_ratio/(energy_ratio + c)
            upsilon = 10**splev(x_new,nots,der=0)
        else:
            raise ValueError('Unrecognized BT92 scaling option.')

        return upsilon

    def _calculate_collision_strengths(self,protons=False):
        """
        Calculate collision strengths and excitation and de-excitation rates.
        """
        if protons:
            filetype = 'psplups'
            scups_key = 'splups'
            btemp = [np.linspace(0,1,n_spline) for n_spline in self._read_chianti_db_h5(filetype,'nspl')]
        else:
            filetype = 'scups'
            scups_key = 'bscups'
            btemp = self._read_chianti_db_h5(filetype,'btemp')
        energy_ratios = np.outer((self._read_chianti_db_h5(filetype,'de')*u.Ry).to(u.erg),
                                1.0/(self.temperature*const.k_B.cgs))
        upsilon = np.array(list(map(self._descale_collision_strengths,btemp,
                                    self._read_chianti_db_h5(filetype,scups_key),
                                    1.0/energy_ratios,
                                    self._read_chianti_db_h5(filetype,'cups'),
                                    self._read_chianti_db_h5(filetype,'ttype'))))
        upsilon = np.where(upsilon>0.,upsilon,0.0)

        #alias some chianti data
        scups_lvl1 = self._read_chianti_db_h5(filetype,'lvl1')
        scups_lvl2 = self._read_chianti_db_h5(filetype,'lvl2')
        elvlc_lvl = list(self._read_chianti_db_h5('elvlc','lvl'))
        elvlc_mult = self._read_chianti_db_h5('elvlc','mult')

        #calculate weights
        lower_weights = np.array([elvlc_mult[elvlc_lvl.index(lvl)] for lvl in scups_lvl1])
        upper_weights = np.array([elvlc_mult[elvlc_lvl.index(lvl)] for lvl in scups_lvl2])
        # modified transition energies
        _tmp_level_energies = list(np.where(self._read_chianti_db_h5('elvlc','eryd')>=0,
                                            self._read_chianti_db_h5('elvlc','eryd'),
                                            self._read_chianti_db_h5('elvlc','erydth')))
        _tmp_transition_energies = np.array(
            [_tmp_level_energies[elvlc_lvl.index(l2)] - _tmp_level_energies[elvlc_lvl.index(l1)] \
            for l1,l2 in zip(scups_lvl1,scups_lvl2)])*u.Ry.to(u.erg)
        energy_ratios = np.outer(_tmp_transition_energies,1.0/(self.temperature*const.k_B.cgs))
        #calculate excitation and deexcitation rates
        _rate_factor = 2.172e-8*np.sqrt((13.6*u.eV).to(u.erg)\
                        /(self.temperature*const.k_B.cgs))*upsilon
        excitation_rate = ((np.exp(-energy_ratios)*_rate_factor).T*1.0/lower_weights).T
        deexcitation_rate = (_rate_factor.T*1.0/upper_weights).T

        return upsilon,excitation_rate,deexcitation_rate

    def _calculate_level_populations(self):
        """
        Calculate level populations for excited states as a function of temperature
        for relevant transitions.
        """
        #alias some of the chianti data
        wgfa_lvl1 = self._read_chianti_db_h5('wgfa','lvl1')
        wgfa_lvl2 = self._read_chianti_db_h5('wgfa','lvl2')
        wgfa_avalue = self._read_chianti_db_h5('wgfa','avalue')
        scups_lvl1 = self._read_chianti_db_h5('scups','lvl1')
        scups_lvl2 = self._read_chianti_db_h5('scups','lvl2')
        self.logger.debug('''Calculating descaled collision strengths and excitation and
                            deexcitation rates for electrons.''')
        upsilon,excitation_rate,deexcitation_rate = self._calculate_collision_strengths()
        # create excitation/deexcitation rate sums for broadcasting
        l1_indices_electron,_electron_ex_broadcast = collect_points(scups_lvl1,excitation_rate)
        l2_indices_electron,_electron_dex_broadcast = collect_points(scups_lvl2,deexcitation_rate)

        # account for protons if the file exists
        if self._has_psplups:
            #alias some of the chianti data
            psplups_lvl1 = self._read_chianti_db_h5('psplups','lvl1')
            psplups_lvl2 = self._read_chianti_db_h5('psplups','lvl2')
            self.logger.debug('''Calculating descaled collision strengths and excitation and
                                deexcitation rates for protons.''')
            upsilon_proton,excitation_rate_proton,deexcitation_rate_proton \
                                                = self._calculate_collision_strengths(protons=True)
            # create excitation/deexcitation rate sums for broadcasting
            l1_indices_proton,_proton_ex_broadcast = collect_points(psplups_lvl1,
                                                                    excitation_rate_proton)
            l2_indices_proton,_proton_dex_broadcast = collect_points(psplups_lvl2,
                                                                    deexcitation_rate_proton)

        process_matrix = np.zeros([self.n_levels,self.n_levels])
        # add spontaneous emission, TODO: correction for recombination and ionization
        self.logger.debug('Adding contributions from A-values to population matrix.')
        process_matrix[wgfa_lvl1-1,wgfa_lvl2-1] += wgfa_avalue
        # sum all of the level 2 Avalues to broadcast
        wgfa_indices,_wgfa_broadcasts = collect_points(wgfa_lvl2,wgfa_avalue)
        process_matrix[wgfa_indices-1,wgfa_indices-1] -= _wgfa_broadcasts
        #TODO: add photoexcitation and stimulated emission

        #b vector used for inversion later on
        b = np.zeros(process_matrix.shape[0])
        b[-1] = 1.0
        #preallocate memory for level populations
        populations = np.zeros([self.n_levels,len(self.temperature)])
        for i,(nel,npr,T) in enumerate(zip(self.electron_density,self.proton_density,
                                            self.temperature)):
            _tmp = np.copy(process_matrix)
            # excitation and de-excitation by electrons
            _tmp[scups_lvl1-1,scups_lvl2-1] += nel*deexcitation_rate[:,i]
            _tmp[scups_lvl2-1,scups_lvl1-1] += nel*excitation_rate[:,i]
            # broadcast summed excitation rates for level 1
            _tmp[l1_indices_electron-1,l1_indices_electron-1] -= nel*_electron_ex_broadcast[:,i]
            # sum deexcitation rates for level 2 to broadcast
            _tmp[l2_indices_electron-1,l2_indices_electron-1] -= nel*_electron_dex_broadcast[:,i]
            # excitation and de-excitation by protons
            if hasattr(self,'_psplups'):
                _tmp[psplups_lvl1-1,psplups_lvl2-1] += npr*deexcitation_rate_proton[:,i]
                _tmp[psplups_lvl2-1,psplups_lvl1-1] += npr*excitation_rate_proton[:,i]
                # sum excitation rates for level 1 broadcast
                _tmp[l1_indices_proton-1,l1_indices_proton-1] -= npr*_proton_ex_broadcast[:,i]
                # sum deexcitation rates for level 2 broadcast
                _tmp[l2_indices_proton-1,l2_indices_proton-1] -= npr*_proton_dex_broadcast[:,i]
            # TODO: add effects from ionization and recombination
            # invert
            self.logger.debug('Calculating level populations for T,ne,np = {}'.format(T,nel,npr))
            _tmp[-1,:] = np.ones(_tmp.shape[0])
            populations[:,i] = np.linalg.solve(_tmp,b)

        return populations

    def calculate_emissivity(self):
        """Calculate the emissivity for each transition"""
        #find where wavelength is nonzero
        wavelength = np.fabs(self._read_chianti_db_h5('wgfa','wvl'))*u.angstrom
        lvl2 = self._read_chianti_db_h5('wgfa','lvl2')
        avalues = self._read_chianti_db_h5('wgfa','avalue')/u.s
        # exclude two-photon decays that are denoted by 0 wavelength
        lvl2 = lvl2[wavelength!=0]
        avalues = avalues[wavelength!=0]
        wavelength = wavelength[wavelength!=0]
        # set energy conversion factor
        if self.meta['rcparams']['flux'] == 'energy':
            energy_factor = (const.h*const.c).to(u.erg*u.angstrom)/wavelength
        else:
            self.logger.info('Expressing emissivity in units of photons')
            energy_factor = 1.0*u.photon
        # calculate level populations
        self.logger.info('Calculating level populations.')
        level_populations = self._calculate_level_populations()
        # calculate emissivity
        self.logger.info('Calculating emissivity')
        emissivity = ((level_populations[lvl2-1,:]).T*avalues*energy_factor).T

        return wavelength,emissivity

    def calculate_ionization_equilibrium(self):
        """Calculate ionization equilibrium."""
        _tmp_ioneq = ch_tools.io.ioneqRead(ioneqname=self.meta['rcparams']['ioneqfile'])
        match_indices = np.where(
            (self.temperature.value>=_tmp_ioneq['ioneqTemperature'].min()) & \
            (self.temperature.value<=_tmp_ioneq['ioneqTemperature'].max()))[0]
        if len(match_indices) != len(self.temperature):
            warnings.warn('''Temperature out of ionization equilibrium range.
                            Those temperatures will have zero fractional ionization.''')
        fractional_ionization = np.zeros(len(self.temperature))
        f_interp = interp1d(np.log10(_tmp_ioneq['ioneqTemperature']),
                                     _tmp_ioneq['ioneqAll'][self.meta['Z']-1,
                                     self.meta['Ion']-1+self.meta['Dielectronic'],:],
                                     kind='cubic')
        fractional_ionization[match_indices] = f_interp(np.log10(
                                                            self.temperature[match_indices].value))
        fractional_ionization[fractional_ionization<0] = 0.0

        # make it a unitless quantity
        return fractional_ionization*u.s/u.s
