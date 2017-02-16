
"""
Low-level functions for computing non-equilibrium population fractions
"""

import os
import logging

import numpy as np
import numba
from ChiantiPy.core import ion
from ChiantiPy.tools.util import el2z,zion2name

from synthesizAR.util import _numba_interpolator_wrapper,_numba_lagrange_interpolator

def get_ion_data(element,zrange=None,logTa=4.0,logTb=9.0,dlogT=0.01):
    """
    Get needed ionization rates, recombination rates, and equilibrium populations
    for a range of ions for a given element over a course temperature range.
    """

    temperature = 10.**(np.arange(logTa,logTb+dlogT,dlogT))
    z = el2z(element)
    if zrange is None:
        zrange = [1,z+1]
    num_z = zrange[1] - zrange[0] + 3
    ionization_rates = np.zeros([len(temperature),num_z])
    recombination_rates = np.zeros([len(temperature),num_z])
    equilibrium_ionization_fractions = np.zeros([len(temperature),num_z])

    for i in range(zrange[0]-1,zrange[1]+2):
        # account for edges where no data is available
        if i==0 or i==z+2:
            continue
        logging.debug('Calculating rates for {} {}'.format(element,i))
        info_ion = ion(zion2name(z,i),temperature=temperature)
        #ionization rates
        info_ion.ionizRate()
        #recombination rates
        info_ion.recombRate()
        #equilibrium populations
        info_ion.ioneqOne()
        #save
        ionization_rates[:,i-zrange[0]-1] = info_ion.IonizRate['rate']
        recombination_rates[:,i-zrange[0]-1] = info_ion.RecombRate['rate']
        equilibrium_ionization_fractions[:,i-zrange[0]-1] = info_ion.IoneqOne

    return ionization_rates,recombination_rates,equilibrium_ionization_fractions,temperature

@numba.jit(nopython=True)
def solve_nei_populations(time,temperature,density,ionization_rate,
                            recombination_rate,eqm_populations,temperature_data,
                            cutoff=1e-10,eps_r=0.6,eps_d=0.1,safety=0.1,safety_increase=2.,
                            safety_decrease=10.):
    """
    Solve the ionization balance equations for a set of temperature and density timeseries using
    ionization and recombination rates from CHIANTI. The populations are initialized with the equilibrium solutions at the initial temperature.
    """

    #initialize
    cur_time = time[0]
    cur_temperature = temperature[0]
    cur_density = density[0]
    cur_populations = _numba_interpolator_wrapper(temperature_data,eqm_populations,temperature[0],
                                        cutoff=cutoff,normalize=True)
    cur_ionization_rate = _numba_interpolator_wrapper(temperature_data,ionization_rate,
                                                        cur_temperature)
    cur_recombination_rate = _numba_interpolator_wrapper(temperature_data,recombination_rate,
                                                    cur_temperature)
    nei_populations = np.zeros((time.shape[0],eqm_populations.shape[1]))
    nei_populations[0,:] = cur_populations

    dt_old = time[1] - time[0]
    step = 1
    save_this_step = False
    #evolve in time
    while cur_time < time[-1]:
        #calculate derivative
        dydt = cur_density*(cur_ionization_rate[:-2]*cur_populations[:-2]
                            + cur_recombination_rate[1:-1]*cur_populations[2:]
                            - cur_ionization_rate[1:-1]*cur_populations[1:-1]
                            - cur_recombination_rate[:-2]*cur_populations[1:-1])
        #calculate timestep
        dt_1 = safety*eps_d/np.fabs(dydt)
        dt_2 = 0.5*(10.**eps_r - 1
                    + np.fabs(10.**(-eps_r) - 1.))*cur_populations[1:-1]/np.fabs(dydt)
        dt_2 = np.where(cur_populations[1:-1]<cutoff,np.inf*np.ones(len(dt_2)),dt_2)
        dt = np.min(np.stack((dt_1,dt_2)))
        #control timestep increase and decrease
        dt = max(dt,dt_old/safety_decrease)
        dt = min(dt,safety_increase*dt_old)
        dt_old = dt
        if cur_time + dt >= time[step]:
            dt = np.fabs(time[step] - cur_time)
            save_this_step = True
        #update
        cur_time += dt
        cur_populations[1:-1] = cur_populations[1:-1] + dydt*dt
        cur_populations /= np.sum(cur_populations)
        cur_populations = np.where(cur_populations<cutoff,np.zeros(len(cur_populations)),
                                    cur_populations)
        cur_temperature = _numba_lagrange_interpolator(time,temperature,cur_time)
        cur_density = _numba_lagrange_interpolator(time,density,cur_time)
        cur_ionization_rate = _numba_interpolator_wrapper(temperature_data,ionization_rate,
                                                            cur_temperature)
        cur_recombination_rate = _numba_interpolator_wrapper(temperature_data,recombination_rate,
                                                                cur_temperature)
        #save
        if save_this_step:
            nei_populations[step,:] = cur_populations
            step += 1
            save_this_step = False

    return nei_populations[1:-1]
