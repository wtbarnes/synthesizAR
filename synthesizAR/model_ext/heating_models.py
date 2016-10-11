"""
Heating models for hydrodynamic simulations
"""

import logging
import random

import numpy as np

class HeatingBase(object):
    """
    Base class for all heating models.
    """

    def __init__(self,heating_options):
        """
        Constructor
        """
        self.logger = logging.getLogger(name=type(self).__name__)
        self.heating_options = heating_options


    @property
    def number_events(self):
        """
        Number of heating events
        """
        return int(np.ceil(self.base_config['total_time']/(self.heating_options['duration'] + self.heating_options['average_waiting_time'])))


    def _calculate_event_times(self):
        """
        Calculate the onset times of phases of all heating events
        """
        start_times = np.array([i*(self.heating_options['duration'] + self.heating_options['average_waiting_time']) for i in range(self.number_events)])
        end_rise_times = start_times+self.heating_options['duration_rise']
        start_decay_times = end_rise_times+(self.heating_options['duration'] - self.heating_options['duration_rise'] - self.heating_options['duration_decay'])
        end_decay_times = start_times + self.heating_options['duration']

        return start_times,end_rise_times,start_decay_times,end_decay_times


class UniformHeating(HeatingBase):
    """
    A model for uniform, equally-spaced heating pulses. Accessed
    through the `calculate_event_properties` method.
    """

    def calculate_event_properties(self,loop):
        """
        Find heating rates and event times
        """
        available_energy = calculate_free_energy(loop.field_aligned_coordinate, loop.field_strength,stress_level=self.heating_options['stress_level'])
        uniform_heating_rate = 2.0*available_energy/(self.number_events*(2.0*self.heating_options['duration'] - self.heating_options['duration_rise'] - self.heating_options['duration_decay']))
        rates = np.array(self.number_events*[uniform_heating_rate])
        tsr,ter,tsd,ted = self._calculate_event_times()

        return {'magnitude':rates, 'rise_start':tsr, 'rise_end':ter, 'decay_start':tsd, 'decay_end':ted}


class PowerLawBase(HeatingBase):
    """
    Base class for power-law models
    """

    def _constrain_distribution(self,available_energy,max_tries = 2000,tol=1e-3,**kwargs):
        """Choose events from power-law distribution such that total desired energy input is conserved."""

        #calculate uniform heating rate for convenience
        uniform_heating_rate = 2.0*available_energy/(2.0*self.heating_options['duration'] - (self.heating_options['duration_rise'] + self.heating_options['duration_decay']))/self.number_events
        #initial guess of bounds
        a0 = 2./(self.heating_options['delta_power_law_bounds']- 1.)*uniform_heating_rate
        a1 = self.heating_options['delta_power_law_bounds']*a0
        #initialize parameters
        tries = 0
        err = 1.e+300
        best_err = err
        while tries < max_tries and err > tol:
            x = np.random.rand(self.number_events)
            h = power_law_transform(x,a0,a1,self.heating_options['alpha'])
            pl_sum = np.sum(h)
            chi = 2.0*available_energy/(2.0*self.heating_options['duration'] - (self.heating_options['duration_rise'] + self.heating_options['duration_decay']))/pl_sum
            a0 = chi*a0
            a1 = self.heating_options['delta_power_law_bounds']*a0
            err = np.fabs(1.-1./chi)
            if err < best_err:
                best = h
                best_err = err
            tries += 1

        self.logger.debug("chi = %f, # of tries = %d, error = %f"%(chi,tries,err))

        if tries >= max_tries:
            self.logger.warning("Power-law constrainer reached max # of tries, using best guess with error = %f"%best_err)

        return np.array(random.sample(list(best),len(best)))


class PowerLawUnscaledWaitingTimes(PowerLawBase):
    """
    Heating rates chosen from power-law distribution but waiting times are not
    dependent on heating rate for each event.
    """

    def calculate_event_properties(self,loop):
        """
        Find heating rates and event times
        """
        available_energy = calculate_free_energy(loop.field_aligned_coordinate, loop.field_strength,stress_level=self.heating_options['stress_level'])
        rates = self._constrain_distribution(available_energy)
        tsr,ter,tsd,ted = self._calculate_event_times()

        return {'magnitude':rates, 'rise_start':tsr, 'rise_end':ter, 'decay_start':tsd, 'decay_end':ted}


class PowerLawScaledWaitingTimes(PowerLawBase):
    """
    Heating rates chosen from power-law distribution but waiting times are dependent
    on heating rate for each event as determined by a scaling factor beta.
    """

    def calculate_event_properties(self,loop):
        """
        Find heating rates and event times
        """
        available_energy = calculate_free_energy(loop.field_aligned_coordinate, loop.field_strength, stress_level=self.heating_options['stress_level'])
        rates = self._constrain_distribution(available_energy)
        tsr,ter,tsd,ted = self._calculate_event_times(rates)
        return {'magnitude':rates, 'rise_start':tsr, 'rise_end':ter, 'decay_start':tsd, 'decay_end':ted}


    def _calculate_event_times(self,rates):
        """
        Calculate the event times that depend on the respective heating rates.
        """
        #calculate constant scaling between waiting time and heating rate
        scaling_constant = (rates**(1.0/self.heating_options['waiting_time_scaling'])).sum()/self.number_events/self.heating_options['average_waiting_time']
        time_start_rise = np.empty([self.number_events])
        wait_time_sum = 0.0
        for i in range(self.number_events):
            time_start_rise[i] = i*self.heating_options['duration'] + wait_time_sum
            wait_time_sum += (rates[i]**(1.0/self.heating_options['waiting_time_scaling']))/scaling_constant

        time_end_rise = time_start_rise+self.heating_options['duration_rise']
        time_start_decay = time_end_rise + (self.heating_options['duration']-self.heating_options['duration_rise'] - self.heating_options['duration_decay'])
        time_end_decay = time_start_decay + self.heating_options['duration_decay']
        return time_start_rise,time_end_rise,time_start_decay,time_end_decay


def calculate_free_energy(coordinate,field,stress_level=0.3):
    """
    Calculate available free energy of the magnetic field using
    Eq. 1 of [1]_

    References
    ----------
    .. [1] Reep et al., 2013, ApJ, `764, 193 <http://adsabs.harvard.edu/abs/2013ApJ...764..193R>`_
    """
    average_field_strength = np.average(field,weights=np.gradient(coordinate))
    return ((stress_level*average_field_strength)**2)/(8.*np.pi)


def power_law_transform(x,a0,a1,alpha):
    """
    Transform uniform distribution to a power-law distribution.

    Parameters
    ----------
    x : array-like
        Uniform distribution
    a0 : `float`
        Lower bound on power-law distribution
    a1 : `float`
        Upper bound on power-law distribution
    alpha : `float`
        Index of the power-law distribution
    """

    return ((a1**(alpha + 1.) - a0**(alpha + 1.))*x + a0**(alpha + 1.))**(1./(alpha + 1.))
