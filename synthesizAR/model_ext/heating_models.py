"""
Heating models for hydrodynamic simulations
"""

import logging

import numpy as np

class UniformHeating(object):
    """
    A model for uniform, equally-spaced heating pulses. Accessed
    through the `calculate_event_properties` method.
    """

    def __init__(self,heating_options):
        """
        Constructor
        """
        self.heating_options = heating_options


    @property
    def number_events(self):
        """
        Number of heating events
        """
        return int(np.ceil(self.base_config['total_time']/(self.heating_options['duration'] + self.heating_options['average_waiting_time'])))


    def calculate_event_properties(self,loop):
        """
        Find heating rates and event times
        """
        rates = self._calculate_heating_rates(loop)
        tsr,ter,tsd,ted = self._calculate_event_times()

        return {'magnitude':rates, 'rise_start':tsr, 'rise_end':ter, 'decay_start':tsd, 'decay_end':ted}


    def _calculate_heating_rates(self,loop):
        """
        Calculate uniform heating rates
        """
        available_energy = calculate_free_energy(loop.field_aligned_coordinate, loop.field_strength,stress_level=self.heating_options['stress_level'])
        uniform_heating_rate = 2.0*available_energy/(self.number_events*(2.0*self.heating_options['duration'] - self.heating_options['duration_rise'] - self.heating_options['duration_decay']))

        return np.array(self.number_events*[uniform_heating_rate])


    def _calculate_event_times(self):
        """
        Calculate the onset times of phases of all heating events
        """
        start_times = np.array([i*(self.heating_options['duration'] + self.heating_options['average_waiting_time']) for i in range(self.number_events)])
        end_rise_times = start_times+self.heating_options['duration_rise']
        start_decay_times = end_rise_times+(self.heating_options['duration'] - self.heating_options['duration_rise'] - self.heating_options['duration_decay'])
        end_decay_times = start_times + self.heating_options['duration']

        return start_times,end_rise_times,start_decay_times,end_decay_times



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
