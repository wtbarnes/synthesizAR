"""
Heating models for hydrodynamic simulations
"""
import abc
import astropy.units as u
import numpy as np

from ebtelplusplus.models import HeatingEvent

from synthesizAR.models.heating import free_magnetic_energy_density
from synthesizAR.util import power_law_transform

__all__ = [
    'NanoflareTrain',
    'PowerLawNanoflareTrain',
    'ScaledPowerLawNanoflareTrain',
    'RandomNanoflare',
]


class AbstractEventBuilder(abc.ABC):

    @abc.abstractmethod
    def __call__(self, strand)->list[HeatingEvent]:
        ...


class NanoflareTrain(AbstractEventBuilder):
    """
    A sequence of impulsive heating events

    Events are evenly spaced by an amount ``waiting_time`` and have a uniform heating
    rate calculated using `~synthesizAR.util.free_magnetic_energy_density`.

    Parameters
    ----------
    period : `~astropy.units.Quantity`
        The start and end time of the nanoflare train.
    duration : `~astropy.units.Quantity`
        Total duration of each event
    average_waiting_time : `~astropy.units.Quantity`
        Average time between successive events in the train.
    duration_rise : `~astropy.units.Quantity`, optional
        Duration of the rise phase. If not specified, defaults
        to half of ``duration``.
    duration_decay : `~astropy.units.Quantity`,optional
        Duration of the decay phase. If not specified, defaults
        to half of ``duration``.
    stress : `float`, optional
        Fraction of field energy density to input into the loop
    """

    @u.quantity_input
    def __init__(self,
                 period: u.s,
                 duration: u.s,
                 average_waiting_time: u.s,
                 duration_rise: u.s=None,
                 duration_decay: u.s=None,
                 stress=0.3):
        self.period = period
        self.duration = duration
        self.duration_rise = duration_rise
        self.duration_decay = duration_decay
        self.average_waiting_time = average_waiting_time
        self.stress = stress

    @property
    @u.quantity_input
    def duration_rise(self):
        return self._duration_rise

    @duration_rise.setter
    def duration_rise(self, val):
        if val is None:
            self._duration_rise = self.duration / 2
        else:
            self._duration_rise = val

    @property
    @u.quantity_input
    def duration_decay(self):
        return self._duration_decay

    @duration_decay.setter
    def duration_decay(self, val):
        if val is None:
            self._duration_decay = self.duration / 2
        else:
            self._duration_decay = val

    @property
    @u.quantity_input
    def train_duration(self) -> u.s:
        return np.diff(self.period).squeeze()

    @property
    @u.quantity_input
    def duration_constant(self):
        return self.duration - self.duration_rise - self.duration_decay

    @property
    def n_events(self):
        return int(np.ceil(self.train_duration / (self.duration + self.average_waiting_time)))

    @property
    @u.quantity_input
    def waiting_times(self):
        if hasattr(self, '_waiting_times'):
            return self._waiting_times
        return self.average_waiting_time * np.ones(self.n_events)

    @waiting_times.setter
    def waiting_times(self, val):
        self._waiting_times = val

    @u.quantity_input
    def heating_rates(self, strand) -> u.Unit('erg cm-3 s-1'):
        max_energy = free_magnetic_energy_density(strand, stress_level=self.stress)
        rate = max_energy / (0.5*self.duration_rise + self.duration_constant + 0.5*self.duration_decay)
        rate /= self.n_events
        return u.Quantity(np.full(self.n_events, rate.value), rate.unit)

    @property
    @u.quantity_input
    def start_times(self) -> u.s:
        durations = self.duration * np.arange(self.n_events)
        waiting_time_sums = np.cumsum(self.waiting_times[:-1])
        waiting_time_sums = np.append(u.Quantity(0, waiting_time_sums.unit), waiting_time_sums)
        return self.period[0] + np.cumsum(durations) + waiting_time_sums

    def __call__(self, strand):
        rates = self.heating_rates(strand)
        start_times = self.start_times
        events = []
        for st, rate in zip(start_times, rates):
            event = HeatingEvent(st,
                                 self.duration,
                                 self.duration_rise,
                                 self.duration_decay,
                                 rate)
            events.append(event)
        return events


class PowerLawNanoflareTrain(NanoflareTrain):

    def __init__(self,
                 period: u.s,
                 duration: u.s,
                 waiting_time: u.s,
                 bounds: u.Unit('erg cm-3 s-1'),
                 index,
                 duration_rise: u.s=None,
                 duration_decay: u.s=None):
        super().__init__(period,
                       duration,
                       waiting_time,
                       duration_rise=duration_rise,
                       duration_decay=duration_decay)
        self.bounds = bounds
        self.index = index

    def heating_rates(self, strand):
        x = np.random.rand(self.n_events)
        return power_law_transform(x, *self.bounds, self.index)


class ScaledPowerLawNanoflareTrain(PowerLawNanoflareTrain):

    def __init__(self, *args, scaling=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaling = scaling

    def __call__(self, strand):
        # NOTE: This is overloaded because the waiting times depend on the heating rates
        # and the heating rates are sampled randomly from a distribution
        rates = self.heating_rates(strand)
        scaling_const = self.n_events * self.average_waiting_time / (rates**(1/self.scaling)).sum()
        self.waiting_times = scaling_const * rates**(1/self.scaling)
        start_times = self.start_times
        events = []
        for st, rate in zip(start_times, rates):
            event = HeatingEvent(st,
                                 self.duration,
                                 self.duration_rise,
                                 self.duration_decay,
                                 rate)
            events.append(event)
        return events


class RandomNanoflare(NanoflareTrain):
    """
    Single nanoflare at a random time during the simulation period

    The heating rate for event is calculated using `calculate_free_energy`
    and is dependent on the particular strand.

    Parameters
    ----------
    period : `~astropy.units.Quantity`
        The period during which the event can occur.
    duration : `~astropy.units.Quantity`
        Total duration of event
    duration_rise : `~astropy.units.Quantity`
        Duration of the rise phase
    duration_decay : `~astropy.units.Quantity`
        Duration of the decay phase
    stress : `float`
        Fraction of field energy density to input into the loop
    """

    @u.quantity_input
    def __init__(self, period: u.s, duration: u.s, **kwargs):
        super().__init__(period, duration, 0*u.s, **kwargs)

    @property
    @u.quantity_input
    def train_duration(self):
        return self.duration

    @property
    @u.quantity_input
    def start_times(self) -> u.s:
        return np.atleast_1d(u.Quantity(np.random.uniform(*self.period.to_value('s')), 's'))
