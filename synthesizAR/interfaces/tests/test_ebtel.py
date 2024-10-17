"""
EBTEL model interface tests
"""
import astropy.units as u
import pytest

from ebtelplusplus.models import HeatingModel, TriangularHeatingEvent

from synthesizAR.interfaces.ebtel import EbtelInterface
from synthesizAR.interfaces.ebtel.heating_models import (
    RandomNanoflare,
    ScaledPowerLawNanoflareTrain,
)


def _check_hydro_quantities(skeleton):
    for s in skeleton.strands:
        for p in ['time', 'electron_temperature', 'ion_temperature', 'density', 'velocity']:
            assert hasattr(s, p)
            assert isinstance(getattr(s, p), u.Quantity)


def test_no_events(bare_skeleton):
    ebtel = EbtelInterface(5e3*u.s,)
    bare_skeleton.load_loop_simulations(ebtel)
    _check_hydro_quantities(bare_skeleton)


def test_exception_events_and_builder():
    # Do not specify events and a way to build them
    builder = RandomNanoflare([500,1000]*u.s, 200*u.s)
    heating_model = HeatingModel(events=[
        TriangularHeatingEvent(100*u.s, 200*u.s, 0.1*u.Unit('erg cm-3 s-1'))
    ])
    with pytest.raises(ValueError, match='Specifying an event_builder'):
        EbtelInterface(5e3*u.s, event_builder=builder, heating_model=heating_model)


def test_custom_events(bare_skeleton):
    heating_model = HeatingModel(partition=1)
    heating_model.events = [TriangularHeatingEvent(100*u.s, 200*u.s, 0.1*u.Unit('erg cm-3 s-1'))]
    ebtel = EbtelInterface(5e3*u.s, heating_model=heating_model)
    bare_skeleton.load_loop_simulations(ebtel)
    _check_hydro_quantities(bare_skeleton)


def test_scaled_nanoflare_train_model(bare_skeleton):
    builder = ScaledPowerLawNanoflareTrain(
        [500,3000]*u.s,
        100*u.s,
        300*u.s,
        [1e-4,0.1]*u.Unit('erg cm-3 s-1'),
        -2.5,
        scaling=1,
    )
    ebtel = EbtelInterface(5e3*u.s, event_builder=builder)
    bare_skeleton.load_loop_simulations(ebtel)
    _check_hydro_quantities(bare_skeleton)
