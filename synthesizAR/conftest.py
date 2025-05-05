"""
Configure test skeletons here so that they can be used everywhere
"""
import astropy.units as u
import pytest

from sunpy.coordinates import get_earth

import synthesizAR

from synthesizAR.interfaces import MartensInterface
from synthesizAR.models import semi_circular_arcade, semi_circular_loop


@pytest.fixture
def earth_observer():
    return get_earth(time='2020-01-01T00:00:00')


@pytest.fixture
def bare_skeleton(earth_observer):
    arcade = semi_circular_arcade(100*u.Mm, 20*u.deg, 10, earth_observer)
    loops = [synthesizAR.Strand(f'{i}', c) for i, c in enumerate(arcade)]
    return synthesizAR.Skeleton(loops)


@pytest.fixture
def skeleton_with_model(bare_skeleton):
    interface = MartensInterface(1*u.erg/u.cm**3/u.s)
    bare_skeleton.load_loop_simulations(interface)
    return bare_skeleton


@pytest.fixture
def skeleton_with_time_dependent_model(bare_skeleton):
    from ebtelplusplus.models import HeatingModel, TriangularHeatingEvent

    from synthesizAR.interfaces.ebtel import EbtelInterface
    heating_model = HeatingModel(partition=1)
    heating_model.events = [TriangularHeatingEvent(0*u.s, 200*u.s, 5e-3*u.Unit('erg cm-3 s-1'))]
    interface = EbtelInterface(5e3*u.s, heating_model=heating_model)
    bare_skeleton.load_loop_simulations(interface)
    return bare_skeleton


@pytest.fixture
def semi_circle_strand():
    coords = semi_circular_loop(length=100*u.Mm)
    return synthesizAR.Strand('test', coords)
