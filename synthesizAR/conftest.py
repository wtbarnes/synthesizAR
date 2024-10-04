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
def bare_skeleton():
    observer = get_earth(time='2020-01-01T00:00:00')
    arcade = semi_circular_arcade(100*u.Mm, 20*u.deg, 10, observer)
    loops = [synthesizAR.Strand(f'{i}', c) for i, c in enumerate(arcade)]
    return synthesizAR.Skeleton(loops)


@pytest.fixture
def skeleton_with_model(bare_skeleton):
    interface = MartensInterface(1*u.erg/u.cm**3/u.s)
    bare_skeleton.load_loop_simulations(interface)
    return bare_skeleton


@pytest.fixture
def semi_circle_strand(bare_skeleton):
    coords = semi_circular_loop(length=100*u.Mm)
    return synthesizAR.Strand('test', coords)
