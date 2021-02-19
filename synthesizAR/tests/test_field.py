"""
Tests for Skeleton object
"""
import pathlib

import pytest
import numpy as np
import astropy.units as u
import astropy.constants as const
import astropy.time
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicStonyhurst
import distributed
from distributed.utils_test import client, loop, cluster_fixture

import synthesizAR
from synthesizAR.models import semi_circular_arcade
from synthesizAR.interfaces import MartensInterface


@pytest.fixture
def coordinates():
    obs = SkyCoord(
        lon=0*u.deg,
        lat=0*u.deg,
        radius=const.au,
        frame=HeliographicStonyhurst,
        obstime=astropy.time.Time.now(),
    )
    return semi_circular_arcade(100*u.Mm, 20*u.deg, 20, obs, gamma=90*u.deg)


@pytest.fixture
def field_strengths(coordinates):
    return [np.nan * np.ones(c.shape) * u.G for c in coordinates]


@pytest.fixture
def interface():
    return MartensInterface(1*u.erg/u.cm**3/u.s)


@pytest.fixture
def skeleton(coordinates, field_strengths, interface, tmpdir, client):
    skeleton = synthesizAR.Skeleton.from_coordinates(coordinates, field_strengths)
    zarr_file = pathlib.Path(tmpdir.mkdir('loop_results')) / 'results.zarr'
    status = skeleton.load_loop_simulations(interface, str(zarr_file))
    distributed.wait(status)  # wait until all simulations loaded
    return skeleton


def test_skeleton_has_loops(skeleton, coordinates):
    assert hasattr(skeleton, 'loops')
    assert type(skeleton.loops) is list
    assert len(skeleton.loops) == len(coordinates)


def test_loops_have_model_type(skeleton, interface):
    for l in skeleton.loops:
        assert l.simulation_type == interface.name

@pytest.mark.parametrize(
    'name',
    ['time',
     'electron_temperature',
     'ion_temperature',
     'density',
     'velocity',
     'velocity_x',
     'velocity_y',
     'velocity_z',]
)
def test_loops_have_model_quantities(skeleton, name):
    "These quantities exist only after an interface is defined"
    for l in skeleton.loops:
        assert isinstance(getattr(l, name), u.Quantity)


def test_create_skeleton_from_loops(coordinates, field_strengths):
    loops = []
    for i, (coord, mag) in enumerate(zip(coordinates, field_strengths)):
        loops.append(synthesizAR.Loop(f'loop{i}', coord, mag))
    skeleton = synthesizAR.Skeleton(loops)
    assert hasattr(skeleton, 'loops')
    assert type(skeleton.loops) is list
    assert len(skeleton.loops) == len(coordinates)


def test_roundtrip(skeleton, tmpdir):
    dirname = tmpdir.mkdir('field_checkpoint')
    filename = pathlib.Path(dirname) / 'test-save.asdf'
    skeleton.to_asdf(filename)
    skeleton_2 = synthesizAR.Skeleton.from_asdf(filename)
    assert len(skeleton.loops) == len(skeleton_2.loops)
    for i in range(len(skeleton.loops)):
        l1 = skeleton.loops[i].coordinate.cartesian.xyz
        l2 = skeleton_2.loops[i].coordinate.cartesian.xyz
        assert u.allclose(l2, l1, rtol=1e-9)
