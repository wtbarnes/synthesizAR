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

import synthesizAR
from synthesizAR.models import semi_circular_loop


@pytest.fixture
def coordinates():
    obs = SkyCoord(
        lon=0*u.deg,
        lat=0*u.deg,
        radius=const.au,
        frame=HeliographicStonyhurst,
        obstime=astropy.time.Time.now(),
    )
    coord_1 = semi_circular_loop(length=50*u.Mm, observer=obs, n_points=500)
    coord_2 = semi_circular_loop(length=100*u.Mm, observer=obs)
    return coord_1, coord_2


@pytest.fixture
def field_strengths(coordinates):
    B_mag_1 = u.Quantity(len(coordinates[0])*[np.nan], 'Gauss')
    B_mag_2 = u.Quantity(len(coordinates[1])*[np.nan], 'Gauss')
    return B_mag_1, B_mag_2


@pytest.fixture
def skeleton(coordinates, field_strengths):
    return synthesizAR.Skeleton.from_coordinates(coordinates, field_strengths)


def test_field_loops(skeleton):
    assert hasattr(skeleton, 'loops')
    assert type(skeleton.loops) is list
    assert len(skeleton.loops) == 2


def test_field_from_loops(coordinates, field_strengths):
    loops = []
    for i, (coord, mag) in enumerate(zip(coordinates, field_strengths)):
        loops.append(synthesizAR.Loop(f'loop{i}', coord, mag))
    skeleton = synthesizAR.Skeleton(loops)
    assert hasattr(skeleton, 'loops')
    assert type(skeleton.loops) is list
    assert len(skeleton.loops) == 2


def test_roundtrip(skeleton, tmpdir):
    dirname = tmpdir.mkdir('field_checkpoint')
    filename = pathlib.Path(dirname) / 'test-save.asdf'
    skeleton.to_asdf(filename)
    skeleton_2 = synthesizAR.Skeleton.from_asdf(filename)
    assert len(skeleton.loops) == len(skeleton_2.loops)
    for i in range(len(skeleton.loops)):
        l1 = skeleton.loops[i].coordinate.cartesian.xyz
        l2 = skeleton_2.loops[i].coordinate.cartesian.xyz
        assert u.quantity.allclose(l2, l1, atol=0.*u.cm, rtol=1e-9)
