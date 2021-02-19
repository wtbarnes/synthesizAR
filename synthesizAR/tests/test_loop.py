"""
Tests for Loop object
"""
import pytest
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicStonyhurst

from synthesizAR import Loop
from synthesizAR.models import semi_circular_loop


@pytest.fixture
def strand():
    coords = semi_circular_loop(length=100*u.Mm)
    B_mag = np.nan * np.ones(coords.shape) * u.G
    return Loop('test', coords, B_mag)


def test_loop_name(strand):
    assert hasattr(strand, 'name')
    assert strand.name == 'test'


def test_loop_coordinates(strand):
    assert hasattr(strand, 'coordinate')
    assert isinstance(strand.coordinate, SkyCoord)


def test_loop_field_strength(strand):
    assert hasattr(strand, 'field_strength')
    assert isinstance(strand.field_strength, u.Quantity)


def test_loop_field_aligned_coordinate(strand):
    assert hasattr(strand, 'field_aligned_coordinate')
    d = np.sqrt((np.diff(strand.coordinate.cartesian.xyz, axis=1)**2).sum(axis=0)).cumsum()
    s = u.Quantity(np.append(0, d.value), d.unit)
    assert u.allclose(strand.field_aligned_coordinate, s, rtol=1e-15)


def test_loop_length(strand):
    assert hasattr(strand, 'length')
    length = np.sqrt((np.diff(strand.coordinate.cartesian.xyz, axis=1)**2).sum(axis=0)).sum()
    assert u.allclose(strand.length, length, rtol=1e-15)
