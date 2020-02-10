"""
Tests for Loop object
"""
import pytest
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicStonyhurst

from synthesizAR import Loop


x, y, z = [1, 2, 3]*u.Mm, [4, 5, 6]*u.Mm, [7, 8, 9]*u.Mm
B_mag = u.Quantity([100, 10, 100], 'Gauss')


@pytest.fixture
def loop():
    coords = SkyCoord(x=x, y=y, z=z, frame=HeliographicStonyhurst, representation_type='cartesian')
    return Loop('test', coords, B_mag)


def test_loop_name(loop):
    assert hasattr(loop, 'name')
    assert loop.name == 'test'


def test_loop_coordinates(loop):
    assert hasattr(loop, 'coordinates')
    assert isinstance(loop.coordinates, SkyCoord)


def test_loop_field_strength(loop):
    assert hasattr(loop, 'field_strength')
    assert isinstance(loop.field_strength, u.Quantity)
    assert np.all(loop.field_strength == B_mag)


def test_loop_field_aligned_coordinate(loop):
    assert hasattr(loop, 'field_aligned_coordinate')
    dx, dy, dz = np.diff(loop.coordinates.x), np.diff(loop.coordinates.y), np.diff(loop.coordinates.z)
    d = np.sqrt(dx**2 + dy**2 + dz**2).cumsum()
    s = u.Quantity(np.append(0, d.value), d.unit)
    assert np.all(loop.field_aligned_coordinate == s)


def test_loop_length(loop):
    assert hasattr(loop, 'length')
    dx, dy, dz = np.diff(loop.coordinates.x), np.diff(loop.coordinates.y), np.diff(loop.coordinates.z)
    assert loop.length == np.sqrt(dx**2 + dy**2 + dz**2).sum()
