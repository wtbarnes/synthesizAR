"""
Tests for Strand object
"""
import astropy.units as u
import numpy as np
import pytest

from astropy.coordinates import SkyCoord
from sunpy.coordinates import get_earth

import synthesizAR

from synthesizAR.models import semi_circular_loop


def test_loop_name(semi_circle_strand):
    assert hasattr(semi_circle_strand, 'name')
    assert semi_circle_strand.name == 'test'


@pytest.mark.parametrize(
    'name',
    ['coordinate',
     'coordinate_center'],
)
def test_loop_coordinate_properties(semi_circle_strand, name):
    assert hasattr(semi_circle_strand, name)
    assert isinstance(getattr(semi_circle_strand, name), SkyCoord)


@pytest.mark.parametrize(
    'name',
    ['field_strength',
     'field_strength_center',
     'cross_sectional_area',
     'cross_sectional_area_center',
     'coordinate_direction',
     'coordinate_direction_center',
     'field_aligned_coordinate',
     'field_aligned_coordinate_norm',
     'field_aligned_coordinate_edge',
     'field_aligned_coordinate_center',
     'field_aligned_coordinate_center_norm',
     'field_aligned_coordinate_width',
     'length',
     'gravity'],
)
def test_loop_quantity_properties(semi_circle_strand, name):
    assert hasattr(semi_circle_strand, name)
    assert isinstance(getattr(semi_circle_strand, name), u.Quantity)


@pytest.fixture
def field_aligned_coordinate():
    return np.linspace(0, 50, 100) * u.Mm


@pytest.fixture
def simple_strand(field_aligned_coordinate):
    observer = get_earth('2020-01-01')
    coord = semi_circular_loop(s=field_aligned_coordinate,
                               observer=observer)
    return synthesizAR.Strand('test', coord)


def test_loop_field_aligned_coordinate(simple_strand,
                                       field_aligned_coordinate):
    assert u.allclose(simple_strand.field_aligned_coordinate,
                      field_aligned_coordinate,
                      rtol=1e-4)


def test_loop_length(simple_strand, field_aligned_coordinate):
    assert u.allclose(simple_strand.length,
                      field_aligned_coordinate[-1],
                      rtol=1e-4)
