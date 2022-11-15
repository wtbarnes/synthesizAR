"""
Tests for Skeleton object
"""
import pathlib

import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord

import synthesizAR


def test_skeleton_has_loops(bare_skeleton):
    assert hasattr(bare_skeleton, 'loops')
    assert type(bare_skeleton.loops) is list
    for l in bare_skeleton.loops:
        assert isinstance(l, synthesizAR.Loop)


def test_create_skeleton_from_coords(bare_skeleton):
    coords = [l.coordinate for l in bare_skeleton.loops]
    field_strengths = [l.field_strength for l in bare_skeleton.loops]
    skeleton = synthesizAR.Skeleton.from_coordinates(coords, field_strengths=field_strengths)
    assert hasattr(skeleton, 'loops')
    assert type(skeleton.loops) is list


def test_roundtrip(bare_skeleton, tmpdir):
    dirname = tmpdir.mkdir('field_checkpoint')
    filename = pathlib.Path(dirname) / 'test-save.asdf'
    bare_skeleton.to_asdf(filename)
    skeleton_2 = synthesizAR.Skeleton.from_asdf(filename)
    assert len(bare_skeleton.loops) == len(skeleton_2.loops)
    for i in range(len(bare_skeleton.loops)):
        l1 = bare_skeleton.loops[i].coordinate.cartesian.xyz
        l2 = skeleton_2.loops[i].coordinate.cartesian.xyz
        assert u.allclose(l2, l1, rtol=1e-9)


def test_refine_loops(bare_skeleton):
    bare_skeleton_refined = bare_skeleton.refine_loops(1*u.Mm)
    assert isinstance(bare_skeleton_refined, synthesizAR.Skeleton)
    assert len(bare_skeleton_refined.loops) == len(bare_skeleton.loops)


@pytest.mark.parametrize(
    'name',
    ['all_coordinates_centers',
     'all_coordinates'],
)
def test_coordinate_properties(bare_skeleton, name):
    assert hasattr(bare_skeleton, name)
    assert isinstance(getattr(bare_skeleton, name), SkyCoord)


def test_loops_have_model_type(skeleton_with_model):
    for l in skeleton_with_model.loops:
        assert hasattr(l, 'simulation_type')


@pytest.mark.parametrize(
    'name',
    ['time',
     'electron_temperature',
     'ion_temperature',
     'density',
     'velocity',
     'velocity_xyz']
)
def test_loops_have_model_quantities(skeleton_with_model, name):
    "These quantities exist only after an interface is defined"
    for l in skeleton_with_model.loops:
        assert isinstance(getattr(l, name), u.Quantity)
