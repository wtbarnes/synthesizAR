"""
Tests for Skeleton object
"""
import astropy.units as u
import pathlib
import pytest

from astropy.coordinates import SkyCoord

import synthesizAR


def test_skeleton_has_strands(bare_skeleton):
    assert hasattr(bare_skeleton, 'strands')
    assert isinstance(bare_skeleton.strands, list)
    for l in bare_skeleton.strands:
        assert isinstance(l, synthesizAR.Strand)


def test_create_skeleton_from_coords(bare_skeleton):
    coords = [l.coordinate for l in bare_skeleton.strands]
    field_strengths = [l.field_strength for l in bare_skeleton.strands]
    skeleton = synthesizAR.Skeleton.from_coordinates(coords, field_strengths=field_strengths)
    assert hasattr(skeleton, 'strands')
    assert isinstance(skeleton.strands, list)


def test_roundtrip(bare_skeleton, tmpdir):
    dirname = tmpdir.mkdir('field_checkpoint')
    filename = pathlib.Path(dirname) / 'test-save.asdf'
    bare_skeleton.to_asdf(filename)
    skeleton_2 = synthesizAR.Skeleton.from_asdf(filename)
    assert len(bare_skeleton.strands) == len(skeleton_2.strands)
    for i in range(len(bare_skeleton.strands)):
        l1 = bare_skeleton.strands[i].coordinate.cartesian.xyz
        l2 = skeleton_2.strands[i].coordinate.cartesian.xyz
        assert u.allclose(l2, l1, rtol=1e-9)
        assert skeleton_2.strands[i].model_results_filename is None


def test_roundtrip_skeleton_with_model(skeleton_with_model, tmpdir):
    # Add an arbitrary results filename to each strand to make sure
    # that serializes correctly
    for l in skeleton_with_model.strands:
        l.model_results_filename = 'foo/bar.zarr'
    dirname = tmpdir.mkdir('model_skeleton_checkpoint')
    filename = pathlib.Path(dirname) / 'test-save.asdf'
    skeleton_with_model.to_asdf(filename)
    skeleton_2 = synthesizAR.Skeleton.from_asdf(filename)
    assert len(skeleton_with_model.strands) == len(skeleton_2.strands)
    for i in range(len(skeleton_with_model.strands)):
        assert skeleton_2.strands[i].model_results_filename == pathlib.Path('foo/bar.zarr')


def test_refine_loops(bare_skeleton):
    bare_skeleton_refined = bare_skeleton.refine_strands(1*u.Mm)
    assert isinstance(bare_skeleton_refined, synthesizAR.Skeleton)
    assert len(bare_skeleton_refined.strands) == len(bare_skeleton.strands)


@pytest.mark.parametrize(
    'name',
    ['all_coordinates_centers',
     'all_coordinates'],
)
def test_coordinate_properties(bare_skeleton, name):
    assert hasattr(bare_skeleton, name)
    assert isinstance(getattr(bare_skeleton, name), SkyCoord)


def test_loops_have_model_type(skeleton_with_model):
    for l in skeleton_with_model.strands:
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
    for l in skeleton_with_model.strands:
        assert isinstance(getattr(l, name), u.Quantity)
