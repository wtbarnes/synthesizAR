"""
Tests for Skeleton object
"""
import pytest
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicStonyhurst, Helioprojective

import synthesizAR
import synthesizAR.extrapolate


@pytest.fixture
def magnetogram():
    arr_shape = [50, 50] * u.pixel
    obs = SkyCoord(lon=0.*u.deg, lat=0.*u.deg, radius=const.au, frame=HeliographicStonyhurst)
    blc = SkyCoord(-150 * u.arcsec, -150 * u.arcsec, frame=Helioprojective(observer=obs))
    trc = SkyCoord(150 * u.arcsec, 150 * u.arcsec, frame=Helioprojective(observer=obs))
    centers = SkyCoord(Tx=[65, -65]*u.arcsec, Ty=[0, 0]*u.arcsec,
                       frame=Helioprojective(observer=obs))
    sigmas = u.Quantity([[15, 15], [15, 15]], 'arcsec')
    amplitudes = u.Quantity([1e3, -1e3], 'Gauss')
    magnetogram = synthesizAR.extrapolate.synthetic_magnetogram(
        blc, trc, arr_shape, centers, sigmas, amplitudes, observer=obs)
    return magnetogram


@pytest.fixture
def coordinates(magnetogram):
    coord_1 = SkyCoord(
        Tx=[65, 0, -65]*u.arcsec,
        Ty=[0, 65, 0]*u.arcsec,
        frame=magnetogram.coordinate_frame)
    coord_2 = SkyCoord(
        Tx=[65, 0, -65]*u.arcsec,
        Ty=[0, -65, 0]*u.arcsec,
        frame=magnetogram.coordinate_frame)
    return coord_1, coord_2


@pytest.fixture
def field_strengths():
    B_mag_1 = u.Quantity([-100, 10, -100], 'Gauss')
    B_mag_2 = u.Quantity([100, 10, -100], 'Gauss')
    return B_mag_1, B_mag_2


@pytest.fixture
def field(coordinates, field_strengths):
    return synthesizAR.Skeleton.from_coordinates(coordinates, field_strengths)


def test_field_loops(field):
    assert hasattr(field, 'loops')
    assert type(field.loops) is list
    assert len(field.loops) == 2


def test_field_from_loops(coordinates, field_strengths):
    loops = []
    for i, (coord, mag) in enumerate(zip(coordinates, field_strengths)):
        loops.append(synthesizAR.Loop(f'loop{i}', coord, mag))
    field = synthesizAR.Skeleton(loops)
    assert hasattr(field, 'loops')
    assert type(field.loops) is list
    assert len(field.loops) == 2


def test_roundtrip(field, tmpdir):
    dirname = tmpdir.mkdir('field_checkpoint')
    field.save(dirname)
    field_2 = synthesizAR.Skeleton.restore(dirname)
    assert len(field.loops) == len(field_2.loops)
    for i in range(len(field.loops)):
        l1 = field.loops[i].coordinates.cartesian.xyz.value
        l2 = field_2.loops[i].coordinates.cartesian.xyz.value
        assert np.all(np.isclose(l2, l1, atol=0., rtol=1e-9))
