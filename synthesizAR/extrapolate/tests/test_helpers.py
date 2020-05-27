"""
Tests for field extrapolation helpers
"""
import pytest
import astropy.units as u
import astropy.time
import astropy.constants as const
from astropy.coordinates import SkyCoord
import sunpy
from sunpy.coordinates import HeliographicStonyhurst, Helioprojective

import synthesizAR
import synthesizAR.extrapolate


@pytest.fixture
def observer():
    return SkyCoord(
        lon=0.*u.deg,
        lat=0.*u.deg,
        radius=const.au,
        frame=HeliographicStonyhurst,
        obstime=astropy.time.Time.now(),
    )


@pytest.fixture
def corners(observer):
    hpc_frame = Helioprojective(observer=observer, obstime=observer.obstime)
    blc = SkyCoord(-150 * u.arcsec, -150 * u.arcsec, frame=hpc_frame)
    trc = SkyCoord(150 * u.arcsec, 150 * u.arcsec, frame=hpc_frame)
    return blc, trc


@pytest.fixture
def magnetogram(corners):
    arr_shape = [50, 50] * u.pixel
    centers = SkyCoord(Tx=[65, -65]*u.arcsec, Ty=[0, 0]*u.arcsec, frame=corners[0].frame)
    sigmas = u.Quantity([[15, 15], [15, 15]], 'arcsec')
    amplitudes = u.Quantity([1e3, -1e3], 'Gauss')
    magnetogram = synthesizAR.extrapolate.synthetic_magnetogram(
        *corners, arr_shape, centers, sigmas, amplitudes, observer=corners[0].observer)
    return magnetogram


def test_synthetic_magnetogram_shape(magnetogram):
    assert magnetogram.data.shape == (50, 50)
    assert magnetogram.dimensions.x == 50*u.pixel
    assert magnetogram.dimensions.y == 50*u.pixel


def test_synthetic_magnetogram_map_properties(magnetogram):
    assert isinstance(magnetogram, sunpy.map.GenericMap)
    # TODO: should test that the corners are as expected too
    # TODO: should test that observer and observer coordinate are as expected too
