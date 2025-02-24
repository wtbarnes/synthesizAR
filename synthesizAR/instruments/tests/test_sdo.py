"""
Tests for SDO instrument classes
"""
import astropy.units as u
import numpy as np
import pytest
import sunpy.map

from synthesizAR.instruments.sdo import InstrumentSDOAIA


@pytest.mark.remote_data
def test_aia_instrument(skeleton_with_model, earth_observer):
    aia = InstrumentSDOAIA([0,]*u.s, earth_observer)
    aia_maps = aia.observe(skeleton_with_model)
    for k, v in aia_maps.items():
        assert len(v) == len(aia.observing_time)
        for m in v:
            assert isinstance(m, sunpy.map.GenericMap)
            assert m.unit.is_equivalent('DN pix-1 s-1')

@pytest.mark.remote_data
@pytest.mark.parametrize(('input', 'result'), [
    ([0,]*u.s, [0,]*u.s),
    ([0,1]*u.s, [0,]*u.s),
    ([0,25]*u.s, [0,12,24]*u.s),
    (np.arange(2000,3000,60)*u.s, np.arange(2000,3000,60)*u.s),
])
def test_aia_observing_times(earth_observer, input, result):
    aia = InstrumentSDOAIA(input, earth_observer)
    assert u.allclose(aia.observing_time, result)


@pytest.mark.remote_data
def test_aia_resolution(earth_observer):
    aia = InstrumentSDOAIA([0,]*u.s, earth_observer)
    assert u.allclose(aia.resolution, [0.600698, 0.600698]*u.arcsec/u.pix)
    aia = InstrumentSDOAIA([0,]*u.s,
                           earth_observer,
                           resolution=[1.2,1.2] * u.arcsec / u.pix)
    assert u.allclose(aia.resolution, [1.2,1.2] * u.arcsec / u.pix)
