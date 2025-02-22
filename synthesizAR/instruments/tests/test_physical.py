"""
Tests for instrument classes that compute physical quantities
"""
import astropy.units as u
import numpy as np
import sunpy.map

from synthesizAR.instruments import InstrumentDEM, InstrumentLOSVelocity

# smoke tests to ensure physical instrument classes work
# NOTE: This does not test correctness of the output

def test_dem_instrument(skeleton_with_model, earth_observer):
    temperature_bin_edges = np.geomspace(1e5, 1e7, num=21,endpoint=True)*u.K
    dem = InstrumentDEM([0,]*u.s,
                        earth_observer,
                        [1,1]*u.arcsec/u.pixel,
                        temperature_bin_edges=temperature_bin_edges)
    dem_maps = dem.observe(skeleton_with_model)
    for k, v in dem_maps.items():
        assert len(v) == 1
        assert isinstance(v[0], sunpy.map.GenericMap)
        assert v[0].unit.is_equivalent('cm-5')


def test_velocity_instrument(skeleton_with_model, earth_observer):
    los_vel = InstrumentLOSVelocity([0,]*u.s, earth_observer, [1,1]*u.arcsec/u.pixel)
    vel_maps = los_vel.observe(skeleton_with_model)
    for k, v in vel_maps.items():
        assert len(v) == 1
        assert isinstance(v[0], sunpy.map.GenericMap)
        assert v[0].unit.is_equivalent('km s-1')
