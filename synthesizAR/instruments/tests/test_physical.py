"""
Tests for instrument classes that compute physical quantities
"""
import astropy.units as u
import ndcube
import numpy as np
import sunpy.map

from synthesizAR.instruments import (
    InstrumentDEM,
    InstrumentLOSVelocity,
    InstrumentTemperature,
    InstrumentVDEM,
)

# smoke tests to ensure physical instrument classes work
# NOTE: This does not test correctness of the output


def test_dem_instrument(skeleton_with_model, earth_observer):
    temperature_bin_edges = np.geomspace(1e5, 1e7, num=21,endpoint=True)*u.K
    dem = InstrumentDEM([0,]*u.s,
                        earth_observer,
                        [1,1]*u.arcsec/u.pixel,
                        temperature_bin_edges=temperature_bin_edges)
    dem_maps = dem.observe(skeleton_with_model)
    assert len(dem_maps) == temperature_bin_edges.shape[0]-1
    for k, v in dem_maps.items():
        assert len(v) == 1
        assert isinstance(v[0], sunpy.map.GenericMap)
        assert v[0].unit.is_equivalent('cm-5')
    dem_cube = dem.maps_to_cube(dem_maps, 0)
    assert isinstance(dem_cube, ndcube.NDCube)


def test_vdem_instrument(skeleton_with_time_dependent_model, earth_observer):
    temperature_bin_edges=10**np.arange(5.6,6.8,0.2)*u.K
    velocity_bin_edges=np.arange(-20,30,10)*u.km/u.s
    vdem = InstrumentVDEM([50]*u.s,
                          earth_observer,
                          [1,1]*u.arcsec/u.pixel,
                          temperature_bin_edges=temperature_bin_edges,
                          velocity_bin_edges=velocity_bin_edges)
    vdem_maps = vdem.observe(skeleton_with_time_dependent_model)
    assert len(vdem_maps) == len(vdem.channels)
    for k, v in vdem_maps.items():
        assert len(v) == 1
        assert isinstance(v[0], sunpy.map.GenericMap)
        assert v[0].unit.is_equivalent('cm-5')
    vdem_cube = vdem.maps_to_cube(vdem_maps, 0)
    assert isinstance(vdem_cube, ndcube.NDCube)


def test_velocity_instrument(skeleton_with_model, earth_observer):
    los_vel = InstrumentLOSVelocity([0,]*u.s, earth_observer, [1,1]*u.arcsec/u.pixel)
    vel_maps = los_vel.observe(skeleton_with_model)
    assert len(vel_maps) == 1
    for k, v in vel_maps.items():
        assert len(v) == 1
        assert isinstance(v[0], sunpy.map.GenericMap)
        assert v[0].unit.is_equivalent('km s-1')


def test_temperature_instrument(skeleton_with_model, earth_observer):
    temperature = InstrumentTemperature([0,]*u.s, earth_observer, [1,1]*u.arcsec/u.pixel)
    vel_maps = temperature.observe(skeleton_with_model)
    assert len(vel_maps) == 1
    for k, v in vel_maps.items():
        assert len(v) == 1
        assert isinstance(v[0], sunpy.map.GenericMap)
        assert v[0].unit.is_equivalent('K')
