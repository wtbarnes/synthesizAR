"""
Helper routines for field extrapolation routines and dealing with vector field data
"""
import numpy as np
import astropy.units as u
import yt
import sunpy.coordinates

from synthesizAR.util import to_heeq

__all__ = ['magnetic_field_to_yt_dataset', 'local_to_heeq', 'heeq_to_local']


@u.quantity_input
def magnetic_field_to_yt_dataset(Bx: u.gauss, By: u.gauss, Bz: u.gauss, range_x: u.cm, 
                                 range_y: u.cm, range_z: u.cm):
    """
    Reshape vector magnetic field data into a yt dataset

    Parameters
    ----------
    Bx,By,Bz : `astropy.Quantity`
        3D arrays holding the x,y,z components of the extrapolated field
    range_x, range_y, range_z : `astropy.Quantity`
        Spatial range in the x,y,z dimensions of the grid
    """
    Bx = Bx.to(u.gauss)
    By = By.to(u.gauss)
    Bz = Bz.to(u.gauss)
    data = dict(
                Bx=(np.swapaxes(Bx.value, 0, 1), Bx.unit.to_string()),
                By=(np.swapaxes(By.value, 0, 1), By.unit.to_string()),
                Bz=(np.swapaxes(Bz.value, 0, 1), Bz.unit.to_string()))
    # Uniform, rectangular grid with symmetric bounds centered on 0
    bbox = np.array([range_x.to(u.cm).value,
                     range_y.to(u.cm).value,
                     range_z.to(u.cm).value])
    return yt.load_uniform_grid(data, data['Bx'][0].shape, bbox=bbox, length_unit=yt.units.cm,
                                geometry=('cartesian', ('x', 'y', 'z')))


@u.quantity_input
def local_to_heeq(x_local: u.cm, y_local: u.cm, z_local: u.cm, center):
    """
    Transform from a cartesian frame centered on the active region (with the z-axis parallel
    to the surface normal) to a HEEQ frame.
    """
    x_center, y_center, z_center = to_heeq(center)
    center = center.transform_to(sunpy.coordinates.frames.HeliographicStonyhurst)
    # NOTE: the coordinates are permuted because the local z-axis is parallel to the surface normal
    coord_heeq = (rotate_z(center.lon) @ rotate_y(-center.lat)
                  @ u.Quantity([z_local, x_local, y_local]))

    return coord_heeq[0, :] + x_center, coord_heeq[1, :] + y_center, coord_heeq[2, :] + z_center


@u.quantity_input
def heeq_to_local(x_heeq: u.cm, y_heeq: u.cm, z_heeq: u.cm, center):
    """
    Transform from HEEQ frame to a cartesian frame centered on the active region
    (with the z-axis normal to the surface).
    """
    x_center, y_center, z_center = to_heeq(center)
    center = center.transform_to(sunpy.coordinates.frames.HeliographicStonyhurst)
    x_heeq -= x_center
    y_heeq -= y_center
    z_heeq -= z_center
    coord_local = (rotate_y(center.lat) @ rotate_z(-center.lon)
                   @ u.Quantity([x_heeq, y_heeq, z_heeq]))
    # NOTE: the coordinates are permuted because the local z-axis is parallel to the surface normal
    return coord_local[1, :], coord_local[2, :], coord_local[0, :]


@u.quantity_input
def rotate_z(angle: u.radian):
    angle = angle.to(u.radian)
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])


@u.quantity_input
def rotate_y(angle: u.radian):
    angle = angle.to(u.radian)
    return np.array([[np.cos(angle), 0, np.sin(angle)],
                     [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])
