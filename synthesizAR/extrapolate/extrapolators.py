"""
Field extrapolation methods for computing 3D vector magnetic fields from LOS magnetograms
"""
import numpy as np
from scipy.interpolate import griddata
import astropy.units as u
import yt
import numba
from sunpy.coordinates.frames import Heliocentric
from sunpy.image.rescale import resample

from synthesizAR.util import SpatialPair, to_heeq

from .helpers import local_to_heeq, heeq_to_local, magnetic_field_to_yt_dataset
from .fieldlines import trace_fieldlines

__all__ = ['ObliqueSchmidt']


class ObliqueSchmidt(object):
    """
    Potential field extrapolation using the oblique Schmidt method as described in [1]_.

    Parameters
    ----------
    magnetogram : `sunpy.map.Map`
    width_z : `astropy.Quantity`
    shape_z : `astropy.Quantity`

    References
    ----------
    .. [1] Sakurai, T., 1981, SoPh, `76, 301 <http://adsabs.harvard.edu/abs/1982SoPh...76..301S>`_
    """
    
    @u.quantity_input
    def __init__(self, magnetogram, width_z: u.cm, shape_z: u.pixel):
        self.magnetogram = magnetogram
        self.shape = SpatialPair(x=magnetogram.dimensions.x, y=magnetogram.dimensions.y, z=shape_z)
        range_x, range_y = self._calculate_range(magnetogram)
        range_z = u.Quantity([0*u.cm, width_z])
        self.range = SpatialPair(x=range_x.to(u.cm), y=range_y.to(u.cm), z=range_z.to(u.cm))
        width_x = np.diff(range_x)[0]
        width_y = np.diff(range_y)[0]
        self.width = SpatialPair(x=width_x.to(u.cm), y=width_y.to(u.cm), z=width_z.to(u.cm))
        self.delta = SpatialPair(x=self.width.x/self.shape.x, y=self.width.y/self.shape.y, 
                                 z=self.width.z/self.shape.z)

    @u.quantity_input
    def as_yt(self, B_field):
        return magnetic_field_to_yt_dataset(B_field.x, B_field.y, B_field.z, self.range.x,
                                            self.range.y, self.range.z)

    @u.quantity_input
    def trace_fieldlines(self, B_field, number_fieldlines, **kwargs):
        """
        Convenience wrapper around the fieldline tracer

        See Also
        --------
        `synthesizAR.extrapolator.trace_fieldlines`
        """
        ds = self.as_yt(B_field)
        lower_boundary = self.project_boundary(self.range.x, self.range.y).value
        lines = trace_fieldlines(ds, number_fieldlines, lower_boundary=lower_boundary, **kwargs)
        fieldlines = []
        for l, b in lines:
            l = u.Quantity(l, self.range.x.unit)
            l_heeq = u.Quantity(local_to_heeq(*l.T, self.magnetogram.center)).T
            m = u.Quantity(b, str(ds.r['Bz'].units))
            fieldlines.append((l_heeq, m))

        return fieldlines
        
    def _calculate_range(self, magnetogram):
        left_corner = heeq_to_local(*u.Quantity(to_heeq(magnetogram.bottom_left_coord))[:, np.newaxis],
                                    magnetogram.center)
        right_corner = heeq_to_local(*u.Quantity(to_heeq(magnetogram.top_right_coord))[:, np.newaxis],
                                     magnetogram.center)
        range_x = u.Quantity([left_corner[0][0], right_corner[0][0]])
        range_y = u.Quantity([left_corner[1][0], right_corner[1][0]])
        return range_x, range_y
    
    def project_boundary(self, range_x, range_y):
        """
        Project the magnetogram onto a plane normal to the surface
        """
        # Get all points in local, rotated coordinate system
        p_y, p_x = np.indices((int(self.shape.x.value), int(self.shape.y.value)))
        pixels = u.Quantity([(i_x, i_y) for i_x, i_y in zip(p_x.flatten(), p_y.flatten())], 'pixel')
        world_coords = self.magnetogram.pixel_to_world(pixels[:, 0], pixels[:, 1])
        local_x, local_y, _ = heeq_to_local(*u.Quantity(to_heeq(world_coords)), self.magnetogram.center)
        # Flatten
        points = np.stack([local_x.to(u.cm).value, local_y.to(u.cm).value], axis=1)
        values = u.Quantity(self.magnetogram.data, self.magnetogram.meta['bunit']).value.flatten()
        # Interpolate
        x_new = np.linspace(range_x[0], range_x[1], int(self.shape.x.value))
        y_new = np.linspace(range_y[0], range_y[1], int(self.shape.y.value))
        x_grid, y_grid = np.meshgrid(x_new.to(u.cm).value, y_new.to(u.cm).value)
        boundary_interp = griddata(points, values, (x_grid, y_grid), fill_value=0.)
        
        return u.Quantity(boundary_interp, self.magnetogram.meta['bunit'])
    
    @property
    def line_of_sight(self):
        """
        LOS vector in the local coordinate system
        """
        l_hat = self.magnetogram.observer_coordinate.transform_to(
                    Heliocentric(observer=self.magnetogram.observer_coordinate))
        l_hat = heeq_to_local(*u.Quantity(to_heeq(l_hat))[:, np.newaxis], self.magnetogram.center)
        return np.squeeze(u.Quantity(l_hat))
        
    def calculate_phi(self):
        """
        Calculate potential
        """
        # Set up grid
        y_grid, x_grid = np.indices((int(self.shape.x.value), int(self.shape.y.value)))
        x_grid = x_grid*self.delta.x.value
        y_grid = y_grid*self.delta.y.value
        z_depth = -self.delta.z.value/np.sqrt(2.*np.pi)
        # Project lower boundary
        boundary = self.project_boundary(self.range.x, self.range.y).value
        # Normalized LOS vector
        l_hat = (self.line_of_sight/np.sqrt((self.line_of_sight**2).sum())).value
        # Calculate phi
        delta = SpatialPair(x=self.delta.x.value, y=self.delta.y.value, z=self.delta.z.value)
        shape = SpatialPair(x=int(self.shape.x.value), y=int(self.shape.y.value),
                            z=int(self.shape.z.value))
        phi = numba.decorators.jit(self._calculate_phi, nopython=True)(boundary, delta, shape,
                                                                       z_depth, l_hat)
                    
        return phi * u.Unit(self.magnetogram.meta['bunit']) * self.delta.x.unit * (1. * u.pixel)
    
    @staticmethod
    def _calculate_phi(boundary, delta, shape, z_depth, l_hat):
        phi = np.empty((shape.x, shape.y, shape.z))
        for i in range(shape.x):
            for j in range(shape.y):
                for k in range(shape.z):
                    x, y, z = i*delta.x, j*delta.y, k*delta.z
                    for i_prime in range(shape.x):
                        for j_prime in range(shape.y):
                            x_prime, y_prime = i_prime*delta.x, j_prime*delta.y
                            green = greens_function(x, y, z, x_prime, y_prime, z_depth, l_hat)
                            phi[j, i, k] += boundary[j_prime, i_prime] * green * delta.x * delta.y
                    
        return phi
                    
    def calculate_field(self, phi):
        """
        Compute vector magnetic field.

        Calculate the vector magnetic field using the current-free approximation,

        .. math::

           \\vec{B} = -\\nabla\phi

        The gradient is computed numerically using a five-point stencil.
        """
        Bfield = u.Quantity(np.zeros(phi.shape + (3,)), self.magnetogram.meta['bunit'])
        # Take gradient--indexed as x,y,z in 4th dimension
        Bfield[2:-2, 2:-2, 2:-2, 0] = -(phi[:-4, 2:-2, 2:-2] - 8.*phi[1:-3, 2:-2, 2:-2] 
                                        + 8.*phi[3:-1, 2:-2, 2:-2]
                                        - phi[4:, 2:-2, 2:-2])/12./(self.delta.x * 1. * u.pixel)
        Bfield[2:-2, 2:-2, 2:-2, 1] = -(phi[2:-2, :-4, 2:-2] - 8.*phi[2:-2, 1:-3, 2:-2]
                                        + 8.*phi[2:-2, 3:-1, 2:-2] 
                                        - phi[2:-2, 4:, 2:-2])/12./(self.delta.y * 1. * u.pixel)
        Bfield[2:-2, 2:-2, 2:-2, 2] = -(phi[2:-2, 2:-2, :-4] - 8.*phi[2:-2, 2:-2, 1:-3]
                                        + 8.*phi[2:-2, 2:-2, 3:-1]
                                        - phi[2:-2, 2:-2, 4:])/12./(self.delta.z * 1. * u.pixel)
        # Set boundary conditions
        for i in range(3):
            for j in [0, 1]:
                Bfield[j, :, :, i] = Bfield[2, :, :, i]
                Bfield[:, j, :, i] = Bfield[:, 2, :, i]
                Bfield[:, :, j, i] = Bfield[:, :, 2, i]
            for j in [-2, -1]:
                Bfield[j, :, :, i] = Bfield[-3, :, :, i]
                Bfield[:, j, :, i] = Bfield[:, -3, :, i]
                Bfield[:, :, j, i] = Bfield[:, :, -3, i]
                
        return SpatialPair(x=Bfield[:, :, :, 1], y=Bfield[:, :, :, 0], z=Bfield[:, :, :, 2])
    
    def extrapolate(self):
        phi = self.calculate_phi()
        bfield = self.calculate_field(phi)
        return bfield

    
@numba.jit(nopython=True)
def greens_function(x, y, z, x_grid, y_grid, z_depth, l_hat):
    Rx = x - x_grid
    Ry = y - y_grid
    Rz = z - z_depth
    R_mag = np.sqrt(Rx**2 + Ry**2 + Rz**2)
    l_dot_R = l_hat[0] * Rx + l_hat[1] * Ry + l_hat[2] * Rz
    mu_dot_R = Rz - l_dot_R * l_hat[2]
    term1 = l_hat[2] / R_mag
    term2 = mu_dot_R / (R_mag * (R_mag + l_dot_R))
    return 1. / (2. * np.pi) * (term1 + term2)
