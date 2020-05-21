"""
Field extrapolation methods for computing 3D vector magnetic fields from LOS magnetograms
"""
import numpy as np
from scipy.interpolate import griddata
import astropy.units as u
import numba
from astropy.utils.console import ProgressBar

from synthesizAR.util import SpatialPair
from synthesizAR.visualize import plot_fieldlines
from .helpers import from_local, to_local, magnetic_field_to_yt_dataset
from .fieldlines import trace_fieldlines

__all__ = ['PotentialField']


class PotentialField(object):
    """
    Local (~1 AR) potential field extrapolation class

    Using the oblique Schmidt method as described in [1]_, compute a potential magnetic vector field
    from an observed LOS magnetogram. Note that this method is only valid for spatial scales
    :math:`\lesssim 1` active region.

    Parameters
    ----------
    magnetogram : `~sunpy.map.Map`
    width_z : `~astropy.units.Quantity`
    shape_z : `~astropy.units.Quantity`

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
        self.delta = SpatialPair(x=self.width.x/self.shape.x,
                                 y=self.width.y/self.shape.y,
                                 z=self.width.z/self.shape.z)

    @u.quantity_input
    def as_yt(self, B_field):
        """
        Wrapper around `~synthesizAR.extrapolate.magnetic_field_to_yt_dataset`
        """
        return magnetic_field_to_yt_dataset(B_field.x, B_field.y, B_field.z,
                                            self.range.x, self.range.y, self.range.z)

    @u.quantity_input
    def trace_fieldlines(self, B_field, number_fieldlines, **kwargs):
        """
        Trace field lines through vector magnetic field.

        This is a wrapper around `~synthesizAR.extrapolate.trace_fieldlines` and
        accepts all of the same keyword arguments. Note that here the field lines are
        automatically converted to the HEEQ coordinate system.

        Parameters
        ----------
        B_field : `~synthesizAR.util.SpatialPair`
        number_fieldlines : `int`

        Returns
        -------
        coordinates : `list`
            `~astropy.coordinates.SkyCoord` objects giving coordinates for all field lines
        field_strengths : `list`
            `~astropy.units.Quantity` for magnitude of :math:`B(s)` for each field line
        """
        ds = self.as_yt(B_field)
        lower_boundary = self.project_boundary(self.range.x, self.range.y).value
        lines = trace_fieldlines(ds, number_fieldlines, lower_boundary=lower_boundary, **kwargs)
        coordinates, field_strengths = [], []
        with ProgressBar(len(lines), ipython_widget=kwargs.get('notebook', True)) as progress:
            for l, b in lines:
                l = u.Quantity(l, self.range.x.unit)
                l_heeq = from_local(l[:, 0], l[:, 1], l[:, 2], self.magnetogram.center)
                coordinates.append(l_heeq)
                field_strengths.append(u.Quantity(b, str(ds.r['Bz'].units)))
                if kwargs.get('verbose', True):  # Optionally suppress progress bar for tests
                    progress.update()

        return coordinates, field_strengths

    def _calculate_range(self, magnetogram):
        left_corner = to_local(magnetogram.bottom_left_coord, magnetogram.center)
        right_corner = to_local(magnetogram.top_right_coord, magnetogram.center)
        range_x = u.Quantity([left_corner[0][0], right_corner[0][0]])
        range_y = u.Quantity([left_corner[1][0], right_corner[1][0]])
        return range_x, range_y
    
    def project_boundary(self, range_x, range_y):
        """
        Project the magnetogram onto a plane defined by the surface normal at the center of the
        magnetogram.
        """
        # Get all points in local, rotated coordinate system
        p_y, p_x = np.indices((int(self.shape.x.value), int(self.shape.y.value)))
        pixels = u.Quantity([(i_x, i_y) for i_x, i_y in zip(p_x.flatten(), p_y.flatten())], 'pixel')
        world_coords = self.magnetogram.pixel_to_world(pixels[:, 0], pixels[:, 1])
        local_x, local_y, _ = to_local(world_coords, self.magnetogram.center)
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
        los = to_local(self.magnetogram.observer_coordinate, self.magnetogram.center)
        return np.squeeze(u.Quantity(los))

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
        shape = SpatialPair(x=int(self.shape.x.value),
                            y=int(self.shape.y.value),
                            z=int(self.shape.z.value))
        phi = np.zeros((shape.x, shape.y, shape.z))
        phi = _calculate_phi_numba(phi, boundary, delta, shape, z_depth, l_hat)
        return phi * u.Unit(self.magnetogram.meta['bunit']) * self.delta.x.unit * (1. * u.pixel)

    @u.quantity_input
    def calculate_field(self, phi: u.G * u.cm):
        """
        Compute vector magnetic field.

        Calculate the vector magnetic field using the current-free approximation,

        .. math::
            \\vec{B} = -\\nabla\phi

        The gradient is computed numerically using a five-point stencil,

        .. math::
            \\frac{\partial B}{\partial x_i} \\approx -\left(\\frac{-B_{x_i}(x_i + 2\Delta x_i) + 8B_{x_i}(x_i + \Delta x_i) - 8B_{x_i}(x_i - \Delta x_i) + B_{x_i}(x_i - 2\Delta x_i)}{12\Delta x_i}\\right)

        Parameters
        ----------
        phi : `~astropy.units.Quantity`

        Returns
        -------
        B_field : `~synthesizAR.util.SpatialPair`
            x, y, and z components of the vector magnetic field in 3D
        """
        Bx = u.Quantity(np.zeros(phi.shape), self.magnetogram.meta['bunit'])
        By = u.Quantity(np.zeros(phi.shape), self.magnetogram.meta['bunit'])
        Bz = u.Quantity(np.zeros(phi.shape), self.magnetogram.meta['bunit'])
        # Take gradient using a five-point stencil
        Bx[2:-2, 2:-2, 2:-2] = -(phi[2:-2, :-4, 2:-2] - 8.*phi[2:-2, 1:-3, 2:-2]
                                 + 8.*phi[2:-2, 3:-1, 2:-2]
                                 - phi[2:-2, 4:, 2:-2])/12./(self.delta.x * 1. * u.pixel)
        By[2:-2, 2:-2, 2:-2] = -(phi[:-4, 2:-2, 2:-2] - 8.*phi[1:-3, 2:-2, 2:-2]
                                 + 8.*phi[3:-1, 2:-2, 2:-2]
                                 - phi[4:, 2:-2, 2:-2])/12./(self.delta.y * 1. * u.pixel)
        Bz[2:-2, 2:-2, 2:-2] = -(phi[2:-2, 2:-2, :-4] - 8.*phi[2:-2, 2:-2, 1:-3]
                                 + 8.*phi[2:-2, 2:-2, 3:-1]
                                 - phi[2:-2, 2:-2, 4:])/12./(self.delta.z * 1. * u.pixel)
        # Set boundary conditions such that the last two cells in either direction in each dimension
        # are the same as the preceding cell.
        for Bfield in (Bx, By, Bz):
            for j in [0, 1]:
                Bfield[j, :, :] = Bfield[2, :, :]
                Bfield[:, j, :] = Bfield[:, 2, :]
                Bfield[:, :, j] = Bfield[:, :, 2]
            for j in [-2, -1]:
                Bfield[j, :, :] = Bfield[-3, :, :]
                Bfield[:, j, :] = Bfield[:, -3, :]
                Bfield[:, :, j] = Bfield[:, :, -3]

        return SpatialPair(x=Bx, y=By, z=Bz)

    def extrapolate(self):
        phi = self.calculate_phi()
        bfield = self.calculate_field(phi)
        return bfield

    def peek(self, fieldlines, **kwargs):
        plot_fieldlines(*[fl for fl, _ in fieldlines], magnetogram=self.magnetogram, **kwargs)


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _calculate_phi_numba(phi, boundary, delta, shape, z_depth, l_hat):
    for i in numba.prange(shape.x):
        for j in numba.prange(shape.y):
            for k in numba.prange(shape.z):
                Rz = k * delta.z - z_depth
                lzRz = l_hat[2] * Rz
                factor = 1. / (2. * np.pi) * delta.x * delta.y
                for i_prime in range(shape.x):
                    for j_prime in range(shape.y):
                        Rx = delta.x * (i - i_prime)
                        Ry = delta.y * (j - j_prime)
                        R_mag = np.sqrt(Rx**2 + Ry**2 + Rz**2)
                        num = l_hat[2] + Rz / R_mag
                        denom = R_mag + lzRz + Rx*l_hat[0] + Ry*l_hat[1]
                        green = num / denom
                        phi[j, i, k] += boundary[j_prime, i_prime] * green * factor

    return phi
