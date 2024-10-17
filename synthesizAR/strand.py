"""
Strand object for holding field-aligned coordinates and quantities
"""
import astropy.units as u
import numpy as np
import sunpy.sun.constants as sun_const
import zarr

from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d, splev, splprep
from sunpy.coordinates import HeliographicStonyhurst

__all__ = ['Strand']


class Strand:
    r"""
    Container for geometric and thermodynamic properties of a coronal loop

    Parameters
    ----------
    name : `str`
    coordinate : `astropy.coordinates.SkyCoord`
        Coordinates in the field-aligned direction; should be able to transform to HEEQ
    field_strength : `astropy.units.Quantity`
        Scalar magnetic field strength along the loop. If not specified, defaults
        to NaN with same shape as ``coordinate``.
    cross_sectional_area : `astropy.units.Quantity`, optional
        Cross-sectional area of the loop. If not specified, defaults to :math:`10^{14}\,\mathrm{cm}^2`.
        This is used to compute the filling factor when computing the line-of-sight intensity.
    model_results_filename : `str`, optional
        Path to file where model results are stored. This will be set by
        `~synthesizAR.Skeleton` when the model results are loaded.

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord
    >>> import synthesizAR
    >>> coordinate = SkyCoord(x=[1,4]*u.Mm, y=[2,5]*u.Mm, z=[3,6]*u.Mm, frame='heliographic_stonyhurst', representation_type='cartesian')
    >>> field_strength = u.Quantity([100,200], 'gauss')
    >>> strand = synthesizAR.Strand('coronal_strand', coordinate, field_strength)
    >>> strand
    synthesizAR Strand
    ------------------
    Name : coronal_strand
    Loop full-length, L : 5.196 Mm
    Footpoints : (1 Mm,2 Mm,3 Mm),(4 Mm,5 Mm,6 Mm)
    Maximum field strength : 200.00 G
    Simulation Type: None
    """

    @u.quantity_input
    def __init__(self,
                 name,
                 coordinate,
                 field_strength: u.G=None,
                 cross_sectional_area: u.cm**2=None,
                 model_results_filename=None):
        self.name = name
        self.coordinate = coordinate
        self.field_strength = field_strength
        self.cross_sectional_area = cross_sectional_area
        self.model_results_filename = model_results_filename
        if self.coordinate.shape != self.field_strength.shape:
            raise ValueError('Coordinates and field strength must have same shape.')

    @property
    def zarr_root(self):
        """
        Root object to Zarr filestore for model results
        """
        if self.model_results_filename is not None:
            return zarr.open(store=self.model_results_filename, mode='r')

    def __repr__(self):
        f0 = f'{self.coordinate.x[0]:.3g},{self.coordinate.y[0]:.3g},{self.coordinate.z[0]:.3g}'
        f1 = f'{self.coordinate.x[-1]:.3g},{self.coordinate.y[-1]:.3g},{self.coordinate.z[-1]:.3g}'
        return f'''synthesizAR Strand
------------------
Name : {self.name}
Loop full-length, L : {self.length.to(u.Mm):.3f}
Footpoints : ({f0}),({f1})
Maximum field strength : {np.max(self.field_strength):.2f}
Simulation Type: {self.simulation_type}'''

    def _interpolate_to_center_coordinate(self, y, **kwargs):
        """
        Interpolate a quantity defined at the cell edges to the center of the coordinate
        """
        return interp1d(self.field_aligned_coordinate.to(u.Mm).value, y.value, **kwargs)(
            self.field_aligned_coordinate_center.to(u.Mm).value) * y.unit

    @property
    def coordinate(self):
        return self._coordinate

    @coordinate.setter
    def coordinate(self, value):
        self._coordinate = value.transform_to(HeliographicStonyhurst)
        self._coordinate.representation_type = 'cartesian'

    @property
    @u.quantity_input
    def coordinate_center(self):
        """
        The coordinates of the centers of the bins.
        """
        tck, _ = splprep(self.coordinate.cartesian.xyz.value, u=self.field_aligned_coordinate_norm)
        x, y, z = splev(self.field_aligned_coordinate_center/self.length, tck)
        unit = self.coordinate.cartesian.xyz.unit
        return SkyCoord(
            x=x*unit,
            y=y*unit,
            z=z*unit,
            frame=self.coordinate.frame,
            representation_type=self.coordinate.representation_type
        )

    @property
    @u.quantity_input
    def coordinate_direction(self) -> u.dimensionless_unscaled:
        """
        Unit vector indicating the direction of :math:`s` in HEEQ
        """
        grad_xyz = np.gradient(self.coordinate.cartesian.xyz.value, axis=1)
        return u.Quantity(grad_xyz / np.linalg.norm(grad_xyz, axis=0))

    @property
    @u.quantity_input
    def coordinate_direction_center(self) -> u.dimensionless_unscaled:
        """
        Unit vector indicating the direction of :math:`s` in HEEQ evaluated
        at the center of the grids
        """
        return self._interpolate_to_center_coordinate(self.coordinate_direction, axis=1)

    @property
    @u.quantity_input
    def field_aligned_coordinate(self) -> u.cm:
        """
        Field-aligned coordinate :math:`s` such that :math:`0<s<L`.

        Technically, the first :math:`N` cells are the left edges of
        each grid cell and the :math:`N+1` cell is the right edge of
        the last grid cell.
        """
        return np.append(0., np.linalg.norm(np.diff(self.coordinate.cartesian.xyz.value, axis=1),
                                            axis=0).cumsum()) * self.coordinate.cartesian.xyz.unit

    @property
    @u.quantity_input
    def field_aligned_coordinate_norm(self) -> u.dimensionless_unscaled:
        """
        Field-aligned coordinate normalized to the total loop length
        """
        return self.field_aligned_coordinate / self.length

    @property
    @u.quantity_input
    def field_aligned_coordinate_edge(self) -> u.cm:
        """
        Left cell edge of the field-aligned coordinate cells
        """
        return self.field_aligned_coordinate[:1]

    @property
    @u.quantity_input
    def field_aligned_coordinate_center(self) -> u.cm:
        """
        Center of the field-aligned coordinate cells
        """
        # Avoid doing this calculation twice
        s = self.field_aligned_coordinate
        return (s[:-1] + s[1:])/2

    @property
    def n_s(self):
        """
        Numb of points along the field-aligned coordinate
        """
        return self.field_aligned_coordinate_center.shape[0]

    @property
    @u.quantity_input
    def field_aligned_coordinate_center_norm(self) -> u.dimensionless_unscaled:
        """
        Center of the field-aligned coordinate normalized to
        the total loop length
        """
        return self.field_aligned_coordinate_center / self.length

    @property
    @u.quantity_input
    def field_aligned_coordinate_width(self) -> u.cm:
        """
        Width of each field-aligned coordinate grid cell
        """
        return np.diff(self.field_aligned_coordinate)

    @property
    @u.quantity_input
    def cross_sectional_area(self) -> u.cm**2:
        """
        Cross-sectional area of each field-aligned coordinate grid cell
        """
        return self._cross_sectional_area

    @cross_sectional_area.setter
    def cross_sectional_area(self, value):
        if value is None:
            value = 1e14*u.cm**2
        # Ensure that is always has the same shape as the coordinate
        self._cross_sectional_area = value * np.ones(self.field_aligned_coordinate.shape)

    @property
    @u.quantity_input
    def cross_sectional_area_center(self) -> u.cm**2:
        return self._interpolate_to_center_coordinate(self.cross_sectional_area)

    @property
    @u.quantity_input
    def field_strength(self) -> u.G:
        return self._field_strength

    @field_strength.setter
    def field_strength(self, value):
        if value is None:
            self._field_strength = np.nan * np.ones(self.coordinate.shape) * u.G
        else:
            self._field_strength = value

    @property
    @u.quantity_input
    def field_strength_center(self) -> u.G:
        return self._interpolate_to_center_coordinate(self.field_strength)

    @property
    @u.quantity_input
    def field_strength_average(self) -> u.G:
        return np.average(self.field_strength_center, weights=self.field_aligned_coordinate_width)

    @property
    @u.quantity_input
    def length(self) -> u.cm:
        """
        Loop full-length :math:`L`, from footpoint to footpoint
        """
        return self.field_aligned_coordinate_width.sum()

    @property
    @u.quantity_input
    def gravity(self) -> u.cm / (u.s**2):
        """
        Gravitational acceleration in the field-aligned direction.
        """
        r_hat = u.Quantity(np.stack([
            np.cos(self.coordinate.spherical.lat)*np.cos(self.coordinate.spherical.lon),
            np.cos(self.coordinate.spherical.lat)*np.sin(self.coordinate.spherical.lon),
            np.sin(self.coordinate.spherical.lat)
        ]))
        r_hat_dot_s_hat = (r_hat * self.coordinate_direction).sum(axis=0)
        return -sun_const.surface_gravity * (
            (sun_const.radius / self.coordinate.spherical.distance)**2) * r_hat_dot_s_hat

    @property
    def simulation_type(self) -> str:
        """
        The model used to produce the field-aligned hydrodynamic quantities
        """
        try:
            return self._simulation_type
        except AttributeError:
            if self.model_results_filename is None:
                return None
            else:
                return self.zarr_root[self.name].attrs['simulation_type']

    @property
    @u.quantity_input
    def velocity_xyz(self) -> u.cm/u.s:
        """Cartesian velocity components in HEEQ as function of loop coordinate and time"""
        s_hat = self.coordinate_direction_center
        return u.Quantity([self.velocity * s_hat[0],
                           self.velocity * s_hat[1],
                           self.velocity * s_hat[2]])

    def _get_quantity(self, quantity):
        try:
            q = getattr(self, f'_{quantity}')
        except AttributeError:
            dset = self.zarr_root[f'{self.name}/{quantity}']
            return u.Quantity(dset, dset.attrs['unit'])
        else:
            return q

    def get_ionization_fraction(self, ion):
        """
        Return the ionization fraction for a particular ion.
        """
        return self._get_quantity(f'ionization_fraction/{ion.ion_name}')


def add_property(name, unit, doc):
    """
    Auto-generate properties for various pieces of data
    """
    @u.quantity_input
    def property_template(self) -> u.Unit(unit):
        return self._get_quantity(name)
    property_template.__doc__ = doc
    property_template.__name__ = name
    setattr(Strand, property_template.__name__, property(property_template))


properties = [
    ('time', 's', 'Simulation time'),
    ('electron_temperature', 'K', 'Electron temperature as function of loop coordinate and time.'),
    ('ion_temperature', 'K', 'Ion temperature as function of loop coordinate and time.'),
    ('density', 'cm-3', 'Density as function of loop coordinate and time.'),
    ('velocity', 'cm s-1', 'Velocity as function of loop coordinate and time.'),
]
for p in properties:
    add_property(*p)
