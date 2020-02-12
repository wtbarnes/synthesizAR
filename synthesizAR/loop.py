"""
Loop object for holding field-aligned coordinates and quantities
"""
import numpy as np
from scipy.interpolate import splprep, splev
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicStonyhurst
import zarr


class Loop(object):
    """
    Container for geometric and thermodynamic properties of a coronal loop

    Parameters
    ----------
    name : `str`
    coordinate : `astropy.coordinates.SkyCoord`
        Loop coordinates; should be able to transform to HEEQ
    field_strength : `astropy.units.Quantity`
        Scalar magnetic field strength along the loop
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
    >>> loop = synthesizAR.Loop('coronal_loop', coordinate, field_strength)
    >>> loop
    Name : coronal_loop
    Loop full-length, L : 5.196 Mm
    Footpoints : (1 Mm,2 Mm,3 Mm),(4 Mm,5 Mm,6 Mm)
    Maximum field strength : 200.00 G
    """

    @u.quantity_input
    def __init__(self, name, coordinate, field_strength: u.G, model_results_filename=None):
        if coordinate.shape != field_strength.shape:
            raise ValueError('Coordinates and field strength must have same shape.')
        self.name = name
        self.coordinate = coordinate.transform_to(HeliographicStonyhurst)
        self.coordinate.representation_type = 'cartesian'
        self.field_strength = field_strength
        self.model_results_filename = model_results_filename

    def __repr__(self):
        f0 = f'{self.coordinate.x[0]:.3g},{self.coordinate.y[0]:.3g},{self.coordinate.z[0]:.3g}'
        f1 = f'{self.coordinate.x[-1]:.3g},{self.coordinate.y[-1]:.3g},{self.coordinate.z[-1]:.3g}'
        return f'''synthesizAR Loop
----------------
Name : {self.name}
Loop full-length, L : {self.length.to(u.Mm):.3f}
Footpoints : ({f0}),({f1})
Maximum field strength : {np.max(self.field_strength):.2f}
Simulation Type: {self.simulation_type}'''

    @property
    @u.quantity_input
    def coordinate_center(self):
        """
        The coordinates of the centers of the bins.
        """
        tck, _ = splprep(self.coordinate.cartesian.xyz.value, u=self.field_aligned_coordinate_norm)
        x, y, z = splev(self.field_aligned_coordinate_center/self.length, tck)
        return SkyCoord(x=x, y=y, z=z, frame=self.coordinate.frame,
                        representation_type=self.coordinate.representation_type)

    @property
    @u.quantity_input
    def coordinate_direction(self):
        """
        Unit vector indicating the direction of :math:`s` in HEEQ
        """
        grad_xyz = np.gradient(self.coordinate.cartesian.xyz.value, axis=1)
        return grad_xyz / np.linalg.norm(grad_xyz, axis=0)

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
    @u.quantity_input
    def field_aligned_coordinate_width(self) -> u.cm:
        """
        Width of each field-aligned coordinate grid cell
        """
        return np.diff(self.field_aligned_coordinate)

    @property
    @u.quantity_input
    def length(self) -> u.cm:
        """
        Loop full-length :math:`L`, from footpoint to footpoint
        """
        return self.field_aligned_coordinate_width.sum()

    @property
    def simulation_type(self) -> str:
        """
        The model used to produce the field-aligned hydrodynamic quantities
        """
        if self.model_results_filename is None:
            return None
        else:
            with zarr.open(store=self.model_results_filename, mode='r') as root:
                return root[self.name].attrs['simulation_type']

    @property
    @u.quantity_input
    def time(self) -> u.s:
        """
        Simulation time
        """
        with zarr.open(store=self.model_results_filename, mode='r') as root:
            dset = root[f'{self.name}/time']
            return u.Quantity(dset, dset.attrs['unit'])

    @property
    @u.quantity_input
    def electron_temperature(self) -> u.K:
        """
        Loop electron temperature as function of coordinate and time.
        """
        with zarr.open(store=self.model_results_filename, mode='r') as root:
            dset = root[f'{self.name}/electron_temperature']
            return u.Quantity(dset, dset.attrs['unit'])

    @property
    @u.quantity_input
    def ion_temperature(self) -> u.K:
        """
        Loop ion temperature as function of coordinate and time.
        """
        with zarr.open(store=self.model_results_filename, mode='r') as root:
            dset = root[f'{self.name}/ion_temperature']
            return u.Quantity(dset, dset.attrs['unit'])

    @property
    @u.quantity_input
    def density(self) -> u.cm**(-3):
        """
        Loop density as a function of coordinate and time.
        """
        with zarr.open(store=self.model_results_filename, mode='r') as root:
            dset = root[f'{self.name}/density']
            return u.Quantity(dset, dset.attrs['unit'])

    @property
    @u.quantity_input
    def velocity(self) -> u.cm/u.s:
        """
        Velcoity in the field-aligned direction of the loop as a function of loop coordinate and
        time.
        """
        with zarr.open(store=self.model_results_filename, mode='r') as root:
            dset = root[f'{self.name}/velocity']
            return u.Quantity(dset, dset.attrs['unit'])

    @property
    @u.quantity_input
    def velocity_x(self) -> u.cm/u.s:
        """
        X-component of velocity in the HEEQ Cartesian coordinate system as a function of time.
        """
        with zarr.open(store=self.model_results_filename, mode='r') as root:
            dset = root[f'{self.name}/velocity_x']
            return u.Quantity(dset, dset.attrs['unit'])

    @property
    @u.quantity_input
    def velocity_y(self) -> u.cm/u.s:
        """
        Y-component of velocity in the HEEQ Cartesian coordinate system as a function of time.
        """
        with zarr.open(store=self.model_results_filename, mode='r') as root:
            dset = root[f'{self.name}/velocity_y']
            return u.Quantity(dset, dset.attrs['unit'])

    @property
    @u.quantity_input
    def velocity_z(self) -> u.cm/u.s:
        """
        Z-component of velocity in the HEEQ Cartesian coordinate system as a function of time.
        """
        with zarr.open(store=self.model_results_filename, mode='r') as root:
            dset = root[f'{self.name}/velocity_z']
            return u.Quantity(dset, dset.attrs['unit'])
