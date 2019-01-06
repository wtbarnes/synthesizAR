"""
Class for an individual loop structure that is part of a larger active region.
"""
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicStonyhurst
import h5py

from synthesizAR.util import get_keys


class Loop(object):
    """
    Container for geometric and thermodynamic properties of a coronal loop

    Parameters
    ----------
    name : `str`
    coordinates : `astropy.coordinates.SkyCoord`
        Loop coordinates; should be able to transform to HEEQ
    field_strength : `astropy.units.Quantity`
        Scalar magnetic field strength along the loop

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord
    >>> import synthesizAR
    >>> coordinate = SkyCoord(x=[1,4]*u.Mm, y=[2,5]*u.Mm, z=[3,6]*u.Mm, frame='heliographic_stonyhurst', representation='cartesian')
    >>> field_strength = u.Quantity([100,200], 'gauss')
    >>> loop = synthesizAR.Loop('coronal_loop', coordinate, field_strength)
    >>> loop
    Name : coronal_loop
    Loop full-length, 2L : 5.196 Mm
    Footpoints : (1 Mm,2 Mm,3 Mm),(4 Mm,5 Mm,6 Mm)
    Maximum field strength : 200.00 G
    """

    @u.quantity_input
    def __init__(self, name, coordinates=None, field_strength=None, coords_savefile=None):
        self.name = name
        if coordinates is not None:
            self._coordinates = coordinates.transform_to(HeliographicStonyhurst)
            self._coordinates.representation = 'cartesian'
        if field_strength is not None:
            self._field_strength = field_strength.to(u.gauss)
        self.coords_savefile = coords_savefile

    def __repr__(self):
        f0 = f'{self.coordinates.x[0]:.3g},{self.coordinates.y[0]:.3g},{self.coordinates.z[0]:.3g}'
        f1 = f'{self.coordinates.x[-1]:.3g},{self.coordinates.y[-1]:.3g},{self.coordinates.z[-1]:.3g}'
        return f'''Name : {self.name}
Loop full-length, 2L : {self.full_length.to(u.Mm):.3f}
Footpoints : ({f0}),({f1})
Maximum field strength : {np.max(self.field_strength):.2f}'''

    @property
    def coordinates(self):
        """
        World coordinates of loop
        """
        if hasattr(self, '_coordinates'):
            return self._coordinates
        else:
            with h5py.File(self.coords_savefile, 'r') as hf:
                grp = hf[self.name]
                x = u.Quantity(grp['coordinates'][0, :],
                               get_keys(grp['coordinates'].attrs, ('unit', 'units')))
                y = u.Quantity(grp['coordinates'][1, :],
                               get_keys(grp['coordinates'].attrs, ('unit', 'units')))
                z = u.Quantity(grp['coordinates'][2, :],
                               get_keys(grp['coordinates'].attrs, ('unit', 'units')))
                return SkyCoord(x=x, y=y, z=z, frame=HeliographicStonyhurst,
                                representation='cartesian')

    @property
    @u.quantity_input
    def field_strength(self) -> u.G:
        """
        Magnetic field strength as a function of the field-aligned coordinate
        """
        if hasattr(self, '_field_strength'):
            return self._field_strength
        else:
            with h5py.File(self.coords_savefile, 'r') as hf:
                grp = hf[self.name]
                return u.Quantity(grp['field_strength'],
                                  get_keys(grp['field_strength'].attrs, ('unit', 'units')))

    @property
    @u.quantity_input
    def field_aligned_coordinate(self) -> u.cm:
        """
        Field-aligned coordinate :math:`s` such that :math:`0<s<L`
        """
        return np.append(0., np.linalg.norm(np.diff(self.coordinates.cartesian.xyz.value, axis=1),
                                            axis=0).cumsum()) * self.coordinates.cartesian.xyz.unit

    @property
    @u.quantity_input
    def full_length(self) -> u.cm:
        """
        Loop full-length :math:`2L`, from footpoint to footpoint
        """
        return np.diff(self.field_aligned_coordinate).sum()

    @property
    @u.quantity_input
    def time(self) -> u.s:
        """
        Simulation time
        """
        with h5py.File(self.parameters_savefile, 'r') as hf:
            dset = hf['/'.join([self.name, 'time'])]
            time = u.Quantity(dset, get_keys(dset.attrs, ('unit', 'units')))
        return time

    @property
    @u.quantity_input
    def electron_temperature(self) -> u.K:
        """
        Loop electron temperature as function of coordinate and time.
        """
        with h5py.File(self.parameters_savefile, 'r') as hf:
            dset = hf['/'.join([self.name, 'electron_temperature'])]
            temperature = u.Quantity(dset, get_keys(dset.attrs, ('unit', 'units')))
        return temperature

    @property
    @u.quantity_input
    def ion_temperature(self) -> u.K:
        """
        Loop ion temperature as function of coordinate and time.
        """
        with h5py.File(self.parameters_savefile, 'r') as hf:
            dset = hf['/'.join([self.name, 'ion_temperature'])]
            temperature = u.Quantity(dset, get_keys(dset.attrs, ('unit', 'units')))
        return temperature

    @property
    @u.quantity_input
    def density(self) -> u.cm**(-3):
        """
        Loop density as a function of coordinate and time.
        """
        with h5py.File(self.parameters_savefile, 'r') as hf:
            dset = hf['/'.join([self.name, 'density'])]
            density = u.Quantity(dset, get_keys(dset.attrs, ('unit', 'units')))
        return density

    @property
    @u.quantity_input
    def velocity(self) -> u.cm/u.s:
        """
        Velcoity in the field-aligned direction of the loop as a function of loop coordinate and
        time.
        """
        with h5py.File(self.parameters_savefile, 'r') as hf:
            dset = hf['/'.join([self.name, 'velocity'])]
            velocity = u.Quantity(dset, get_keys(dset.attrs, ('unit', 'units')))
        return velocity

    @property
    @u.quantity_input
    def velocity_x(self) -> u.cm/u.s:
        """
        X-component of velocity in the HEEQ Cartesian coordinate system as a function of time.
        """
        with h5py.File(self.parameters_savefile, 'r') as hf:
            dset = hf['/'.join([self.name, 'velocity_x'])]
            velocity = u.Quantity(dset, get_keys(dset.attrs, ('unit', 'units')))
        return velocity

    @property
    @u.quantity_input
    def velocity_y(self) -> u.cm/u.s:
        """
        Y-component of velocity in the HEEQ Cartesian coordinate system as a function of time.
        """
        with h5py.File(self.parameters_savefile, 'r') as hf:
            dset = hf['/'.join([self.name, 'velocity_y'])]
            velocity = u.Quantity(dset, get_keys(dset.attrs, ('unit', 'units')))
        return velocity

    @property
    @u.quantity_input
    def velocity_z(self) -> u.cm/u.s:
        """
        Z-component of velocity in the HEEQ Cartesian coordinate system as a function of time.
        """
        with h5py.File(self.parameters_savefile, 'r') as hf:
            dset = hf['/'.join([self.name, 'velocity_z'])]
            velocity = u.Quantity(dset, get_keys(dset.attrs, ('unit', 'units')))
        return velocity
