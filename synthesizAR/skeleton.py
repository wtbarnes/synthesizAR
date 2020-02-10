"""
Container for fieldlines in three-dimensional magnetic skeleton
"""
from scipy.interpolate import splev, splrep
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.utils.console import ProgressBar
import asdf
import h5py

from synthesizAR import Loop
from synthesizAR.extrapolate import peek_fieldlines


class Skeleton(object):
    """
    Construct magnetic field skeleton fieldlines

    Parameters
    ----------
    loops : `list`
        List of `Loop` objects

    Examples
    --------
    >>> import synthesizAR
    >>> import astropy.units as u
    >>> loop = synthesizAR.Loop('loop', SkyCoord(x=[1,4]*u.Mm, y=[2,5]*u.Mm, z=[3,6]*u.Mm,frame='heliographic_stonyhurst', representation_type='cartesian'), [1e2,1e3] * u.G)
    >>> field = synthesizAR.Skeleton([loop,])
    """

    def __init__(self, loops):
        self.loops = loops

    @classmethod
    def from_coordinates(cls, coordinates, field_strengths):
        """
        Construct `Skeleton` from list of coordinates and field strengths

        Parameters
        ----------
        coordinates : `list`
            List of `~astropy.coordinates.SkyCoord` loop coordinates
        field_strengths : `list`
            List of `~astropy.units.Quantity` scalar magnetic field strength along the loop
        """
        loops = []
        for i, (coord, mag) in enumerate(zip(coordinates, field_strengths)):
            loops.append(Loop(f'loop{i:06d}', coord, mag))
        return cls(loops)

    def __repr__(self):
        return f'''synthesizAR Skeleton Object
------------------------
Number of loops: {len(self.loops)}'''

    def to_asdf(self, filename):
        """
        Serialize this instance of `Skeleton` to an ASDF file
        """
        tree = {}
        for l in self.loops:
            tree[l.name] = {
                'field_strength': l.field_strength,
                'coordinates': l.coordinates,
                'model_results_filename': l.model_results_filename,
            }
        with asdf.AsdfFile(tree) as asdf_file:
            asdf_file.write_to(filename)

    @classmethod
    def from_asdf(cls, filename):
        """
        Restore a `Skeleton` instance from an ASDF file

        Examples
        --------
        >>> import synthesizAR
        >>> restored_field = synthesizAR.Skeleton.from_asdf('/path/to/skeleton.asdf') # doctest: +SKIP
        """
        exclude_keys = ['asdf_library', 'history']
        loops = []
        with asdf.open(filename) as af:
            for k in af.keys():
                if k in exclude_keys:
                    continue
                model_results_filename = af.tree[k].get('model_results_filename', None)
                loops.append(Loop(
                    k,
                    SkyCoord(af.tree[k]['coordinates']),
                    af.tree[k]['field_strength'],
                    model_results_filename=model_results_filename,
                ))
        return cls(loops)

    @property
    def all_coordinates(self):
        """
        Coordinates for all loops in the skeleton
        """
        return SkyCoord([l.coordinate for l in self.loops],
                        frame=self.loops[0].coordinate.frame,
                        representation_type=self.loops[0].coordinate.representation_type)

    def peek(self, magnetogram, **kwargs):
        """
        Show extracted fieldlines overlaid on magnetogram.
        """
        fieldlines = [loop.coordinates for loop in self.loops]
        peek_fieldlines(magnetogram, fieldlines, **kwargs)

    def configure_loop_simulations(self, interface, **kwargs):
        """
        Configure hydrodynamic simulations for each loop object
        """
        with ProgressBar(len(self.loops), ipython_widget=kwargs.get('notebook', True)) as progress:
            for loop in self.loops:
                interface.configure_input(loop)
                progress.update()

    def load_loop_simulations(self, interface, filename, **kwargs):
        """
        Load in loop parameters from hydrodynamic results.
        """
        with h5py.File(filename, 'w') as hf:
            with ProgressBar(len(self.loops),
                             ipython_widget=kwargs.get('notebook', True),) as progress:
                for loop in self.loops:
                    # Load in parameters from interface
                    (time, electron_temperature, ion_temperature,
                     density, velocity) = interface.load_results(loop)
                    # Convert velocity to loop coordinate system
                    # NOTE: the direction is evaluated at the left edges + the last right edge.
                    # But the velocity is evaluated at the center of each cell so we need
                    # to interpolate the direction to the cell centers for each component
                    s = loop.field_aligned_coordinate.to(u.Mm).value
                    s_center = loop.field_aligned_coordinate_center.to(u.Mm).value
                    s_hat = loop.coordinate_direction
                    velocity_x = velocity * splev(s_center, splrep(s, s_hat[0, :]))
                    velocity_y = velocity * splev(s_center, splrep(s, s_hat[1, :]))
                    velocity_z = velocity * splev(s_center, splrep(s, s_hat[2, :]))
                    # Write to file
                    loop.model_results_filename = filename
                    grp = hf.create_group(loop.name)
                    grp.attrs['simulation_type'] = interface.name
                    # time
                    dset_time = grp.create_dataset('time', data=time.value)
                    dset_time.attrs['unit'] = time.unit.to_string()
                    # electron temperature
                    dset_electron_temperature = grp.create_dataset(
                        'electron_temperature', data=electron_temperature.value)
                    dset_electron_temperature.attrs['unit'] = electron_temperature.unit.to_string()
                    # ion temperature
                    dset_ion_temperature = grp.create_dataset(
                        'ion_temperature', data=ion_temperature.value)
                    dset_ion_temperature.attrs['unit'] = ion_temperature.unit.to_string()
                    # number density
                    dset_density = grp.create_dataset('density', data=density.value)
                    dset_density.attrs['unit'] = density.unit.to_string()
                    # field-aligned velocity
                    dset_velocity = grp.create_dataset('velocity', data=velocity.value)
                    dset_velocity.attrs['unit'] = velocity.unit.to_string()
                    dset_velocity.attrs['note'] = 'Velocity in the field-aligned direction'
                    # Cartesian xyz velocity
                    dset_velocity_x = grp.create_dataset('velocity_x', data=velocity_x.value)
                    dset_velocity_x.attrs['unit'] = velocity_x.unit.to_string()
                    dset_velocity_x.attrs['note'] = 'x-component of velocity in HEEQ coordinates'
                    dset_velocity_y = grp.create_dataset('velocity_y', data=velocity_y.value)
                    dset_velocity_y.attrs['unit'] = velocity_y.unit.to_string()
                    dset_velocity_y.attrs['note'] = 'y-component of velocity in HEEQ coordinates'
                    dset_velocity_z = grp.create_dataset('velocity_z', data=velocity_z.value)
                    dset_velocity_z.attrs['unit'] = velocity_z.unit.to_string()
                    dset_velocity_z.attrs['note'] = 'z-component of velocity in HEEQ coordinates'
                    progress.update()
