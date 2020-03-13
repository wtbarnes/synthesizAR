"""
Container for fieldlines in three-dimensional magnetic skeleton
"""
import numpy as np
from scipy.interpolate import splev, splprep, interp1d
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.utils.console import ProgressBar
import asdf
import zarr

from synthesizAR import Loop
from synthesizAR.visualize import peek_fieldlines
from synthesizAR.atomic import Element


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
                'coordinate': l.coordinate,
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
        with asdf.open(filename, mode='r', copy_arrays=True) as af:
            for k in af.keys():
                if k in exclude_keys:
                    continue
                model_results_filename = af.tree[k].get('model_results_filename', None)
                loops.append(Loop(
                    k,
                    SkyCoord(af.tree[k]['coordinate']),
                    af.tree[k]['field_strength'],
                    model_results_filename=model_results_filename,
                ))
        return cls(loops)

    @u.quantity_input
    def refine_loops(self, delta_s: u.cm):
        """
        Interpolate loop coordinates and field strengths to a specified spatial resolution
        and return a new `Skeleton` object.

        This can be important in order to ensure that an adequate number of points are used
        to represent each fieldline when binning intensities onto the instrument grid.
        """
        new_loops = []
        for l in self.loops:
            tck, _ = splprep(l.coordinate.cartesian.xyz.value, u=l.field_aligned_coordinate_norm)
            new_s = np.arange(0, l.length.to(u.Mm).value, delta_s.to(u.Mm).value) * u.Mm
            x, y, z = splev((new_s/l.length).decompose(), tck)
            unit = l.coordinate.cartesian.xyz.unit
            new_coord = SkyCoord(x=x*unit, y=y*unit, z=z*unit, frame=l.coordinate.frame,
                                 representation_type=l.coordinate.representation_type)
            f_B = interp1d(l.field_aligned_coordinate.to(u.Mm), l.field_strength)
            new_field_strength = f_B(new_s.to(u.Mm)) * l.field_strength.unit
            new_loops.append(Loop(l.name, new_coord, new_field_strength,
                                  model_results_filename=l.model_results_filename))

        return Skeleton(new_loops)

    @property
    def all_coordinates(self):
        """
        Coordinates for all loops in the skeleton.

        .. note:: This should be treated as a collection of points and NOT a
                  continuous structure.
        """
        return SkyCoord([l.coordinate for l in self.loops],
                        frame=self.loops[0].coordinate.frame,
                        representation_type=self.loops[0].coordinate.representation_type)

    @property
    def all_coordinates_centers(self):
        """
        Coordinates for all grid cell centers of all loops in the skeleton

        .. note:: This should be treated as a collection of points and NOT a 
                  continuous structure.
        """
        return SkyCoord([l.coordinate_center for l in self.loops],
                        frame=self.loops[0].coordinate_center.frame,
                        representation_type=self.loops[0].coordinate_center.representation_type)

    def interpolate_loop_coordinates(self):
        """
        Interpolate all loop coordinates to ensure each is represented by a suitable number
        of points.
        """
        raise NotImplementedError

    def peek(self, magnetogram, **kwargs):
        """
        Plot loop coordinates overlaid on magnetogram.
        """
        peek_fieldlines(magnetogram, [l.coordinate for l in self.loops], **kwargs)

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
        root = zarr.open(store=filename, mode='w', **kwargs)
        with ProgressBar(len(self.loops)) as progress:
            for loop in self.loops:
                loop.model_results_filename = filename
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
                velocity_x = velocity * interp1d(s, s_hat[0, :])(s_center)
                velocity_y = velocity * interp1d(s, s_hat[1, :])(s_center)
                velocity_z = velocity * interp1d(s, s_hat[2, :])(s_center)
                # Write to file
                grp = root.create_group(loop.name)
                grp.attrs['simulation_type'] = interface.name
                # time
                dset_time = grp.create_dataset('time', data=time.value)
                dset_time.attrs['unit'] = time.unit.to_string()
                # NOTE: Set the chunk size such that accessing all entries for a given timestep
                # is the most efficient pattern.
                chunks = (None,) + s_center.shape
                # electron temperature
                dset_electron_temperature = grp.create_dataset(
                    'electron_temperature', data=electron_temperature.value, chunks=chunks)
                dset_electron_temperature.attrs['unit'] = electron_temperature.unit.to_string()
                # ion temperature
                dset_ion_temperature = grp.create_dataset(
                    'ion_temperature', data=ion_temperature.value, chunks=chunks)
                dset_ion_temperature.attrs['unit'] = ion_temperature.unit.to_string()
                # number density
                dset_density = grp.create_dataset('density', data=density.value, chunks=chunks)
                dset_density.attrs['unit'] = density.unit.to_string()
                # field-aligned velocity
                dset_velocity = grp.create_dataset('velocity', data=velocity.value, chunks=chunks)
                dset_velocity.attrs['unit'] = velocity.unit.to_string()
                dset_velocity.attrs['note'] = 'Velocity in the field-aligned direction'
                # Cartesian xyz velocity
                dset_velocity_x = grp.create_dataset(
                    'velocity_x', data=velocity_x.value, chunks=chunks)
                dset_velocity_x.attrs['unit'] = velocity_x.unit.to_string()
                dset_velocity_x.attrs['note'] = 'x-component of velocity in HEEQ coordinates'
                dset_velocity_y = grp.create_dataset(
                    'velocity_y', data=velocity_y.value, chunks=chunks)
                dset_velocity_y.attrs['unit'] = velocity_y.unit.to_string()
                dset_velocity_y.attrs['note'] = 'y-component of velocity in HEEQ coordinates'
                dset_velocity_z = grp.create_dataset(
                    'velocity_z', data=velocity_z.value, chunks=chunks)
                dset_velocity_z.attrs['unit'] = velocity_z.unit.to_string()
                dset_velocity_z.attrs['note'] = 'z-component of velocity in HEEQ coordinates'
                progress.update()

    def load_ionization_fractions(self, emission_model, interface=None, **kwargs):
        """
        Load the ionization fractions for each ion in the emission model.

        Parameters
        ----------
        emission_model : `synthesizAR.atomic.EmissionModel`
        interface : optional
            A model interface. Only necessary if loading the ionization fractions
            from the model

        If the model interface provides a method for loading the population fraction
        from the model, use that to get the population fractions. Otherwise, compute
        the ion population fractions in equilibrium. This should be done after
        calling `load_loop_simulations`.
        """
        root = zarr.open(store=self.loops[0].model_results_filename, mode='a', **kwargs)
        # Check if we can load from the model
        FROM_MODEL = False
        if interface is not None and hasattr(interface, 'load_ionization_fraction'):
            frac = interface.load_ionization_fraction(self.loops[0], emission_model[0])
            # Some models may optionally output the ionization fractions such that
            # they will have this method, but it may not return anything
            if frac is not None:
                FROM_MODEL = True
        # Get the unique elements from all of our ions
        elements = list(set([ion.element_name for ion in emission_model]))
        elements = [Element(e, emission_model.temperature) for e in elements]
        for el in elements:
            ions = [i for i in emission_model if i.element_name == el.element_name]
            if not FROM_MODEL:
                ioneq = el.equilibrium_ionization()
            for loop in self.loops:
                chunks = (None,) + loop.field_aligned_coordinate_center.shape
                if not FROM_MODEL:
                    frac_el = interp1d(el.temperature,
                                       ioneq,
                                       axis=0,
                                       kind='linear',
                                       fill_value='extrapolate')(loop.electron_temperature)
                if 'ionization_fraction' in root[loop.name]:
                    grp = root[f'{loop.name}/ionization_fraction']
                else:
                    grp = root[loop.name].create_group('ionization_fraction')
                for ion in ions:
                    if FROM_MODEL:
                        frac = interface.load_ionization_fraction(loop, ion)
                        desc = f'Ionization fraction of {ion.ion_name} as computed by {interface.name}'
                    else:
                        frac = frac_el[:, :, ion.charge_state]
                        desc = f'Ionization fraction of {ion.ion_name} in equilibrium.'
                    dset = grp.create_dataset(f'{ion.ion_name}', data=frac, chunks=chunks)
                    dset.attrs['unit'] = ''
                    dset.attrs['description'] = desc
