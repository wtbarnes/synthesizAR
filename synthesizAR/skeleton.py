"""
Container for fieldlines in three-dimensional magnetic skeleton
"""
import asdf
import astropy.units as u
import numpy as np
import zarr

from astropy.coordinates import SkyCoord
from functools import cached_property
from scipy.interpolate import interp1d, splev, splprep

from synthesizAR import Strand
from synthesizAR.visualize import plot_fieldlines

__all__ = ['Skeleton']


class Skeleton:
    """
    Construct magnetic field skeleton fieldlines

    Parameters
    ----------
    strands : `list` of `~synthesizAR.Strand` objects
        List of objects containing the information about each strand in the magnetic
        skeleton.

    Examples
    --------
    >>> import synthesizAR
    >>> import astropy.units as u
    >>> strand = synthesizAR.Strand('strand', SkyCoord(x=[1,4]*u.Mm, y=[2,5]*u.Mm, z=[3,6]*u.Mm,frame='heliographic_stonyhurst', representation_type='cartesian'), [1e2,1e3] * u.G)
    >>> field = synthesizAR.Skeleton([strand,])
    """

    def __init__(self, strands):
        self.strands = strands

    @classmethod
    def from_coordinates(cls, coordinates, field_strengths=None, **kwargs):
        """
        Construct `Skeleton` from list of coordinates and field strengths

        Parameters
        ----------
        coordinates : `list` of `~astropy.coordinates.SkyCoord` objects
            Coordinate of each strand
        field_strengths : `list` of `~astropy.units.Quantity`
            Scalar magnetic field strength along the strand
        """
        strands = []
        if field_strengths is None:
            field_strengths = len(coordinates) * [None]
        for i, (coord, fs) in enumerate(zip(coordinates, field_strengths)):
            strands.append(Strand(f'strand{i:06d}', coord, field_strength=fs, **kwargs))
        return cls(strands)

    def __repr__(self):
        return f'''synthesizAR Skeleton Object
------------------------
Number of strands: {len(self.strands)}'''

    def to_asdf(self, filename):
        """
        Serialize this instance of `Skeleton` to an ASDF file
        """
        tree = {}
        for l in self.strands:
            tree[l.name] = {
                'field_strength': l.field_strength,
                'coordinate': l.coordinate,
                'cross_sectional_area': l.cross_sectional_area,
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
        strands = []
        with asdf.open(filename, mode='r', memmap=False) as af:
            for k in af.keys():
                if k in exclude_keys:
                    continue
                model_results_filename = af.tree[k].get('model_results_filename', None)
                cross_sectional_area = af.tree[k].get('cross_sectional_area', None)
                strand = Strand(k,
                                SkyCoord(af.tree[k]['coordinate']),
                                af.tree[k]['field_strength'],
                                cross_sectional_area=cross_sectional_area,
                                model_results_filename=model_results_filename)
                strands.append(strand)
        return cls(strands)

    @u.quantity_input
    def refine_strands(self, delta_s: u.cm, **kwargs):
        """
        Interpolate strand coordinates and field strengths to a specified spatial resolution
        and return a new `Skeleton` object.

        This can be important in order to ensure that an adequate number of points are used
        to represent each fieldline when binning intensities onto the instrument grid.
        """
        new_strands = []
        for l in self.strands:
            _l = self.refine_strand(l, delta_s, **kwargs)
            new_strands.append(_l)

        return Skeleton(new_strands)

    @staticmethod
    @u.quantity_input
    def refine_strand(strand, delta_s: u.cm, **kwargs):
        evkwargs = kwargs.get('evkwargs', {})
        prepkwargs = kwargs.get('prepkwargs', {})
        try:
            tck, _ = splprep(strand.coordinate.cartesian.xyz.value,
                             u=strand.field_aligned_coordinate_norm, **prepkwargs)
            new_s = np.arange(0, strand.length.to(u.Mm).value, delta_s.to(u.Mm).value) * u.Mm
            x, y, z = splev((new_s/strand.length).decompose(), tck, **evkwargs)
        except (ValueError, TypeError) as e:
            raise Exception(f'Failed to refine {strand.name}') from e
        unit = strand.coordinate.cartesian.xyz.unit
        new_coord = SkyCoord(x=x*unit,
                             y=y*unit,
                             z=z*unit,
                             frame=strand.coordinate.frame,
                             representation_type=strand.coordinate.representation_type)
        f_B = interp1d(strand.field_aligned_coordinate.to(u.Mm), strand.field_strength)
        new_field_strength = f_B(new_s.to(u.Mm)) * strand.field_strength.unit
        f_A = interp1d(strand.field_aligned_coordinate.to(u.Mm), strand.cross_sectional_area)
        new_area = f_A(new_s.to(u.Mm)) * strand.cross_sectional_area.unit
        return Strand(strand.name,
                      new_coord,
                      field_strength=new_field_strength,
                      cross_sectional_area=new_area,
                      model_results_filename=strand.model_results_filename)

    @property
    def all_coordinates(self):
        """
        Coordinates for all stands in the skeleton.

        .. note:: This should be treated as a collection of points and NOT a
                  continuous structure.
        """
        return SkyCoord([l.coordinate for l in self.strands],
                        frame=self.strands[0].coordinate.frame,
                        representation_type=self.strands[0].coordinate.representation_type)

    @property
    def all_coordinates_centers(self):
        """
        Coordinates for all grid cell centers of all strands in the skeleton

        .. note:: This should be treated as a collection of points and NOT a
                  continuous structure.
        """
        return SkyCoord([l.coordinate_center for l in self.strands],
                        frame=self.strands[0].coordinate_center.frame,
                        representation_type=self.strands[0].coordinate_center.representation_type)

    @cached_property
    def all_widths(self) -> u.cm:
        """
        Widths for all strands concatenated together
        """
        return np.concatenate([l.field_aligned_coordinate_width for l in self.strands])

    @cached_property
    def all_cross_sectional_areas(self) -> u.cm**2:
        """
        Cross-sectional areas for all strands concatenated together.

        .. note:: These are the cross-sectional areas evaluated at the center of the strand.
        """
        return np.concatenate([l.cross_sectional_area_center for l in self.strands])

    def peek(self, **kwargs):
        """
        Plot strand coordinates on the solar disk.

        See Also
        --------
        synthesizAR.visualize.plot_fieldlines
        """
        plot_fieldlines(*[_.coordinate for _ in self.strands], **kwargs)

    def configure_loop_simulations(self, interface, **kwargs):
        """
        Configure hydrodynamic simulations for each strand object
        """
        for strand in self.strands:
            interface.configure_input(strand, **kwargs)

    @staticmethod
    def _load_loop_simulation(strand, root=None, interface=None):
        # Load in parameters from interface
        (time, electron_temperature, ion_temperature,
            density, velocity) = interface.load_results(strand)
        # If no Zarr file is passed, set the quantites as attributes on the loops
        if root is None:
            strand._time = time
            strand._electron_temperature = electron_temperature
            strand._ion_temperature = ion_temperature
            strand._density = density
            strand._velocity = velocity
            strand._simulation_type = interface.name
        else:
            # Write to file
            grp = root.create_group(strand.name)
            grp.attrs['simulation_type'] = interface.name
            # time
            dset_time = grp.create_dataset('time', data=time.value)
            dset_time.attrs['unit'] = time.unit.to_string()
            # NOTE: Set the chunk size such that accessing all entries for a given timestep
            # is the most efficient pattern.
            chunks = (None,) + strand.field_aligned_coordinate_center.shape
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

    def load_loop_simulations(self, interface, filename=None, **kwargs):
        """
        Load results from hydrodynamic results.

        Parameters
        ----------
        interface : model interface object
            Interface to the hydrodynamic simulation from which to load the results
        filename : `str` or path-like
            Path to `zarr` store to write hydrodynamic results to
        """
        if filename is None:
            root = None
        else:
            root = zarr.open(store=filename, mode='w', **kwargs)
        for strand in self.strands:
            strand.model_results_filename = filename
        try:
            import distributed
            client = distributed.get_client()
        except (ImportError, ValueError):
            for l in self.strands:
                self._load_loop_simulation(l, root=root, interface=interface)
        else:
            status = client.map(
                self._load_loop_simulation,
                self.strands,
                root=root,
                interface=interface,
            )
            return status

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
        from fiasco import Element

        from synthesizAR.atomic import equilibrium_ionization

        root = zarr.open(store=self.strands[0].model_results_filename, mode='a', **kwargs)
        # Check if we can load from the model
        FROM_MODEL = False
        if interface is not None and hasattr(interface, 'load_ionization_fraction'):
            frac = interface.load_ionization_fraction(self.strands[0], emission_model[0])
            # Some models may optionally output the ionization fractions such that
            # they will have this method, but it may not return anything
            if frac is not None:
                FROM_MODEL = True
        # Get the unique elements from all of our ions
        element_names = list(set([ion.element_name for ion in emission_model]))
        for el_name in element_names:
            el = Element(el_name, emission_model.temperature)
            ions = [i for i in emission_model if i.element_name == el.element_name]
            for strand in self.strands:
                chunks = (None,) + strand.field_aligned_coordinate_center.shape
                if not FROM_MODEL:
                    frac_el = equilibrium_ionization(el, strand.electron_temperature)
                if 'ionization_fraction' in root[strand.name]:
                    grp = root[f'{strand.name}/ionization_fraction']
                else:
                    grp = root[strand.name].create_group('ionization_fraction')
                for ion in ions:
                    if FROM_MODEL:
                        frac = interface.load_ionization_fraction(strand, ion)
                        desc = f'{ion.ion_name} ionization fraction computed by {interface.name}'
                    else:
                        frac = frac_el[:, :, ion.charge_state]
                        desc = f'{ion.ion_name} ionization fraction in equilibrium.'
                    dset = grp.create_dataset(f'{ion.ion_name}', data=frac, chunks=chunks)
                    dset.attrs['unit'] = ''
                    dset.attrs['description'] = desc
