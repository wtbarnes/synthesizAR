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
    >>> strand = synthesizAR.Strand('strand', SkyCoord(x=[1,4]*u.Mm, y=[2,5]*u.Mm, z=[3,6]*u.Mm, frame='heliographic_stonyhurst', representation_type='cartesian'), [1e2,1e3] * u.G)
    >>> field = synthesizAR.Skeleton([strand,])
    """

    def __init__(self, strands):
        self.strands = strands
        from synthesizAR import log
        self.log = log

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
        with asdf.open(filename, mode='r', memmap=False, lazy_load=False) as af:
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

    @u.quantity_input
    def get_chromosphere_mask(self, footpoint_height:u.Mm):
        "Returns result of `synthesizAR.Strand.get_chromosphere_mask` for all strands in skeleton."
        return np.concatenate([l.get_chromosphere_mask(footpoint_height) for l in self.strands])

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
    def _load_loop_simulation(strand, root=None, interface=None, emission_model=None):
        # Load in parameters from interface
        results = interface.load_results(strand, emission_model=emission_model)
        # If no Zarr file is passed, set the quantites as attributes on the loops
        if root is None:
            strand._simulation_type = interface.name
            for name, quantity in results.items():
                setattr(strand, f'_{name}', quantity)
        else:
            # Write to file
            grp = root.create_group(strand.name)
            grp.attrs['simulation_type'] = interface.name
            # NOTE: Set the chunk size such that accessing all entries for a given timestep
            # is the most efficient pattern.
            chunks = results['time'].shape + (1,)
            for name, quantity in results.items():
                dset = grp.create_array(name,
                                        data=quantity.value,
                                        chunks='auto' if name=='time' else chunks)
                dset.attrs['unit'] = quantity.unit.to_string()

    def load_loop_simulations(self, interface, filename=None, parallelize=False, emission_model=None):
        """
        Load results from hydrodynamic results.

        Parameters
        ----------
        interface : model interface object
            Interface to the hydrodynamic simulation from which to load the results
        filename : `str` or path-like
            Path to `zarr` store to write hydrodynamic results to
        parallelize : `bool`
            If True and a `distributed.Client` exists, load loop simulations in parallel.
        emission_model : `synthesizAR.atomic.EmissionModel`
            Emission model that specifies the ions used in the emission modeling process.
            This can be optionally specified in order to load the time-dependent ionization
            fractions for some models.
        """
        if filename is None:
            root = None
        else:
            root = zarr.open(store=filename, mode='w')
        for strand in self.strands:
            strand.model_results_filename = filename
        try:
            import distributed
            client = distributed.get_client()
        except (ImportError, ValueError):
            pass
        else:
            if parallelize:
                status = client.map(
                    self._load_loop_simulation,
                    self.strands,
                    root=root,
                    interface=interface,
                    emission_model=emission_model,
                )
                return status
        for l in self.strands:
            self.log.debug(f'Loading results for strand {l.name}')
            self._load_loop_simulation(l, root=root, interface=interface, emission_model=emission_model)
