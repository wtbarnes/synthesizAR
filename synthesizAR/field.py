"""
Active region object definition. This object holds all the important information about our
synthesized active region.
"""
import os
import sys
import logging
import datetime
import pickle
import glob
import functools

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sunpy.map
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.utils.console import ProgressBar
import h5py
import yt
import solarbextrapolation.map3dclasses
import solarbextrapolation.extrapolators

from synthesizAR.util import convert_angle_to_length, find_seed_points
from synthesizAR import Loop


class Skeleton(object):
    """
    Construct magnetic field skeleton from HMI fits file

    Parameters
    ----------
    hmi_fits_file : `str`
        Path to HMI magnetogram FITS file

    Examples
    --------

    Notes
    -----
    Right now, this class just accepts an HMI fits file. Could be adjusted to do the actual query
    as well.
    """

    def __init__(self, hmi_fits_file=None, **kwargs):
        self.logger = logging.getLogger(name=type(self).__name__)
        if hmi_fits_file is not None:
            tmp_map = sunpy.map.Map(hmi_fits_file)
            self.hmi_map = self._process_map(tmp_map, **kwargs)
        else:
            self.logger.warning('No HMI fits file supplied. A new HMI map object will not be created.')

    def __repr__(self):
        num_loops = ''
        sim_type = ''
        if hasattr(self, 'loops'):
            num_loops = len(self.loops)
        if hasattr(self, 'simulation_type'):
            sim_type = self.simulation_type
        return '''synthesizAR Field Object
------------------------
Number of loops: {num_loops}
Simulation Type: {sim_type}
Magnetogram Info:
-----------------
{hmi_map_info}

        '''.format(num_loops=num_loops, sim_type=sim_type, hmi_map_info=self.hmi_map.__repr__())

    def _process_map(self, tmp_map, crop=None, resample=None):
        """
        Rotate, crop and resample map if needed. Can do any other needed processing here too.

        Parameters
        ----------
        map : `~sunpy.map.Map`
            Original HMI map
        crop : `tuple` `[bottom_left_corner,top_right_corner]`, optional
            The lower left and upper right corners of the cropped map. Both should be of type `~astropy.units.Quantity` and have the same units as `map.xrange` and `map.yrange`
        resample : `~astropy.units.Quantity`, `[new_xdim,new_ydim]`, optional
            The new x- and y-dimensions of the resampled map, should have the same units as
            `map.dimensions.x` and `map.dimensions.y`
        """
        tmp_map = tmp_map.rotate()
        if crop is not None:
            bottom_left = SkyCoord(*crop[0], frame=tmp_map.coordinate_frame)
            top_right = SkyCoord(*crop[1], frame=tmp_map.coordinate_frame)
            tmp_map = tmp_map.submap(bottom_left, top_right)
        if resample is not None:
            tmp_map = tmp_map.resample(resample, method='linear')

        return tmp_map

    def save(self, savedir=None):
        """
        Save the components of the field object to be reloaded later.
        """
        if savedir is None:
            savedir = 'synthesizAR-{}-save_{}'.format(type(self).__name__,
                                                      (datetime.datetime.now()
                                                      .strftime('%Y%m%d-%H%M%S')))
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        # loops
        if not os.path.exists(os.path.join(savedir, 'loops')):
            os.makedirs(os.path.join(savedir, 'loops'))
        for l in self.loops:
            with open(os.path.join(savedir, 'loops', l.name+'.pickle'), 'wb') as f:
                pickle.dump(l, f)
        # streamlines
        with open(os.path.join(savedir, 'streamlines.pickle'), 'wb') as f:
            pickle.dump(self.streamlines, f)
        # sunpy maps
        self.hmi_map.save(os.path.join(savedir, 'hmi_map.fits'))
        # 3d extrapolated field
        with h5py.File(os.path.join(savedir, 'map_3d.h5'),'w') as hf:
            hf.create_dataset('map_3d', data=self._map_3d)
            zrange = hf.create_dataset('zrange', data=self._zrange.value)
            zrange.attrs['units'] = self._zrange.unit.to_string()

    @classmethod
    def restore(cls, savedir):
        """
        Restore the field from a set of serialized files

        Examples
        --------
        >>> import synthesizAR
        >>> restored_field = synthesizAR.Skeleton.restore_field('/path/to/restored/field/dir')
        """
        # loops
        loop_files = glob.glob(os.path.join(savedir, 'loops', '*'))
        loop_files = sorted([lf.split('/')[-1] for lf in loop_files],
                            key=lambda l: int(l.split('.')[0][4:]))
        loops = []
        for lf in loop_files:
            with open(os.path.join(savedir, 'loops', lf), 'rb') as f:
                loops.append(pickle.load(f))
        # streamlines
        with open(os.path.join(savedir, 'streamlines.pickle'), 'rb') as f:
            streamlines = pickle.load(f)
        # sunpy maps
        hmi_map = sunpy.map.Map(os.path.join(savedir, 'hmi_map.fits'))
        # 3d extapolated fields
        with h5py.File(os.path.join(savedir, 'map_3d.h5'), 'r') as hf:
            map_3d = np.array(hf['map_3d'])
            zrange = u.Quantity(hf['zrange'], hf['zrange'].attrs['units'])
        field = cls()
        field.loops = loops
        field.hmi_map = hmi_map
        field.streamlines = streamlines
        field.extrapolated_3d_field = field._transform_to_yt(map_3d, zrange)
        field._map_3d = map_3d
        field._zrange = zrange

        return field

    def _convert_angle_to_length(self, angle_or_length, working_units=u.meter):
        """
        Recast the `synthesizAR.util.convert_angle_to_length` to automatically use the supplied HMI map.
        """
        return convert_angle_to_length(self.hmi_map, angle_or_length, working_units=working_units)

    def _transform_to_yt(self, map_3d, zrange, boundary_clipping=(2, 2, 2)):
        """
        Reshape data structure to something yt can work with.

        Parameters
        ----------
        map_3d : `np.array`
            3D+x,y,z array holding the x,y,z components of the extrapolated field
        zrange : `astropy.Quantity`
            Spatial range of the extrapolated field
        boundary_clipping : `tuple`, optional
            The extrapolated volume has a layer of ghost cells in each dimension. This tuple of
            (nx,ny,nz) tells how many cells to contract the volume and map in each direction.
        """
        # reshape the magnetic field data
        _tmp = map_3d[boundary_clipping[0]:-boundary_clipping[0],
                      boundary_clipping[1]:-boundary_clipping[1],
                      boundary_clipping[2]:-boundary_clipping[2], :]
        # some annoying and cryptic translation between yt and SunPy
        data = dict(
                    Bx=(np.swapaxes(_tmp[:, :, :, 1], 0, 1), 'T'),
                    By=(np.swapaxes(_tmp[:, :, :, 0], 0, 1), 'T'),
                    Bz=(np.swapaxes(_tmp[:, :, :, 2], 0, 1), 'T'))
        # trim the boundary hmi map appropriately
        lcx, rcx = self.hmi_map.xrange + self.hmi_map.scale.axis1*u.Quantity([boundary_clipping[0], -boundary_clipping[0]], u.pixel)
        lcy, rcy = self.hmi_map.yrange + self.hmi_map.scale.axis2*u.Quantity([boundary_clipping[1], -boundary_clipping[1]], u.pixel)
        bottom_left = SkyCoord(lcx, lcy, frame=self.hmi_map.coordinate_frame)
        top_right = SkyCoord(rcx, rcy, frame=self.hmi_map.coordinate_frame)
        self.clipped_hmi_map = self.hmi_map.submap(bottom_left, top_right)
        # create the bounding box
        zscale = np.diff(zrange)[0]/(map_3d.shape[2]*u.pixel)
        clipped_zrange = zrange + zscale*u.Quantity([boundary_clipping[2]*u.pixel, -boundary_clipping[2]*u.pixel])
        bbox = np.array([self._convert_angle_to_length(self.clipped_hmi_map.xrange).value,
                         self._convert_angle_to_length(self.clipped_hmi_map.yrange).value,
                         self._convert_angle_to_length(clipped_zrange).value])
        # assemble the dataset
        return yt.load_uniform_grid(data, data['Bx'][0].shape, bbox=bbox, length_unit=yt.units.cm,
                                    geometry=('cartesian', ('x', 'y', 'z')))

    @u.quantity_input(loop_length_range=u.cm)
    def _filter_streamlines(self, streamline, close_threshold=0.05,
                            loop_length_range=[2.e+9, 5.e+10]*u.cm, **kwargs):
        """
        Check extracted loop to make sure it fits given criteria. Return True if it passes.

        Parameters
        ----------
        streamline : yt streamline object
        close_threshold : `float`
            percentage of domain width allowed between loop endpoints
        loop_length_range : `~astropy.Quantity`
            minimum and maximum allowed loop lengths (in centimeters)
        """
        streamline = streamline[np.all(streamline != 0.0, axis=1)]
        loop_length = np.sum(np.linalg.norm(np.diff(streamline, axis=0), axis=1))
        if np.fabs(streamline[0, 2] - streamline[-1, 2]) > close_threshold*self.extrapolated_3d_field.domain_width[2]:
            return False
        elif loop_length > loop_length_range[1].to(u.cm).value or loop_length < loop_length_range[0].to(u.cm).value:
            return False
        else:
            return True

    @u.quantity_input(zrange=u.arcsec)
    def extrapolate_field(self, zshape, zrange, use_numba_for_extrapolation=True):
        """
        Extrapolate the 3D field and transform it into a yt data object.
        """
        # extrapolate field
        self.logger.debug('Extrapolating field.')
        extrapolator = solarbextrapolation.extrapolators.PotentialExtrapolator(self.hmi_map, zshape=zshape, zrange=zrange)
        map_3d = extrapolator.extrapolate(enable_numba=use_numba_for_extrapolation)
        # preserve the 3d numpy array for restoration purposes
        self._map_3d = map_3d.data
        self._zrange = zrange
        # hand it to yt
        self.logger.debug('Transforming to yt data object')
        self.extrapolated_3d_field = self._transform_to_yt(map_3d.data, zrange)

    def extract_streamlines(self, number_fieldlines, max_tries=100, **kwargs):
        """
        Trace the fieldlines through extrapolated 3D volume
        """
        # trace field and return list of field lines
        self.logger.info('Tracing fieldlines')
        # wrap the streamline filter method so we can pass a loop length range to it
        streamline_filter_wrapper = functools.partial(self._filter_streamlines, **kwargs)
        self.streamlines = []
        seed_points = []
        i_tries = 0
        while len(self.streamlines) < number_fieldlines and i_tries < max_tries:
            remaining_fieldlines = number_fieldlines - len(self.streamlines)
            self.logger.debug('Remaining number of streamlines is {}'.format(remaining_fieldlines))
            # calculate seed points
            seed_points = find_seed_points(self.extrapolated_3d_field,
                                           self.clipped_hmi_map, remaining_fieldlines,
                                           preexisting_seeds=seed_points,
                                           mask_threshold=kwargs.get('mask_threshold', 0.1),
                                           safety=kwargs.get('safety', 2.))
            # trace fieldlines
            streamlines = yt.visualization.api.Streamlines(self.extrapolated_3d_field, 
                                                           (seed_points
                                                            * self.extrapolated_3d_field.domain_width
                                                            / self.extrapolated_3d_field.domain_width.value), 
                                                           xfield='Bx', yfield='By', zfield='Bz',
                                                           get_magnitude=True, direction=kwargs.get('direction',-1))
            streamlines.integrate_through_volume()
            streamlines.clean_streamlines()
            # filter
            keep_streamline = list(map(streamline_filter_wrapper, streamlines.streamlines))
            if True not in keep_streamline:
                i_tries += 1
                self.logger.debug('No acceptable streamlines found. # of tries left = {}'.format(max_tries-i_tries))
                continue
            else:
                i_tries = 0
            # save strealines
            self.streamlines += [(stream[np.all(stream != 0.0, axis=1)], mag) for stream, mag, keep 
                                 in zip(streamlines.streamlines, streamlines.magnitudes, keep_streamline) 
                                 if keep is True]

        if i_tries == max_tries:
            self.logger.warning('Maxed out number of tries. Only found {} acceptable streamlines'.format(len(self.streamlines)))

    def peek(self, figsize=(10, 10), color='b', alpha=0.75,
             print_to_file=None, **kwargs):
        """
        Show extracted fieldlines overlaid on HMI image.
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.gca(projection=self.hmi_map)
        self.hmi_map.plot()
        ax.set_autoscale_on(False)
        for stream, _ in self.streamlines:
            ax.plot(self._convert_angle_to_length(stream[:, 0]*u.cm,
                                                  working_units=u.arcsec).to(u.deg),
                    self._convert_angle_to_length(stream[:, 1]*u.cm,
                                                  working_units=u.arcsec).to(u.deg),
                    alpha=alpha, color=color, transform=ax.get_transform('world'))

        if print_to_file is not None:
            plt.savefig(print_to_file, **kwargs)
        plt.show()

    def make_loops(self):
        """
        Make list of `Loop` objects from the extracted streamlines
        """
        loops = []
        for i,stream in enumerate(self.streamlines):
            loops.append(Loop('loop{:06d}'.format(i), stream[0].value, stream[1].value))
        self.loops = loops

    def configure_loop_simulations(self, interface, **kwargs):
        """
        Configure hydrodynamic simulations for each loop object
        """
        self.simulation_type = interface.name
        with ProgressBar(len(self.loops), ipython_widget=kwargs.get('notebook', True)) as progress:
            for loop in self.loops:
                interface.configure_input(loop)
                progress.update()

    def load_loop_simulations(self, interface, savefile=None, **kwargs):
        """
        Load in loop parameters from hydrodynamic results.
        """
        with ProgressBar(len(self.loops), ipython_widget=kwargs.get('notebook', True)) as progress:
            for loop in self.loops:
                self.logger.debug('Loading parameters for {}'.format(loop.name))
                time, electron_temperature, ion_temperature, density, velocity = interface.load_results(loop, **kwargs)
                loop.time = time
                # convert velocity to cartesian coordinates
                grad_xyz = np.gradient(loop.coordinates.value, axis=0)
                s_hat = grad_xyz/np.expand_dims(np.linalg.norm(grad_xyz, axis=1), axis=-1)
                velocity_xyz = np.stack([velocity.value*s_hat[:, 0],
                                        velocity.value*s_hat[:, 1],
                                        velocity.value*s_hat[:, 2]], axis=2)*velocity.unit
                if savefile is not None:
                    loop.parameters_savefile = savefile
                    with h5py.File(savefile, 'a') as hf:
                        if loop.name not in hf:
                            hf.create_group(loop.name)
                        # electron temperature
                        dset_electron_temperature = hf[loop.name].create_dataset('electron_temperature',
                                                                                data=electron_temperature.value)
                        dset_electron_temperature.attrs['units'] = electron_temperature.unit.to_string()
                        # ion temperature
                        dset_ion_temperature = hf[loop.name].create_dataset('ion_temperature',
                                                                            data=ion_temperature.value)
                        dset_ion_temperature.attrs['units'] = ion_temperature.unit.to_string()
                        # number density
                        dset_density = hf[loop.name].create_dataset('density', data=density.value)
                        dset_density.attrs['units'] = density.unit.to_string()
                        # field-aligned velocity
                        dset_velocity = hf[loop.name].create_dataset('velocity', data=velocity.value)
                        dset_velocity.attrs['units'] = velocity.unit.to_string()
                        dset_velocity.attrs['note'] = 'Velocity in the field-aligned direction'
                        # Cartesian xyz velocity
                        dset_velocity_xyz = hf[loop.name].create_dataset('velocity_xyz',
                                                                        data=velocity_xyz.value)
                        dset_velocity_xyz.attrs['units'] = velocity_xyz.unit.to_string()
                        dset_velocity_xyz.attrs['note'] = '''Velocity in the Cartesian coordinate
                                                            system of the extrapolated magnetic
                                                            field. Index 0->x, index 1->y, index
                                                            2->z.'''
                else:
                    loop._electron_temperature = temperature
                    loop._ion_temperature = temperature
                    loop._density = density
                    loop._velocity = velocity
                    loop._velocity_xyz = velocity_xyz

                progress.update()

    def calculate_emission(self, emission_model, savefile=None, **kwargs):
        """
        Calculate emission (energy or photons per unit volume per unit time per unit solid angle)
        as function of time and space for each loop
        """
        for loop in self.loops:
            self.logger.info('Calculating emissivity for loop {}'.format(loop.name))
            loop.resolved_wavelengths = emission_model.resolved_wavelengths
            emiss, meta = emission_model.calculate_emission(loop, **kwargs)
            if savefile is not None:
                loop.emission_savefile = savefile
                with h5py.File(savefile, 'a') as hf:
                    if loop.name not in hf:
                        hf.create_group(loop.name)
                    for key in emiss:
                        self.logger.debug('Saving emission for {}'.format(key))
                        dset = hf[loop.name].create_dataset(key, data=emiss[key].value)
                        dset.attrs['units'] = emiss[key].unit.to_string()
                        for m in meta[key]:
                            dset.attrs[m] = meta[key][m]
            else:
                loop._emission = emiss

    def calculate_fractional_ionization(self, emission_model, interface=None, savefile=None, **kwargs):
        """
        Find the fractional ionization for each loop in the model as defined by the loop
        model interface.
        """
        ion_list = [(i.chianti_ion.meta['Element'], i.chianti_ion.meta['Ion']) for i in emission_model.ions]
        for loop in self.loops:
            if interface and hasattr(interface, 'get_fractional_ionization'):
                fractional_ionization = interface.get_fractional_ionization(ion_list, loop, **kwargs)
            else:
                self.logger.warning('''Model interface None or get_fractional_ionization method
                                    not defined. Falling back to ionization equilibrium.''')
                for ion in emission_model.ions:
                    f_ioneq = interp1d(emission_model.temperature_mesh[:, 0],
                                       ion.fractional_ionization[:, 0])
                    key = '{}_{}'.format(ion.chianti_ion.meta['Element'], ion.chianti_ion.meta['Ion'])
                    tmp = f_ioneq(loop.electron_temperature)
                    fractional_ionization[key] = np.where(tmp > 0.0, tmp, 0.0)

            if savefile is not None:
                loop.fractional_ionization_savefile = savefile
                with h5py.File(savefile, 'a') as hf:
                    if loop.name not in hf:
                        hf.create_group(loop.name)
                    for key in fractional_ionization:
                        self.logger.debug('Saving fractional ionization for {}'.format(key))
                        dset = hf[loop.name].create_dataset(key, data=fractional_ionization[key])
                        dset.attrs['description'] = '''Ion populations as a function of
                                                        temperature and density, either
                                                        interpolated from ionization equilibrium
                                                        data or calculated from the level
                                                        population equations.'''
                        dset.attrs['units'] = u.dimensionless_unscaled.to_string()
                        dset.attrs['element'] = key.split('_')[0]
                        dset.attrs['ion'] = key.split('_')[1]
            else:
                loop._fractional_ionization = fractional_ionization
