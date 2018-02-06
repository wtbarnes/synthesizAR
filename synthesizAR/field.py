"""
Active region object definition. This object holds all the important information about our
synthesized active region.
"""
import os
import warnings
import datetime
import pickle
import glob

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sunpy.map
import astropy.units as u
from astropy.utils.console import ProgressBar
import h5py
import yt

from synthesizAR import Loop


class Field(object):
    """
    Construct magnetic field skeleton from magnetogram and fieldlines

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

    def __init__(self, magnetogram, streamlines=None):
        self.magnetogram = sunpy.map.Map(magnetogram)
        if streamlines is not None:
            self.loops = self.make_loops(streamlines)
        else:
            warnings.warn('Streamlines not found. No loops will be created.')

    def __repr__(self):
        num_loops = len(self.loops) if hasattr(self, 'loops') else 0
        sim_type = self.simulation_type if hasattr(self, 'simulation_type') else ''
        return f'''synthesizAR Field Object
------------------------
Number of loops: {num_loops}
Simulation Type: {sim_type}
Magnetogram Info:
-----------------
{self.magnetogram.__repr__()}'''

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
        if not os.path.exists(os.path.join(savedir, 'loops')):
            os.makedirs(os.path.join(savedir, 'loops'))
        for l in self.loops:
            with open(os.path.join(savedir, 'loops', l.name+'.pickle'), 'wb') as f:
                pickle.dump(l, f)
        if not os.path.isfile(os.path.join(savedir, 'magnetogram.fits')):
            self.magnetogram.save(os.path.join(savedir, 'magnetogram.fits'))

    @classmethod
    def restore(cls, savedir):
        """
        Restore the field from a set of serialized files

        Examples
        --------
        >>> import synthesizAR
        >>> restored_field = synthesizAR.Field.restore('/path/to/restored/field/dir')
        """
        # loops
        loop_files = glob.glob(os.path.join(savedir, 'loops', '*'))
        loop_files = sorted([lf.split('/')[-1] for lf in loop_files],
                            key=lambda l: int(l.split('.')[0][4:]))
        loops = []
        for lf in loop_files:
            with open(os.path.join(savedir, 'loops', lf), 'rb') as f:
                loops.append(pickle.load(f))
        magnetogram = sunpy.map.Map(os.path.join(savedir, 'magnetogram.fits'))
        field = cls(magnetogram)
        field.loops = loops

        return field

    def peek(self, figsize=(10, 10), color='b', alpha=0.75, **kwargs):
        """
        Show extracted fieldlines overlaid on HMI image.
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.gca(projection=self.magnetogram)
        self.magnetogram.plot()
        ax.set_autoscale_on(False)
        for stream, _ in self.streamlines:
            ax.plot(self._convert_angle_to_length(stream[:, 0]*u.cm,
                                                  working_units=u.arcsec).to(u.deg),
                    self._convert_angle_to_length(stream[:, 1]*u.cm,
                                                  working_units=u.arcsec).to(u.deg),
                    alpha=alpha, color=color, transform=ax.get_transform('world'))

        plt.show()

    def make_loops(self, streamlines):
        """
        Make list of `Loop` objects from the extracted streamlines
        """
        loops = []
        for i, stream in enumerate(streamlines):
            loops.append(Loop('loop{:06d}'.format(i), stream[0].value, stream[1].value))        
        return loops

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
