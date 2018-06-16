"""
Active region object definition. This object holds all the important information about our
synthesized active region.
"""
import os
import datetime

import numpy as np
import sunpy.map
from sunpy.coordinates import HeliographicStonyhurst
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.utils.console import ProgressBar
import h5py

from synthesizAR import Loop
from synthesizAR.extrapolate import peek_fieldlines
from synthesizAR.util import get_keys


class Field(object):
    """
    Construct magnetic field skeleton from magnetogram and fieldlines

    Parameters
    ----------
    magnetogram : `sunpy.map.Map`
        Magnetogram map for the active region
    fieldlines : `list`
        List of tuples, coordinates and field strengths for each loop

    Examples
    --------
    >>> import synthesizAR
    >>> from sunpy.map import Map
    >>> import astropy.units as u
    >>> m = Map('/path/to/sample_hmi.fits') # doctest: +SKIP
    >>> fieldlines = [(SkyCoord(x=[1,4]*u.Mm, y=[2,5]*u.Mm, z=[3,6]*u.Mm,frame='heliographic_stonyhurst', representation='cartesian'), [1e2,1e3] * u.G)]
    >>> field = synthesizAR.Field(m, fieldlines) # doctest: +SKIP
    """

    def __init__(self, magnetogram, fieldlines):
        self.magnetogram = sunpy.map.Map(magnetogram)
        self.loops = self._make_loops(fieldlines)

    def _make_loops(self, fieldlines):
        """
        Make list of `Loop` objects from the extracted streamlines
        """
        loops = []
        for i, (line, mag) in enumerate(fieldlines):
            loops.append(Loop(f'loop{i:06d}', line, mag))
        return loops

    def __repr__(self):
        sim_type = self.simulation_type if hasattr(self, 'simulation_type') else ''
        return f'''synthesizAR Active Region Object
------------------------
Number of loops: {len(self.loops)}
Simulation Type: {sim_type}
Magnetogram Info:
-----------------
{self.magnetogram.__repr__()}'''

    def save(self, savedir=None):
        """
        Save the components of the field object to be reloaded later.
        """
        if savedir is None:
            dt = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            savedir = f'synthesizAR-{type(self).__name__}-save_{dt}'
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        if not os.path.isfile(os.path.join(savedir, 'magnetogram.fits')):
            self.magnetogram.save(os.path.join(savedir, 'magnetogram.fits'))
        with h5py.File(os.path.join(savedir, 'loops.h5'), 'w') as hf:
            for i, loop in enumerate(self.loops):
                grp = hf.create_group(loop.name)
                grp.attrs['index'] = i
                if hasattr(loop, 'parameters_savefile'):
                    grp.attrs['parameters_savefile'] = loop.parameters_savefile
                else:
                    grp.attrs['parameters_savefile'] = ''
                ds = grp.create_dataset('coordinates', data=loop.coordinates.cartesian.xyz.value)
                ds.attrs['unit'] = loop.coordinates.cartesian.xyz.unit.to_string()
                ds = grp.create_dataset('field_strength', data=loop.field_strength.value)
                ds.attrs['unit'] = loop.field_strength.unit.to_string()

    @classmethod
    def restore(cls, savedir):
        """
        Restore the field from a set of serialized files

        Examples
        --------
        >>> import synthesizAR
        >>> restored_field = synthesizAR.Field.restore('/path/to/restored/field/dir') # doctest: +SKIP
        """
        fieldlines = []
        with h5py.File(os.path.join(savedir, 'loops.h5'), 'r') as hf:
            for grp_name in hf:
                grp = hf[grp_name]
                x = u.Quantity(grp['coordinates'][0, :], 
                               get_keys(grp['coordinates'].attrs, ('unit', 'units')))
                y = u.Quantity(grp['coordinates'][1, :],
                               get_keys(grp['coordinates'].attrs, ('unit', 'units')))
                z = u.Quantity(grp['coordinates'][2, :],
                               get_keys(grp['coordinates'].attrs, ('unit', 'units')))
                coordinates = SkyCoord(x=x, y=y, z=z, frame=HeliographicStonyhurst,
                                       representation='cartesian')
                field_strength = u.Quantity(
                    grp['field_strength'], get_keys(grp['field_strength'].attrs, ('unit', 'units')))
                fieldlines.append({'index': grp.attrs['index'],
                                   'parameters_savefile': grp.attrs['parameters_savefile'],
                                   'coordinates': coordinates, 'field_strength': field_strength})

        fieldlines = sorted(fieldlines, key=lambda x: x['index'])
        magnetogram = sunpy.map.Map(os.path.join(savedir, 'magnetogram.fits'))
        field = cls(magnetogram, [(f['coordinates'], f['field_strength']) for f in fieldlines])
        for f in fieldlines:
            if f['parameters_savefile']:
                field.loops[f['index']].parameters_savefile = f['parameters_savefile']

        return field

    def peek(self, **kwargs):
        """
        Show extracted fieldlines overlaid on magnetogram.
        """
        fieldlines = [loop.coordinates for loop in self.loops]
        peek_fieldlines(self.magnetogram, fieldlines, **kwargs)

    def configure_loop_simulations(self, interface, **kwargs):
        """
        Configure hydrodynamic simulations for each loop object
        """
        self.simulation_type = interface.name
        with ProgressBar(len(self.loops), ipython_widget=kwargs.get('notebook', True)) as progress:
            for loop in self.loops:
                interface.configure_input(loop)
                progress.update()

    def load_loop_simulations(self, interface, savefile, **kwargs):
        """
        Load in loop parameters from hydrodynamic results.
        """
        notebook = kwargs.get('notebook', True)
        with h5py.File(savefile, 'w') as hf:
            with ProgressBar(len(self.loops), ipython_widget=notebook) as progress:
                for loop in self.loops:
                    # Load in parameters from interface
                    (time, electron_temperature, ion_temperature,
                     density, velocity) = interface.load_results(loop, **kwargs)
                    # convert velocity to loop coordinate system
                    grad_xyz = np.gradient(loop.coordinates.cartesian.xyz.value, axis=1)
                    s_hat = grad_xyz / np.linalg.norm(grad_xyz, axis=0)
                    velocity_x = velocity * s_hat[0, :]
                    velocity_y = velocity * s_hat[1, :]
                    velocity_z = velocity * s_hat[2, :]
                    # Write to file
                    loop.parameters_savefile = savefile
                    grp = hf.create_group(loop.name)
                    # time
                    dset_time = grp.create_dataset('time', data=time.value)
                    dset_time.attrs['unit'] = time.unit.to_string()
                    # electron temperature
                    dset_electron_temperature = grp.create_dataset('electron_temperature',
                                                                   data=electron_temperature.value)
                    dset_electron_temperature.attrs['unit'] = electron_temperature.unit.to_string()
                    # ion temperature
                    dset_ion_temperature = grp.create_dataset('ion_temperature',
                                                              data=ion_temperature.value)
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
