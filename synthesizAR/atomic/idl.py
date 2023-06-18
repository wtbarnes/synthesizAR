"""
Functions for computing isothermal spectra from CHIANTI IDL
"""
import contextlib
import io
import os

import asdf
import astropy.units as u
import ndcube
from ndcube.extra_coords import QuantityTableCoordinate
import numpy as np

__all__ = [
    'compute_spectral_table',
    'write_spectral_table',
    'spectrum_to_cube',
    'read_spectral_table',
]

_chianti_script = '''
ioneq_name = '{{ [ ssw_home, 'packages/chianti/dbase/ioneq', ioneq_file ] | join('/') }}'
abund_name = '{{ [ ssw_home, 'packages/chianti/dbase/abundance', abundance_file ] | join('/') }}'
wave_min = {{ wave_min | to_unit('Angstrom') | force_double_precision }}
wave_max = {{ wave_max | to_unit('Angstrom') | force_double_precision }}
wave_range = [wave_min, wave_max]
delta_wave = {{ delta_wave | to_unit('Angstrom') | force_double_precision }}
log_temperature = {{ temperature | to_unit('K') | log10 | force_double_precision }}
log_em = {{ emission_measure | to_unit('cm-5') | log10 | force_double_precision }}
density = {{ density | to_unit('cm-3') | force_double_precision }}

; define the wavelength grid here so that we can access it even if make_chianti_spec fails
; this is the same way the wavelength grid is defined in make_chianti_spec
; this will always be in Angstroms given the above unit conversion
wavelength = dindgen(ROUND((wave_range[1] - wave_range[0]) / delta_wave + 1 )) * delta_wave + wave_range[0]
in = where(wavelength GE wave_range[0] AND wavelength LE  wave_range[1], ng)
wavelength = wavelength[in]

;generate transition structure for selected wavelength and temperature range
ch_synthetic, wave_min,$
              wave_max,$
              output=transitions,$
              ioneq_name=ioneq_name,$
              logt_isothermal=log_temperature,$
              logem_isothermal=log_em,$
              {% if ion_list -%}sngl_ion=[{{ ion_list | string_list | join(',') }}],${%- endif %}
              {% if use_lookup_table %}/lookup,${%- endif%}
              density=density

;compute the spectra as a function of lambda and T
make_chianti_spec, transitions,$
                   wavelength,$
                   spectrum,$
                   abund_name=abund_name,$
                   bin_size=delta_wave,$
                   wrange=wave_range,$
                   {% if include_continuum -%}/continuum,${%- endif %}
                   {% if photons -%}/photons,${%- endif %}
                   {% if include_all_lines -%}/all,${%- endif %}
                   /no_thermal_width
'''


@u.quantity_input
def compute_spectral_table(temperature: u.K,
                           density: u.cm**(-3),
                           wave_min: u.angstrom,
                           wave_max: u.angstrom,
                           delta_wave: u.angstrom,
                           ioneq_filename,
                           abundance_filename,
                           photons=True,
                           emission_measure=1*u.Unit('cm-5'),
                           ion_list=None,
                           include_continuum=True,
                           use_lookup_table=False,
                           include_all_lines=True,
                           chianti_dir=None):
    """
    Compute spectra for a range of temperatures using CHIANTI IDL in SSW.

    Given a specified temperature and density profile, compute the spectra
    over the desired wavelength range at each temperature and density pair.
    This is computing by first calling the ``ch_synthetic`` and then
    ``make_chianti_spec`` in CHIANTI IDL as distributed in SolarSoft.

    Parameters
    ----------
    temperature: `~astropy.units.Quantity`
    density: `~astropy.units.Quantity`
        Must have same dimensions as `temperature`. Both `temperature`
        and `density` are assumed to vary along the same axis.
    wave_min: `~astropy.units.Quantity`
        Lower limit on wavelength range
    wave_max: `~astropy.units.Quantity`
        Upper limit on wavelength range
    delta_wave: `~astropy.units.Quantity`
        Spectral bin width
    ioneq_filename: `str`
        Name of ionization equilibrium file, including extension
    abundance_filename: `str`
        Name of abundance file, including extension
    emission_measure: `~astropy.units.Quantity`, optional
        Emission measure used when computing synthetic spectra. Note that
        this is divided out anyway.
    ion_list: `list`, optional
        List of ions to include in the synthetic spectra, e.g. 'fe_13' for
        Fe XIII. If None, all ions will be included.
    include_continuum: `bool`, optional
        If True, include free-free, free-bound, and two-photon continuum
        contributions to the spectra.
    include_all_lines: `bool`, optional
        If True, include "unobserved" lines (which are denoted by negative
        wavelengths in CHIANTI).
    chianti_dir: `str` or path-like, optional
        Path to the top level of CHIANTI installation, including directories
        for IDL code and database files. If not specified, the default CHIANTI
        available in the local SSW installation will be used.
    use_lookup_table: `bool`, optional
        If True, use the `/lookup` option for `ch_synthetic` to speed up the
        level populations calculation.

    Returns
    --------
    spectra: `~ndcube.NDCube`
        Resulting spectra as a function of temperature and wavelength
    """
    # Import here to avoid circular imports
    from synthesizAR import log
    # setup SSW environment and inputs
    input_args = {
        'wave_min': wave_min,
        'wave_max': wave_max,
        'delta_wave': delta_wave,
        'emission_measure': emission_measure,
        'ioneq_file': ioneq_filename,
        'abundance_file': abundance_filename,
        'ion_list': ion_list,
        'include_continuum': include_continuum,
        'include_all_lines': include_all_lines,
        'use_lookup_table': use_lookup_table,
        'photons': photons,
    }
    # NOTE: do not want this as a hard dependency, particularly if
    # just reading a spectral file
    import hissw
    if chianti_dir is not None:
        ssw_packages = None
        idl_root = os.path.join(chianti_dir, 'idl')
        dbase_root = os.path.join(chianti_dir, 'dbase')
        # Set up extra paths
        extra_paths = [d for d, _, _ in os.walk(idl_root)]
        header = f'''
        defsysv,'!xuvtop','{dbase_root}'
        defsysv,'!abund_file','{os.path.join(dbase_root, 'abundance', abundance_filename)}'
        defsysv,'!ioneq_file','{os.path.join(dbase_root, 'ioneq', ioneq_filename)}'
        '''
    else:
        # Use SSW installed CHIANTI
        ssw_packages = ['chianti']
        header = None
        extra_paths = None
    env = hissw.Environment(
        ssw_packages=ssw_packages,
        header=header,
        extra_paths=extra_paths,
    )

    # Iterate over T and n values
    all_spectra = []
    for T, n in zip(temperature, density):
        input_args['temperature'] = T
        input_args['density'] = n
        log.debug(f'Computing spectra for (T,n) = ({T}, {n})')
        wavelength, spec = _get_isothermal_spectra(env, input_args)
        all_spectra.append(spec)

    # Filter out any none entries
    spec_unit = f"{'ph' if photons else 'erg'} cm3 angstrom-1 s-1 sr-1"
    for i, spec in enumerate(all_spectra):
        if spec is None:
            all_spectra[i] = u.Quantity(np.zeros(wavelength.shape), spec_unit)

    # Build NDCube
    spectrum = u.Quantity(all_spectra)
    meta = {
        'version': get_chianti_version(env),
        'ioneq_filename': ioneq_filename,
        'abundance_filename': abundance_filename,
        'ion_list': 'all' if ion_list is None else ion_list,
        'include_continuum': include_continuum,
    }
    cube = spectrum_to_cube(spectrum,
                            wavelength,
                            temperature,
                            density=density,
                            meta=meta)

    return cube


def _get_isothermal_spectra(env, input_args):
    # Import here to avoid circular imports
    from synthesizAR import log
    log.debug(input_args)
    # NOTE: capturing the STDOUT here as when there are no relevant
    # transitions in the wavelength and temperature range, CHIANTI
    # returns nothing and so we have to treat this as an exception
    # to catch and return a placeholder
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        output = env.run(_chianti_script, args=input_args,
                         save_vars=['spectrum', 'wavelength'])
    idl_msg = f.getvalue()
    log.debug(idl_msg)
    wavelength = output['wavelength']
    wave_unit = u.Unit('Angstrom')
    wavelength = u.Quantity(wavelength, wave_unit)
    if 'No lines in the selected wavelength range !' in idl_msg:
        log.warning(idl_msg)
        return wavelength, None
    spectrum = output['spectrum']['spectrum'][0]
    units = output['spectrum']['units'][0]
    # The unit string from CHIANTI uses representations astropy
    # does not like so we fake those units
    u.add_enabled_units([
        u.def_unit('photons', represents=u.photon),
        u.def_unit('Angstroms', represents=u.Angstrom)
    ])
    spectrum_unit = u.Unit(units[1].decode('utf-8'))
    spectrum = u.Quantity(spectrum, spectrum_unit)
    # Originally, the spectrum was computed assuming unit EM.
    # Divide through to get the units right
    spectrum /= input_args['emission_measure']

    spec_unit = 'ph' if input_args['photons'] else 'erg'
    spec_unit = f'{spec_unit} cm3 angstrom-1 s-1 sr-1'

    return wavelength.to('Angstrom'), spectrum.to(spec_unit)


def get_chianti_version(env):
    output = env.run('version = ch_get_version()',
                     save_vars=['version'],
                     verbose=False)
    return output['version'].decode('utf-8')


def write_spectral_table(filename, cube):
    """
    Write result of `compute_spectral_table` to an ASDF file
    """
    tree = {}
    tree['spectrum'] = u.Quantity(cube.data, cube.unit)
    tree['temperature'] = cube.axis_world_coords(0)[0]
    if 'density' in cube.extra_coords.keys():
        # FIXME: there has to be a better way of accessing the data for the extra coord
        # FIXME: use gwcs instead?
        tree['density'] = cube.extra_coords['density']._lookup_tables[0][1].model.lookup_table
    tree['wavelength'] = cube.axis_world_coords(1)[0]
    for k, v in cube.meta.items():
        tree[k] = v
    with asdf.AsdfFile(tree) as asdf_file:
        asdf_file.write_to(filename)


def spectrum_to_cube(spectrum, wavelength, temperature, density=None, meta=None):
    """
    Build an NDCube of spectra as a function of wavelength and temperature

    Parameters
    ----------
    spectrum: `~astropy.units.Quantity`
        Spectra as a function of wavelength and temperature
    wavelength: `~astropy.units.Quantity`
        Wavelength array corresponding to the first axis of `spectrum`
    temperature: `~astropy.units.Quantity`
        Temperature array corresponding to the second axis of `spectrum`
    density: `~astropy.units.Quantity`, optional
        Density variation along the temperature axis. If specified, should
        be same dimensionality as `temperature`
    meta: `dict`, optional
        Metadata dictionary

    Returns
    -------
    : `~ndcube.NDCube`
    """
    # FIXME: This should properly account for the fact that these are actually wavelength
    # bins with finite width. Note quite sure how to do this with a gwcs yet, but Dan's
    # advice is to properly construct your gwcs so that it knows that certain coordinates
    # correspond to bind centers while others correspond to bin edges.
    gwcs = (
        QuantityTableCoordinate(wavelength, physical_types='em.wl') &
        QuantityTableCoordinate(temperature, physical_types='phys.temperature')
    ).wcs
    spec_cube = ndcube.NDCube(spectrum, wcs=gwcs, meta=meta)
    if density is not None:
        spec_cube.extra_coords.add('density', (0,), density, physical_types='phys.density')
    return spec_cube


def read_spectral_table(filename):
    """
    Read a spectral table file and return an NDCube

    Parameters
    ----------
    filename : `str` or path-like
        Path to ASDF file containing CHIANTI spectral table

    Returns
    -------
    : `~ndcube.NDCube`
    """
    # Read file
    with asdf.open(filename, mode='r', copy_arrays=True) as af:
        temperature = af['temperature']
        density = af['density']
        wavelength = af['wavelength']
        spectrum = af['spectrum']
        meta_keys = ['ioneq_filename',
                     'abundance_filename',
                     'ion_list',
                     'version',
                     'include_continuum']
        meta = {k: af[k] if k in af else None for k in meta_keys}
    return spectrum_to_cube(spectrum, wavelength, temperature, density=density, meta=meta)
