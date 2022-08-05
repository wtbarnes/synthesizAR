"""
Functions for computing isothermal spectra from CHIANTI IDL
"""
import contextlib
import io

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
wave_min = {{ wave_min | to_unit('Angstrom') }}
wave_max = {{ wave_max | to_unit('Angstrom') }}
wave_range = [wave_min, wave_max]
delta_wave = {{ delta_wave | to_unit('Angstrom') }}
log_temperature = {{ temperature | to_unit('K') | log10 }}
log_em = {{ emission_measure | to_unit('cm-5') | log10 }}
density = {{ density | to_unit('cm-3') }}

;generate transition structure for selected wavelength and temperature range
ch_synthetic, wave_min,$
              wave_max,$
              output=transitions,$
              ioneq_name=ioneq_name,$
              logt_isothermal=log_temperature,$
              logem_isothermal=log_em,$
              {% if ion_list -%}sngl_ion=[{{ ion_list | join(',') }}],${%- endif %}
              density=density

;compute the spectra as a function of lambda and T
make_chianti_spec, transitions,$
                   wavelength,$
                   spectrum,$
                   bin_size=delta_wave,$
                   wrange=wave_range,$
                   abund_name=abund_name,$
                   /continuum,$
                   /photons
'''


@u.quantity_input
def compute_spectral_table(temperature: u.K,
                           density: u.cm**(-3),
                           wave_min: u.angstrom,
                           wave_max: u.angstrom,
                           delta_wave: u.angstrom,
                           ioneq_filename,
                           abundance_filename,
                           emission_measure=1*u.Unit('cm-5'),
                           ion_list=None):
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

    Returns
    --------
    wavelength: `~astropy.units.Quantity`
        Wavelength axis of the spectra
    spectra: `~astropy.units.Quantity`
        Resulting spectra
    """
    # setup SSW environment and inputs
    input_args = {
        'wave_min': wave_min,
        'wave_max': wave_max,
        'delta_wave': delta_wave,
        'emission_measure': emission_measure,
        'ioneq_file': ioneq_filename,
        'abundance_file': abundance_filename,
        'ion_list': ion_list,
    }
    # NOTE: do not want this as a hard dependency, particularly if
    # just reading a spectral file
    import hissw
    env = hissw.Environment(ssw_packages=['chianti'])

    # Iterate over T and n values
    all_spectra = []
    for T, n in zip(temperature, density):
        input_args['temperature'] = T
        input_args['density'] = n
        _wavelength, spec = _get_isothermal_spectra(env, input_args)
        all_spectra.append(spec)
        # This ensures that wavelength is never None unless
        # everything is None
        if _wavelength is not None:
            wavelength = _wavelength

    # Filter out any none entries
    for i, spec in enumerate(all_spectra):
        if spec is None:
            all_spectra[i] = u.Quantity(np.zeros(wavelength.shape),
                                        'cm3 ph Angstrom-1 s-1 sr-1')

    return wavelength, u.Quantity(all_spectra)


def _get_isothermal_spectra(env, input_args):
    # NOTE: capturing the STDOUT here as when there are no relevant
    # transitions in the wavelength and temperature range, CHIANTI
    # returns nothing and so we have to treat this as an exception
    # to catch and return a placeholder
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        output = env.run(_chianti_script, args=input_args,
                         save_vars=['spectrum'])
    idl_msg = f.getvalue()
    if 'No lines in the selected wavelength range !' in idl_msg:
        return None, None
    spectrum = output['spectrum']['spectrum'][0]
    wavelength = output['spectrum']['lambda'][0]
    units = output['spectrum']['units'][0]
    # The unit string from CHIANTI uses representations astropy
    # does not like so we fake those units
    u.add_enabled_units([
        u.def_unit('photons', represents=u.photon),
        u.def_unit('Angstroms', represents=u.Angstrom)
    ])
    wave_unit = u.Unit(units[0].decode('utf-8'))
    wavelength = u.Quantity(wavelength, wave_unit)
    spectrum_unit = u.Unit(units[1].decode('utf-8'))
    spectrum = u.Quantity(spectrum, spectrum_unit)
    # Originally, the spectrum was computed assuming unit EM.
    # Divide through to get the units right
    spectrum = spectrum / input_args['emission_measure']

    return wavelength.to('Angstrom'), spectrum.to('cm3 ph Angstrom-1 s-1 sr-1')


def write_spectral_table(filename,
                         spectrum,
                         temperature,
                         density,
                         wavelength,
                         ioneq_filename,
                         abundance_filename,
                         ion_list=None):
    """
    Write result of `compute_spectral_table` to an ASDF file
    """
    tree = {}
    tree['temperature'] = temperature
    tree['density'] = density
    tree['wavelength'] = wavelength
    tree['ioneq_filename'] = ioneq_filename
    tree['abundance_filename'] = abundance_filename
    tree['ion_list'] = 'all' if ion_list is None else ion_list
    tree['spectrum'] = spectrum
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
    gwcs = (
        QuantityTableCoordinate(wavelength, physical_types='em.wl') &
        QuantityTableCoordinate(temperature, physical_types='phys.temperature')
    ).wcs
    spec_cube = ndcube.NDCube(spectrum, wcs=gwcs, meta=meta)
    if density:
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
        ioneq_filename = af['ioneq_filename']
        abundance_filename = af['abundance_filename']
        ion_list = af['ion_list']
        spectrum = af['spectrum']
    meta = {
        'ioneq_filename': ioneq_filename,
        'abundance_filename': abundance_filename,
        'ion_list': ion_list,
    }
    return spectrum_to_cube(spectrum, wavelength, temperature, density=density, meta=meta)
