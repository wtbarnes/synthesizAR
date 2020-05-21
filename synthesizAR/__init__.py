# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
synthesizAR is a package for forward modeling emission from solar active regions using
hydrodynamic simulations of coronal loops.
"""
from ._astropy_init import *
try:
    from .version import __version__
except ImportError:
    __version__ = "unknown"

from .loop import Loop
from .skeleton import Skeleton
# Alias counts as DN for convenience
import astropy.units
DN = astropy.units.def_unit('DN', astropy.units.count)
astropy.units.add_enabled_units(DN)
