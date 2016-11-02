# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
synthesizAR is a package for forward modeling emission from solar active regions using hydrodynamic simulations of coronal loops.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from . import version
    Version = version._last_generated_version
    from .field import Skeleton
    from .observe import Observer
