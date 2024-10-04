"""
synthesizAR is a package for forward modeling emission from solar active regions using
hydrodynamic simulations of coronal loops.
"""
try:
    from synthesizAR.version import __version__
except ImportError:
    __version__ = "unknown"


# Set up logger
from synthesizAR.util.logger import _init_log

from .skeleton import *
from .strand import *

log = _init_log()
