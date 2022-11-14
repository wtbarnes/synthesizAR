"""
synthesizAR is a package for forward modeling emission from solar active regions using
hydrodynamic simulations of coronal loops.
"""
try:
    from .version import __version__
except ImportError:
    __version__ = "unknown"


from .loop import *
from .skeleton import *

# Set up logger
from synthesizAR.util.logger import _init_log
log = _init_log()
