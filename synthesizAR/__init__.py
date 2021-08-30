"""
synthesizAR is a package for forward modeling emission from solar active regions using
hydrodynamic simulations of coronal loops.
"""
try:
    from .version import __version__
except ImportError:
    __version__ = "unknown"

try:
    # Alias counts as DN for convenience
    import astropy.units
    DN = astropy.units.def_unit('DN', astropy.units.count)
    astropy.units.add_enabled_units(DN)
except ValueError:
    # If a unit with the name 'DN' has already been registered,
    # a ValueError will be thrown, but we don't want our import
    # to explode over this.
    pass


from .loop import *
from .skeleton import *
