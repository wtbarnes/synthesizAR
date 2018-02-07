"""
Extensions for configuring input files for and loading results from hydrodynamic models
"""

from .ebtel import EbtelInterface
from .heating_models import UniformHeating, PowerLawScaledWaitingTimes, PowerLawUnscaledWaitingTimes
from .heating_models import calculate_free_energy, power_law_transform
