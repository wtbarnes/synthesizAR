"""
Interface to the 0D EBTEL model
"""
from .ebtel import EbtelInterface
from .heating_models import UniformHeating, PowerLawScaledWaitingTimes, PowerLawUnscaledWaitingTimes
from .heating_models import calculate_free_energy, power_law_transform
