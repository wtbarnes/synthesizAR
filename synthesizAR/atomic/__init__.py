"""
Emissivity models and atomic calculations for simulating radiation from coronal loops
"""

from .chianti import ChIon
from .emission_models import EquilibriumEmissionModel
from .nei import get_ion_info,solve_nei_populations
