"""
Class for Hinode/EIS instrument. Holds information about spectral, temporal, and spatial resolution
and other instrument-specific information.
"""

import os
import sys
import logging

import numpy as np
import sunpy.map
import astropy.units as u

from synthesizAR.instruments import InstrumentBase,Pair


class InstrumentHinodeEIS(InstrumentBase):
    """
    Class for Extreme-ultraviolet Imaging Spectrometer (EIS) instrument on the Hinode spacecraft.
    Converts emissivity calculations for each loop into detector units based on the spectral,
    spatial, and temporal resolution along with the instrument response functions.
    """


    def __init__(self):
        pass

    def detect(self):
        pass
