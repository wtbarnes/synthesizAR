"""
Interface between loop object and scaling law calculations by Martens
"""


class MartensInterface(object):

    def __init__(self,):
        ...

    def load_results(self, loop):
        return time, electron_temperature, ion_temperature, density, velocity
