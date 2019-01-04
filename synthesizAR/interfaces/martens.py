"""
Interface between loop object and scaling law calculations by Martens
"""

__all__ = ['MartensInterface']


class MartensInterface(object):

    def __init__(self,):
        ...

    def load_results(self, loop):
        ...
