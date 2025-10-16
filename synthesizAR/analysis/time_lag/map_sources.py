"""
Map sources for diagnostic maps.
"""
from astropy.visualization import ImageNormalize
from sunpy.map import GenericMap

import synthesizAR.visualize.colormaps  # NOQA


class TimeLagMap(GenericMap):
    """
    A map that represents the time lag between two images.
    """
    def __init__(self, data, header, **kwargs):
        super().__init__(data, header, **kwargs)
        self.plot_settings['cmap'] = 'idl_bgry_004'
        self.plot_settings['norm'] = ImageNormalize(vmin=-7200, vmax=7200)
        self.nickname = f"{header.get('chan_a')}-{header.get('chan_b')}"

    @property
    def measurement(self):
        return ' '.join(self.meta.get('measrmnt', '').split('_')).capitalize()

    @classmethod
    def is_datasource_for(cls, data, header, **kwargs):
        return ('chan_a' in header and
                'chan_b' in header and
                header.get('measrmnt')=='time_lag')


class CrossCorrelationMap(GenericMap):
    """
    A map that represents the maximum cross-correlation value between two images
    """
    def __init__(self, data, header, **kwargs):
        super().__init__(data, header, **kwargs)
        self.plot_settings['cmap'] = 'magma'
        self.plot_settings['norm'] = ImageNormalize(vmin=0, vmax=1)
        self.nickname = f"{header.get('chan_a')}-{header.get('chan_b')}"

    @property
    def measurement(self):
        return ' '.join(self.meta.get('measrmnt', '').split('_')).capitalize()

    @classmethod
    def is_datasource_for(cls, data, header, **kwargs):
        return ('chan_a' in header and
                'chan_b' in header and
                header.get('measrmnt')=='max_cross_correlation')
