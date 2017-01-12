"""
Object to deal with x,y,lambda data cubes
"""

import copy

import numpy as np
import astropy.io.fits
from sunpy.map import Map,MapCube,MapMeta
from sunpy.io.fits import get_header


class EISCube(MapCube):
    """
    Spectral and spatial cube for holding Hinode EIS data
    """


    def __init__(self,data,header,wavelength):
        self.meta = header.copy()
        self.wavelength = wavelength
        # construct individual maps
        meta_map2d = header.copy()
        for k in ['naxis3','ctype3','cunit3','cdelt3']:
            del meta_map2d[k]
        map_list = []
        for i,wvl in enumerate(self.wavelength):
            meta_map2d['wavelnth'] = wvl.value
            map_list.append(Map(data[:,:,i],meta_map2d.copy()))
        super().__init__(map_list)

    def save(self,filename):
        """
        Save to FITS file
        """
        #build header (steal some code here from SunPy)
        header = self.meta.copy()
        # The comments need to be added to the header separately from the normal
        # kwargs. Find and deal with them:
        fits_header = astropy.io.fits.Header()
        key_comments = header.pop('KEYCOMMENTS', False)

        for k,v in header.items():
            if isinstance(v, astropy.io.fits.header._HeaderCommentaryCards):
                if k == 'comments':
                    comments = str(v).split('\n')
                    for com in comments:
                        fits_header.add_comments(com)
                elif k == 'history':
                    hists = str(v).split('\n')
                    for hist in hists:
                        fits_header.add_history(hist)
                elif k != '':
                    fits_header.append(astropy.io.fits.Card(k, str(v).split('\n')))

            else:
                fits_header.append(astropy.io.fits.Card(k, v))

        if isinstance(key_comments, dict):
            for k,v in key_comments.items():
                fits_header.comments[k] = v
        elif key_comments:
            raise TypeError("KEYCOMMENTS must be a dictionary")

        #create table file to hold wavelength array
        table_hdu = astropy.io.fits.BinTableHDU.from_columns(
            [astropy.io.fits.Column(name='wavelength',
                                    format='D',
                                    unit=self.wavelength.unit.to_string(),
                                    array=self.wavelength.value)])
        #create image file to hold 3D array
        image_hdu = astropy.io.fits.PrimaryHDU(self.as_array())
        #write to file
        hdulist = astropy.io.fits.HDUList([image_hdu,table_hdu])
        hdulist.writeto(filename)

    @classmethod
    def restore(self,filename):
        """
        Restore from FITS file
        """
        pass
