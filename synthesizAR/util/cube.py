"""
Object to deal with x,y,lambda data cubes
"""

import os

import numpy as np
import astropy.io.fits
import astropy.units as u
import sunpy.cm
from sunpy.map import Map,MapMeta
from sunpy.io.fits import get_header


class EISCube(object):
    """
    Spectral and spatial cube for holding Hinode EIS data
    """


    def __init__(self,*args,**kwargs):
        if len(args)==1 and os.path.exists(args[0]):
            data,header,wavelength = self._restore_from_fits(args[0])
        elif all([k in kwargs for k in ['data','header','wavelength']]):
            data = kwargs.get('data')
            header = kwargs.get('header')
            wavelength = kwargs.get('wavelength')
        else:
            raise ValueError('''EISCube can only be initialized with a valid FITS file or NumPy
                                array with an associated wavelength and header.''')
        self.meta = header.copy()
        self.wavelength = wavelength
        self.data = data
        self.cmap = kwargs.get('cmap',sunpy.cm.get_cmap('hinodexrt'))

    def __repr__(self):
        return '''synthesizAR {obj_name}
-----------------------------------------
Telescope : {tel}
Instrument : {instr}
Area : x={x_range}, y={y_range}
Dimension : {dim}
Scale : {scale}
Wavelength range : {wvl_range}
Wavelength dimension : {wvl_dim}
        '''.format(obj_name=type(self).__name__,tel=self.meta['telescop'],
                    instr=self.meta['instrume'],dim=u.Quantity(self[0].dimensions),
                    scale=u.Quantity(self[0].scale),wvl_dim=len(self.wavelength),
                    wvl_range=u.Quantity([self.wavelength[0],self.wavelength[-1]]),
                    x_range=self[0].xrange,y_range=self[0].yrange)

    def __getitem__(self,key):
        """
        Overriding indexing. If key is just one index, returns a normal `Map` object. Otherwise, another `EISCube` object is returned.
        """
        if type(self.wavelength[key].value)==np.ndarray:
            new_meta = self.meta.copy()
            new_meta['wavelnth'] = (self.wavelength[key][0].value+self.wavelength[key][1].value)/2.
            return EISCube(data=self.data[:,:,key],header=new_meta,wavelength=self.wavelength[key])
        else:
            meta_map2d = self.meta.copy()
            meta_map2d['naxis'] = 2
            for k in ['naxis3','ctype3','cunit3','cdelt3']:
                del meta_map2d[k]
            meta_map2d['wavelnth'] = self.wavelength[key].value
            tmp_map = Map(self.data[:,:,key],meta_map2d)
            tmp_map.plot_settings.update({'cmap':self.cmap})
            return tmp_map

    def save(self,filename):
        """
        Save to FITS file
        """
        #sanitize header
        header = self.meta.copy()
        if 'keycomments' in header:
            del header['keycomments']

        #create table file to hold wavelength array
        table_hdu = astropy.io.fits.BinTableHDU.from_columns(
            [astropy.io.fits.Column(name='wavelength',
                                    format='D',
                                    unit=self.wavelength.unit.to_string(),
                                    array=self.wavelength.value)])
        #create image file to hold 3D array
        image_hdu = astropy.io.fits.PrimaryHDU(np.swapaxes(self.data.T,1,2),
                                                header=astropy.io.fits.Header(header))
        #write to file
        hdulist = astropy.io.fits.HDUList([image_hdu,table_hdu])
        hdulist.writeto(filename,output_verify='silentfix')

    def _restore_from_fits(self,filename):
        """
        Helper to load cube from FITS file
        """
        tmp = astropy.io.fits.open(filename)
        data = tmp[0].data
        header = MapMeta(get_header(tmp)[0])
        wavelength = tmp[1].data.field(0)*u.Unit(tmp[1].header['TUNIT1'])
        tmp.close()

        return np.swapaxes(data.T,0,1),header,wavelength

    @property
    def integrated_intensity(self):
        """
        Map of the intensity integrated over wavelength.
        """
        tmp = np.dot(self.data,np.gradient(self.wavelength.value))
        tmp_meta = self[0].meta.copy()
        tmp_meta['wavelnth'] = self.meta['wavelnth']
        tmp_meta['bunit'] = (u.Unit(self.meta['bunit'])*self.wavelength.unit).to_string()
        tmp_map = Map(tmp,tmp_meta)
        tmp_map.plot_settings.update({'cmap':self.cmap})

        return tmp_map
