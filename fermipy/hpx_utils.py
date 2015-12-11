
try:
    import astropy.io.fits as pf
except:
    import pyfits as pf

import healpy as hp
import numpy as np
import re


def coords_to_vec(lon,lat):
    """ Converts longitute and latitude coordinates to a unit 3-vector """
    phi = np.radians(lon)
    theta = (np.pi/2) - np.radians(lat)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    vec = np.ndarray((3),'f')    
    vec[0] = sin_t * np.cos(phi)
    vec[1] = sin_t * np.sin(phi)
    vec[2] = cos_t
    return vec


class HPX(object):
    """ Encapsulation of basic healpix map parameters """

    def __init__(self,nside,nest,coordsys,order=-1,region=None,ebins=None):
        """
        """
        if nside >= 0:
            if order >= 0:
                raise Exception('Specify either nside or oder, not both.')
            else:
                self._nside = nside
                self._order = -1
        else:
            if order >= 0:
                self._nside = 2**order
                self._order = order
            else:
                raise Exception('Specify either nside or oder, not both.')
        self._nest = nest
        self._coordsys = coordsys
        self._region = region
        self._maxpix = 12*self._nside*self._nside
        if self._region:
            self._ipix = self.get_index_list(self._region)
            self._npix = len(self._ipix)
            
        else:
            self._ipix = None
            self._npix = self._maxpix

        self._ebins = ebins
        if self._ebins is not None:
            self._evals = np.sqrt(self._ebins[0:-1]*self._ebins[1:])
        else:
            self._evals = None
            
    @property
    def ordering(self):
        if self._nest: 
            return "NESTED"
        return "RING"

    @property
    def npix(self):
        return self._npix

    @property
    def ebins(self):
        return self._ebins

    @property
    def evals(self):
        return self._evals    

    @property
    def region(self):
        return self._region

    @property
    def ipix(self):
        return self._ipix    

    def make_header(self):
        """ Build a fits header for this healpix map """
        cards = [pf.Card("TELESCOP","GLAST"),
                 pf.Card("INSTRUME", "LAT"),
                 pf.Card("COORDSYS",self._coordsys),                 
                 pf.Card("PIXTYPE","HEALPIX"),
                 pf.Card("ORDERING",self.ordering),
                 pf.Card("ORDER",self._order),
                 pf.Card("NSIDE",self._nside),
                 pf.Card("FIRSTPIX",0),
                 pf.Card("LASTPIX",self._maxpix-1)]
        if self._coordsys=="CEL":
            cards.append(pf.Card("EQUINOX", 2000.0,"Equinox of RA & DEC specifications"))
            
        if self._region:
            cards.append(pf.Card("HPXREGION", self._region))
            
        header = pf.Header(cards)
        return header


    def make_hdu(self,data,extname="SKYMAP"):
        """
        """
        shape = data.shape
        if shape[-1] != self._npix:
            raise Exception("Size of data array does not match number of pixels")
        cols = []
        if self._region:
            cols.append(pf.Column("PIX","J",array=self._ipix))
        if len(shape) == 1:
            cols.append(pf.Column("CHANNEL1","D",array=data))
        elif len(shape) == 2:
            for i in range(shape[0]):
                cols.append(pf.Column("CHANNEL%i"%i,"D",array=data[i]))
                pass
        else:
            raise Exception("HPX.write_fits only handles 1D and 2D maps")
        
        hdu = pf.BinTableHDU.from_columns(cols,self.make_header(),name=extname)
        return hdu

    
    def write_fits(self,data,outfile,extname="SKYMAP",clobber=True):
        """
        """
        hdu_prim = pf.PrimaryHDU()
        hdu_hpx = self.make_hdu(data,extname)
        hdulist = pf.HDUList([hdu_prim,hdu_hpx])
        hdulist.writeto(outfile,clobber=clobber)
        

    def get_index_list(self,region):
        """
        """
        tokens = re.split('\(|\)|,',region)
        if tokens[0] == 'DISK':
            vec = coords_to_vec(float(tokens[1]),float(tokens[2]))
            ilist = hp.query_disc(self._nside,vec,np.radians(float(tokens[3])),
                                  inclusive=False,nest=self._nest)
            pass
        elif tokens[0] == 'DISK_INC':
            vec = coords_to_vec(float(tokens[1]),float(tokens[2]))
            ilist = hp.query_disc(self._nside,vec,np.radians(float(tokens[3])),
                                  inclusive=True,fact=int(tokens[4]),
                                  nest=self._nest)
            pass
        else:
            raise Exception("HPX.get_index_list did not recognize region type %s"%tokens[0])
        return ilist




if __name__ == "__main__":
    
    import numpy as np
    import fermipy.hpx_utils           
    n = np.ones((10,192),'d')
    hpx = fermipy.hpx_utils.HPX(4,False,"GAL")
    hpx.write_fits(n,"test.fits",clobber=True)


