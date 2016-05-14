from __future__ import absolute_import, division, print_function, \
    unicode_literals

import copy
import numpy as np

import astropy.io.fits as pyfits
import astropy.wcs as pywcs

import fermipy
import fermipy.utils as utils
import fermipy.wcs_utils as wcs_utils
from fermipy.hpx_utils import HPX


def read_energy_bounds(hdu):
    """ Reads and returns the energy bin edges from a FITs HDU
    """
    nebins = len(hdu.data)
    ebin_edges = np.ndarray((nebins+1))
    ebin_edges[0:-1] = np.log10(hdu.data.field("E_MIN")) - 3.
    ebin_edges[-1] = np.log10(hdu.data.field("E_MAX")[-1]) - 3.
    return ebin_edges


def read_spectral_data(hdu):
    """ Reads and returns the energy bin edges, fluxes and npreds from
    a FITs HDU
    """
    ebins = read_energy_bounds(hdu)
    fluxes = np.ndarray((len(ebins)))
    try:
        fluxes[0:-1] = hdu.data.field("E_MIN_FL")
        fluxes[-1] = hdu.data.field("E_MAX_FL")[-1]
        npreds = hdu.data.field("NPRED")
    except:
        fluxes =  np.ones((len(ebins)))
        npreds =  np.ones((len(ebins)))
    return ebins,fluxes,npreds


def write_maps(primary_map, maps, outfile):
    
    hdu_images = [primary_map.create_primary_hdu()]
    for k, v in sorted(maps.items()):
        hdu_images += [v.create_image_hdu(k)]

    hdulist = pyfits.HDUList(hdu_images)
    for h in hdulist:    
        h.header['CREATOR'] = 'fermipy ' + fermipy.__version__
    hdulist.writeto(outfile, clobber=True)

    
def read_projection_from_fits(fitsfile, extname=None):
    """
    Load a WCS or HPX projection.
    """
    f = pyfits.open(fitsfile)
    nhdu = len(f)
    # Try and get the energy bounds
    try:
        ebins = read_energy_bounds(f['EBOUNDS'])
    except:
        ebins = None
    
    if extname is None:
        # If there is an image in the Primary HDU we can return a WCS-based projection
        if f[0].header['NAXIS'] != 0:
            proj = pywcs.WCS(f[0].header)
            return proj,f,f[0]
    else:
        if f[extname].header['XTENSION'] == 'IMAGE':
            proj = pywcs.WCS(f[extname].header)
            return proj,f,f[extname]
        elif f[extname].header['XTENSION'] == 'BINTABLE':
            try: 
                if f[extname].header['PIXTYPE'] == 'HEALPIX':
                    proj = HPX.create_from_header(f[extname].header,ebins)
                    return proj,f,f[extname]
            except:
                pass
        return None,f,None 
            
    # Loop on HDU and look for either an image or a table with HEALPix data
    for i in range(1,nhdu):
        # if there is an image we can return a WCS-based projection
        if f[i].header['XTENSION'] == 'IMAGE':
            proj = pywcs.WCS(f[i].header)
            return proj,f,f[i]
        elif f[i].header['XTENSION'] == 'BINTABLE':
            try: 
                if f[i].header['PIXTYPE'] == 'HEALPIX':
                    proj = HPX.create_from_header(f[i].header,ebins)
                    return proj,f,f[i]
            except:
                pass
        pass
    return None,f,None



