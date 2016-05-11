from __future__ import absolute_import, division, print_function, \
    unicode_literals

import copy
import numpy as np

import astropy.io.fits as pyfits
import astropy.wcs as pywcs

import fermipy.utils as utils
import fermipy.wcs_utils as wcs_utils
from fermipy.utils import Map, read_energy_bounds
from fermipy.hpx_utils import HpxMap, HPX


def write_maps(primary_map, maps, outfile):
    
    hdu_images = [primary_map.create_primary_hdu()]
    for k, v in sorted(maps.items()):
        hdu_images += [v.create_image_hdu(k)]

    hdulist = pyfits.HDUList(hdu_images)
    hdulist.writeto(outfile, clobber=True)

    
def read_map_from_fits(fitsfile, extname=None):
    """
    """
    proj, f, hdu = read_projection_from_fits(fitsfile, extname)
    if isinstance(proj, pywcs.WCS):
        m = Map(hdu.data, proj)
    elif isinstance(proj, HPX):
        m = HpxMap.create_from_hdu(hdu,proj.ebins)
    else:
        raise Exception("Did not recognize projection type %s" % type(proj))
    return m,f


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


def make_coadd_map(maps, proj, shape):
    # this is a hack
    from fermipy.hpx_utils import make_coadd_hpx, HPX
    if isinstance(proj, pywcs.WCS):
        return make_coadd_wcs(maps, proj, shape)
    elif isinstance(proj, HPX):
        return make_coadd_hpx(maps, proj, shape)
    else:
        raise Exception("Can't co-add map of unknown type %s" % type(proj))


def make_coadd_wcs(maps, wcs, shape):
    data = np.zeros(shape)
    axes = wcs_utils.wcs_to_axes(wcs, shape)

    for m in maps:
        c = wcs_utils.wcs_to_coords(m.wcs, m.counts.shape)
        o = np.histogramdd(c.T, bins=axes[::-1], weights=np.ravel(m.counts))[0]
        data += o

    return utils.Map(data, copy.deepcopy(wcs))
