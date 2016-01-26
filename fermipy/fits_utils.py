import copy
import numpy as np

import astropy.io.fits as pyfits
import astropy.wcs as pywcs

from fermipy.utils import Map, read_energy_bounds
from fermipy.hpx_utils import HpxMap, HPX


def read_map_from_fits(fitsfile, extname=None):
    """
    """
    proj, f, hdu = read_projection_from_fits(fitsfile, extname)
    if isinstance(proj, pywcs.WCS):
        m = Map(hdu.data.T, proj)
        return m, f
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
    from hpx_utils import make_coadd_hpx, HPX
    if isinstance(proj, pywcs.WCS):
        return make_coadd_wcs(maps, proj, shape)
    elif isinstance(proj, HPX):
        return make_coadd_hpx(maps, proj, shape)
    else:
        raise Exception("Can't co-add map of unknown type %s" % type(proj))


def make_coadd_wcs(maps, wcs, shape):
    data = np.zeros(shape)
    axes = wcs_to_axes(wcs, shape)

    for m in maps:
        c = wcs_to_coords(m.wcs, m.counts.shape)
        o = np.histogramdd(c.T, bins=axes[::-1], weights=np.ravel(m.counts))[0]
        data += o

    return Map(data, copy.deepcopy(wcs))


def wcs_to_axes(w, npix):
    """Generate a sequence of bin edge vectors corresponding to the
    axes of a WCS object."""

    npix = npix[::-1]

    x = np.linspace(-(npix[0]) / 2., (npix[0]) / 2.,
                    npix[0] + 1) * np.abs(w.wcs.cdelt[0])
    y = np.linspace(-(npix[1]) / 2., (npix[1]) / 2.,
                    npix[1] + 1) * np.abs(w.wcs.cdelt[1])

    cdelt2 = np.log10((w.wcs.cdelt[2] + w.wcs.crval[2]) / w.wcs.crval[2])

    z = (np.linspace(0, npix[2], npix[2] + 1)) * cdelt2
    z += np.log10(w.wcs.crval[2])

    return x, y, z


def wcs_to_coords(w, shape):
    """Generate an N x D list of pixel center coordinates where N is
    the number of pixels and D is the dimensionality of the map."""
    if w.naxis == 2:
        y, x = wcs_to_axes(w,shape)
    elif w.naxis == 3:
        z, y, x = wcs_to_axes(w,shape)
    else:
        raise Exception("Wrong number of WCS axes %i"%w.naxis)
    
    x = 0.5*(x[1:] + x[:-1])
    y = 0.5*(y[1:] + y[:-1])

    if w.naxis == 2:
        x = np.ravel(np.ones(shape)*x[:,np.newaxis])
        y = np.ravel(np.ones(shape)*y[np.newaxis,:])
        return np.vstack((x,y))    

    z = 0.5*(z[1:] + z[:-1])    
    x = np.ravel(np.ones(shape)*x[:,np.newaxis,np.newaxis])
    y = np.ravel(np.ones(shape)*y[np.newaxis,:,np.newaxis])       
    z = np.ravel(np.ones(shape)*z[np.newaxis,np.newaxis,:])
         
    return np.vstack((x,y,z))    
