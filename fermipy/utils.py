import os
import copy
import yaml
import numpy as np
from collections import OrderedDict
import xml.etree.cElementTree as et
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.io.fits as pyfits

class AnalysisBase(object):
    """The base class provides common facilities like configuration
    parsing and saving state. """
    def __init__(self,config,**kwargs):
        self._config = self.get_config()
        self.configure(config,**kwargs)

    def configure(self,config,**kwargs):

        config = merge_dict(config,kwargs,add_new_keys=True)
        validate_config(config,self.defaults)
        
        self._config = merge_dict(self._config,config)
#        self._config = merge_dict(self._config,kwargs)
        
    @classmethod
    def get_config(cls):
        # Load defaults
        return load_config(cls.defaults)

    @property
    def config(self):
        """Return the configuration dictionary of this class."""
        return self._config

    def write_config(self,outfile):
        """Write the configuration dictionary to an output file."""
        yaml.dump(self.config,open(outfile,'w'),default_flow_style=False)
    
    def print_config(self,logger,loglevel=None):

        if loglevel is None:
            logger.info('Configuration:\n'+ yaml.dump(self.config,
                                                      default_flow_style=False))
        else:
            logger.log(loglevel,'Configuration:\n'+ yaml.dump(self.config,
                                                              default_flow_style=False))
        

def valToBin(edges,x):
    """Convert axis coordinate to bin index."""
    ibin = np.digitize(np.array(x,ndmin=1),edges)-1
    return ibin

def valToEdge(edges,x):
    """Convert axis coordinate to bin index."""
    edges = np.array(edges)
    w = edges[1:] - edges[:-1]
    w = np.insert(w,0,w[0])
    ibin = np.digitize(np.array(x,ndmin=1),edges-0.5*w)-1
    ibin[ibin<0] = 0    
    return ibin

def valToBinBounded(edges,x):
    """Convert axis coordinate to bin index."""
    nbins = len(edges)-1
    ibin = valToBin(edges,x)
    ibin[ibin < 0] = 0
    ibin[ibin > nbins-1] = nbins-1
    return ibin
            
def mkdir(dir):
    if not os.path.exists(dir):  os.makedirs(dir)
    return dir
    
def fits_recarray_to_dict(table):
    """Convert a FITS recarray to a python dictionary."""

    cols = {}
    for icol, col in enumerate(table.columns.names):

        col_data = table.data[col]
#            print icol, col, type(col_data[0])
        
        if type(col_data[0]) == np.float32: 
            cols[col] = np.array(col_data,dtype=float)
        elif type(col_data[0]) == np.float64: 
            cols[col] = np.array(col_data,dtype=float)
        elif type(col_data[0]) == str: 
            cols[col] = np.array(col_data,dtype=str)
        elif type(col_data[0]) == np.int16: 
            cols[col] = np.array(col_data,dtype=int)
        elif type(col_data[0]) == np.ndarray: 
            cols[col] = np.array(col_data)
        else:
            raise Exception('Unrecognized column type.')
    
    return cols

def create_xml_element(root,name,attrib):
    el = et.SubElement(root,name)
    for k, v in attrib.iteritems(): el.set(k,v)
    return el

def load_xml_elements(root,path):

    o = {}
    for p in root.findall(path):

        if 'name' in p.attrib:
            o[p.attrib['name']] = copy.deepcopy(p.attrib)
        else:
            o.update(p.attrib)
    return o
        
def prettify_xml(elem):
    """Return a pretty-printed XML string for the Element.
    """
    from xml.dom import minidom
    import xml.etree.cElementTree as et

    rough_string = et.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")  

def load_config(defaults):
    """Create a configuration dictionary from a defaults dictionary.
    The defaults dictionary defines valid configuration keys with
    default values and docstrings.  Each dictionary element should be
    a tuple or list containing (default value,docstring,type)."""

    o = {}
    for key, item in defaults.items():

        if isinstance(item,dict):
            o[key] = load_config(item)
        elif isinstance(item,tuple):
            item_list = [None,'',str]
            item_list[:len(item)] = item        
            value, comment, item_type = item_list

            if len(item) == 1:
                raise Exception('Option tuple must have at least one element.')
                    
            if value is None and (item_type == list or item_type == dict):
                value = item_type()
            
            if key in o: raise Exception('Duplicate key.')
                
            o[key] = value            
        else:

            print key, item, type(item)
            
            raise Exception('Unrecognized type for default dict element.')

    return o


def validate_config(config,defaults,block='root'):

    for key, item in config.items():
        
        if not key in defaults:
            raise Exception('Invalid key in \'%s\' block of configuration: %s'%
                            (block,key))
        
        if isinstance(item,dict):
            validate_config(config[key],defaults[key],key)

def merge_dict(d0,d1,add_new_keys=False,append_arrays=False):
    """Recursively merge the contents of python dictionary d0 with
    the contents of another python dictionary, d1.

    add_new_keys : Do not skip keys that only exist in d1.

    append_arrays : If an element is a numpy array set the value of
    that element by concatenating the two arrays.
    """
    
    if d1 is None: return d0
    elif d0 is None: return d1
    elif d0 is None and d1 is None: return {}

    od = {}
    
    for k, v in d0.iteritems():

        t0 = None
        t1 = None
        
        if k in d0: t0 = type(d0[k])
        if k in d1: t1 = type(d1[k])
        
        if not k in d1:
            od[k] = copy.deepcopy(d0[k])
        elif isinstance(v,dict) and isinstance(d1[k],dict):
            od[k] = merge_dict(d0[k],d1[k],add_new_keys,append_arrays)
        elif isinstance(v,list) and isinstance(d1[k],str):
            od[k] = d1[k].split(',')            
        elif isinstance(v,np.ndarray) and append_arrays:
            od[k] = np.concatenate((v,d1[k]))
        elif (d0[k] is not None and d1[k] is not None) and t0 != t1:

            if t0 == dict or t0 == list: 
                raise Exception('Conflicting types in dictionary merge for '
                                'key %s %s %s'%(k,t0,t1))
            od[k] = t0(d1[k]) 
        else: od[k] = copy.copy(d1[k])

    if add_new_keys:
        for k, v in d1.iteritems(): 
            if not k in d0: od[k] = copy.deepcopy(d1[k])

    return od

def tolist(x):
    """ convenience function that takes in a 
        nested structure of lists and dictionaries
        and converts everything to its base objects.
        This is useful for dupming a file to yaml.

        (a) numpy arrays into python lists

            >>> type(tolist(np.asarray(123))) == int
            True
            >>> tolist(np.asarray([1,2,3])) == [1,2,3]
            True

        (b) numpy strings into python strings.

            >>> tolist([np.asarray('cat')])==['cat']
            True

        (c) an ordered dict to a dict

            >>> ordered=OrderedDict(a=1, b=2)
            >>> type(tolist(ordered)) == dict
            True

        (d) converts unicode to regular strings

            >>> type(u'a') == str
            False
            >>> type(tolist(u'a')) == str
            True

        (e) converts numbers & bools in strings to real represntation,
            (i.e. '123' -> 123)

            >>> type(tolist(np.asarray('123'))) == int
            True
            >>> type(tolist('123')) == int
            True
            >>> tolist('False') == False
            True
    """
    if isinstance(x,list):
        return map(tolist,x)
    elif isinstance(x,dict):
        return dict((tolist(k),tolist(v)) for k,v in x.items())
    elif isinstance(x,np.ndarray) or isinstance(x,np.number):
        # note, call tolist again to convert strings of numbers to numbers
        return tolist(x.tolist())
    elif isinstance(x,OrderedDict):
        return dict(x)
    elif isinstance(x,basestring) or isinstance(x,np.str):
        x=str(x) # convert unicode & numpy strings 
        try:
            return int(x)
        except:
            try:
                return float(x)
            except:
                if x == 'True': return True
                elif x == 'False': return False
                else: return x
    else:
        return x


def create_wcs(skydir,coordsys='CEL',projection='AIT',
               cdelt=1.0,crpix=1.,naxis=2):
    """Create a WCS object.

    Parameters
    ----------

    skydir : SkyCoord
        Sky coordinate of the WCS reference point.

    """
    
    from astropy import wcs

    w = wcs.WCS(naxis=naxis)
#    w = wcs.WCS()
    
    if coordsys == 'CEL':
        w.wcs.ctype[0] = 'RA---%s'%(projection)
        w.wcs.ctype[1] = 'DEC--%s'%(projection)
        w.wcs.crval[0]=skydir.ra.deg
        w.wcs.crval[1]=skydir.dec.deg
    elif coordsys == 'GAL':
        w.wcs.ctype[0] = 'GLON-%s'%(projection)
        w.wcs.ctype[1] = 'GLAT-%s'%(projection)        
        w.wcs.crval[0]=skydir.galactic.l.deg
        w.wcs.crval[1]=skydir.galactic.b.deg
    else:
        raise Exception('Unrecognized coordinate system.')
    
    w.wcs.crpix[0] = crpix
    w.wcs.crpix[1] = crpix
    w.wcs.cdelt[0] = -cdelt
    w.wcs.cdelt[1] = cdelt

    w = wcs.WCS(w.to_header())    
    return w
    
def offset_to_sky(skydir,offset_lon,offset_lat,
                  coordsys='CEL',projection='AIT'):
    """Convert a coordinate offset (X,Y) in the given projection into
    a pair of spherical coordinates."""
    
    offset_lon = np.array(offset_lon,ndmin=1)
    offset_lat = np.array(offset_lat,ndmin=1)

    w = create_wcs(skydir,coordsys,projection)
    pixcrd = np.vstack((offset_lon,offset_lat)).T
    
    return w.wcs_pix2world(pixcrd,0)

def sky_to_offset(skydir,lon,lat,coordsys='CEL',projection='AIT'):
    """Convert sky coordinates to a projected offset.  This function
    is the inverse of offset_to_sky."""
    
    w = create_wcs(skydir,coordsys,projection)
    skycrd = np.vstack((lon,lat)).T
    
    return w.wcs_world2pix(skycrd,0)

def wcs_to_axes(w,npix):
    """Generate a sequence of bin edge vectors corresponding to the
    axes of a WCS object."""
    
    npix = npix[::-1]
    
    x = np.linspace(-(npix[0])/2.,(npix[0])/2.,
                    npix[0]+1)*np.abs(w.wcs.cdelt[0])
    y = np.linspace(-(npix[1])/2.,(npix[1])/2.,
                    npix[1]+1)*np.abs(w.wcs.cdelt[1])

    cdelt2 = np.log10((w.wcs.cdelt[2]+w.wcs.crval[2])/w.wcs.crval[2])
    
    z = (np.linspace(0,npix[2],npix[2]+1))*cdelt2
    z += np.log10(w.wcs.crval[2])

    return x, y, z

def wcs_to_coords(w,shape):
    """Generate an N x D list of pixel center coordinates where N is
    the number of pixels and D is the dimensionality of the map."""
    
    z, y, x = wcs_to_axes(w,shape)

    x = 0.5*(x[1:] + x[:-1])
    y = 0.5*(y[1:] + y[:-1])
    z = 0.5*(z[1:] + z[:-1])    

    x = np.ravel(np.ones(shape)*x[:,np.newaxis,np.newaxis])
    y = np.ravel(np.ones(shape)*y[np.newaxis,:,np.newaxis])
    z = np.ravel(np.ones(shape)*z[np.newaxis,np.newaxis,:])
            
    return np.vstack((x,y,z))    
    
def get_target_skydir(config):

    radec = config.get('radec',None)

    if isinstance(radec,str):
        return SkyCoord(radec,unit=u.deg)
    elif isinstance(radec,list):
        return SkyCoord(radec[0],radec[1],unit=u.deg)
    
    ra = config.get('ra',None)
    dec = config.get('dec',None)
    
    if not ra is None and not dec is None:
        return SkyCoord(ra,dec,unit=u.deg)

    glon = config.get('glon',None)
    glat = config.get('glat',None)

    if not glon is None and not glat is None:
        return SkyCoord(glon,glat,unit=u.deg,
                        frame='galactic').transform_to('icrs')

    return None


def make_gaussian_kernel(sigma,npix=501,cdelt=0.01):

    sigma /= 1.5095921854516636        
    sigma /= cdelt
    
    fn = lambda t, s: 1./(2*np.pi*s**2)*np.exp(-t**2/(s**2*2.0))
    
    b = np.abs(np.linspace(0,npix-1,npix) - (npix-1)/2.)
    k = np.zeros((npix,npix)) + np.sqrt(b[np.newaxis,:]**2 +
                                        b[:,np.newaxis]**2)
    k = fn(k,sigma)

    # Normalize map to 1
    k /= (np.sum(k)*np.radians(cdelt)**2)

    return k
    
def make_disk_kernel(sigma,npix=501,cdelt=0.01):

    sigma /= cdelt    
    fn = lambda t, s: 0.5 * (np.sign(s-t) + 1.0)
    
    b = np.abs(np.linspace(0,npix-1,npix) - (npix-1)/2.)
    k = np.zeros((npix,npix)) + np.sqrt(b[np.newaxis,:]**2 +
                                        b[:,np.newaxis]**2)
    k = fn(k,sigma)

    # Normalize map to 1
    k /= (np.sum(k)*np.radians(cdelt)**2)

    return k

def make_gaussian_spatial_map(skydir,sigma,outfile,npix=501,cdelt=0.01):
    
    w = create_wcs(skydir,cdelt=cdelt,crpix=npix/2.+0.5)    
    hdu_image = pyfits.PrimaryHDU(np.zeros((npix,npix)),
                                  header=w.to_header())
    
    hdu_image.data[:,:] = make_gaussian_kernel(sigma,npix=npix,cdelt=cdelt)
    hdulist = pyfits.HDUList([hdu_image])
    hdulist.writeto(outfile,clobber=True) 

def make_disk_spatial_map(skydir,sigma,outfile,npix=501,cdelt=0.01):

    sigma /= cdelt    
    w = create_wcs(skydir,cdelt=cdelt,crpix=npix/2.+0.5)
    
    hdu_image = pyfits.PrimaryHDU(np.zeros((npix,npix)),
                                  header=w.to_header())
    
    hdu_image.data[:,:] = make_disk_kernel(sigma,npix=npix,cdelt=cdelt)
    hdulist = pyfits.HDUList([hdu_image])
    hdulist.writeto(outfile,clobber=True) 

def make_coadd_map(maps,wcs,shape):
        
    header = wcs.to_header()
    data = np.zeros(shape)
    axes = wcs_to_axes(wcs,shape)
    
    for z, w in maps:
        c = wcs_to_coords(w,z.shape)
        o = np.histogramdd(c.T,bins=axes[::-1],weights=np.ravel(z))[0]
        data += o
        
    return data

def write_fits_image(data,wcs,outfile):
    
    hdu_image = pyfits.PrimaryHDU(data,header=wcs.to_header())
#        hdulist = pyfits.HDUList([hdu_image,h['GTI'],h['EBOUNDS']])
    hdulist = pyfits.HDUList([hdu_image])        
    hdulist.writeto(outfile,clobber=True)    
