import os
import copy
import yaml
import numpy as np
from collections import OrderedDict
import xml.etree.cElementTree as et

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
        """Return the internal configuration state of this class."""
        return self._config

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
            raise Exception('Invalid key in \'%s\' block of configuration: %s'%(block,key))
        
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
            od[k] = copy.copy(d0[k])
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
            if not k in d0: od[k] = copy.copy(d1[k])

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


def get_offset_wcs(lon,lat,coordsys='GAL',projection='AIT'):

    from astropy import wcs
    
    if coordsys == 'CEL':
        ctype1 = 'RA--'
        ctype2 = 'DEC-'
    elif coordsys == 'GAL':
        ctype1 = 'GLON'
        ctype2 = 'GLAT'
    else:
        raise Exception('Unrecognized coordinate system.')
    
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [1.,1.]
    w.wcs.cdelt = np.array([-1.0, 1.0])
    w.wcs.crval = [np.squeeze(lon), np.squeeze(lat)]
    w.wcs.ctype = ["%s-%s"%(ctype1,projection),
                   "%s-%s"%(ctype2,projection)]

    return w
    
def offset_to_sky(lon,lat,offset_lon,offset_lat,
                  coordsys='GAL',projection='AIT'):
    """Convert a coordinate offset (X,Y) in the given projection into
    a pair of spherical coordinates."""
    
    offset_lon = np.array(offset_lon,ndmin=1)
    offset_lat = np.array(offset_lat,ndmin=1)

    w = get_offset_wcs(lon,lat,coordsys,projection)
    pixcrd = np.vstack((offset_lon,offset_lat)).T
    
    return w.wcs_pix2world(pixcrd,0)

def sky_to_offset(lon,lat,lon1,lat1,coordsys='GAL',projection='AIT'):
    """Convert sky coordinates to a projected offset.  This function
    is the inverse of offset_to_sky."""
    
    lon = np.array(lon,ndmin=1)
    lat = np.array(lat,ndmin=1)
    
    w = get_offset_wcs(lon,lat,coordsys,projection)
    skycrd = np.vstack((lon1,lat1)).T
    
    return w.wcs_world2pix(skycrd,0)
