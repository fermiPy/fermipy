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
        self._config = merge_dict(self._config,config)
        self._config = merge_dict(self._config,kwargs)
        
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
            logger.info('Configuration:\n'+ yaml.dump(self.config))
        else:
            logger.log(loglevel,'Configuration:\n'+ yaml.dump(self.config))
        

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
    The defaults dictionary predefines valid key/value pairs and
    populates those with default values.  The config dictionary and
    kwargs are then used to update the values in the default
    configuration dictionary."""

    o = {}
    for key, item in defaults.iteritems():

        if isinstance(item,dict):
            o[key] = load_config(item)
        elif isinstance(item,tuple):
            item_list = [None,'',None,str]
            item_list[:len(item)] = item        
            value, comment, groupname, item_type = item_list

            if len(item) == 1:
                raise Exception('Option tuple must have at least one element.')
                    
            if value is None and (item_type == list or item_type == dict):
                value = item_type()
            
            keypath = key.split('.')

            if len(keypath) > 1:
                groupname = keypath[0]
                key = keypath[1]
                    
            if groupname:
                group = o.setdefault(groupname,{})
                group[key] = value
            else:
                if key in o: raise Exception('Duplicate key.')
                
                o[key] = value
        else:
            raise Exception('Unrecognized type for default dict element.')

    return o


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
        
        if not k in d1:
            od[k] = copy.copy(d0[k])
        elif isinstance(v,dict) and isinstance(d1[k],dict):
            od[k] = merge_dict(d0[k],d1[k],add_new_keys,append_arrays)
        elif isinstance(v,list) and isinstance(d1[k],str):
            od[k] = d1[k].split(',')            
        elif isinstance(v,np.ndarray) and append_arrays:
            od[k] = np.concatenate((v,d1[k]))
        elif (d0[k] is not None and d1[k] is not None) and \
                (type(d0[k]) != type(d1[k])):
            raise Exception('Conflicting types in dictionary merge.')
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
