import os
import yaml
import numpy as np

import xml.etree.cElementTree as et

class AnalysisBase(object):
    """The base class provides common facilities like configuration
    parsing and saving state. """
    def __init__(self,config,**kwargs):
        self._config = self.get_config()
        self.configure(config,**kwargs)

    def configure(self,config,**kwargs):
        update_dict(self._config,config)
        update_dict(self._config,kwargs)
        
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


def update_dict(d0,d1,add_new_keys=False,append=False):
    """Recursively update the contents of python dictionary d0 with
    the contents of another python dictionary, d1.

    add_new_keys : Do not skip keys that already exist in d0.
    """

    if d0 is None or d1 is None: return
    
    for k, v in d0.iteritems():

        if not k in d1: continue

        if isinstance(v,dict) and isinstance(d1[k],dict):
            update_dict(d0[k],d1[k],add_new_keys,append)
        elif isinstance(v,list) and isinstance(d1[k],str):
            d0[k] = d1[k].split(',')            
        elif isinstance(v,np.ndarray) and append:
            d0[k] = np.concatenate((v,d1[k]))
        else: d0[k] = d1[k]

    if add_new_keys:
        for k, v in d1.iteritems(): 
            if not k in d0: d0[k] = d1[k]
