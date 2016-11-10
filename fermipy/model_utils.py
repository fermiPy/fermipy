# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import os
import copy
import yaml
import numpy as np
from fermipy import utils


def get_function_par_names(name):
    """Get the list of parameters associated with a function.

    Parameters
    ----------
    name : str
        Name of the function.
    """
    
    fn_spec = get_function_spec(name)
    return copy.deepcopy(fn_spec['par_names'])


def get_function_norm_par_name(name):
    """Get the normalization parameter associated with a function.

    Parameters
    ----------
    name : str
        Name of the function.
    """
    
    fn_spec = get_function_spec(name)
    return fn_spec['norm_par']


def get_function_defaults(name):
    fn_spec = get_function_spec(name)
    return copy.deepcopy(fn_spec['defaults'])


def get_function_spec(name):
    """Return a dictionary with the specification of a function:
    parameter names and defaults (value, bounds, scale, etc.).

    Returns
    -------
    par_names : list
        List of parameter names for this function.

    norm_par : str
        Name of normalization parameter.
    
    default : dict
        Parameter defaults dictionary.
    """
    if not hasattr(get_function_spec,'fndict'):
        modelfile = os.path.join('$FERMIPY_ROOT',
                                 'data','models.yaml')
        modelfile = os.path.expandvars(modelfile)
        get_function_spec.fndict = yaml.load(open(modelfile))

    if not name in get_function_spec.fndict.keys():
        raise Exception('Invalid Function Name: %s'%name)
        
    return get_function_spec.fndict[name]
    

def get_source_type(spatial_type):
    """Translate a spatial type string to a source type."""

    if spatial_type == 'SkyDirFunction':
        return 'PointSource'
    else:
        return 'DiffuseSource'


def get_spatial_type(spatial_model):
    """Translate a spatial model string to a spatial type."""

    if spatial_model in ['SkyDirFunction', 'PointSource',
                         'Gaussian', 'PSFSource']:
        return 'SkyDirFunction'
    elif spatial_model in ['GaussianSource', 'DiskSource', 'SpatialMap']:
        return 'SpatialMap'
    elif spatial_model in ['RadialGaussian','RadialDisk']:
        try:
            import pyLikelihood        
            if hasattr(pyLikelihood,'RadialGaussian'):
                return spatial_model
            else:
                return 'SpatialMap'
        except Exception:
            return spatial_model            
    else:
        return spatial_model

    
def create_pars_dict(name,pars_dict=None):
    """Create a dictionary for the parameters of a function.

    Parameters
    ----------
    name : str
        Name of the function.

    pars_dict : dict    
        Existing parameter dict that will be merged with the
        default dictionary created by this method.
        
    """
    
    default_pars_dict = get_function_defaults(name)

    if pars_dict is None:
        pars_dict = {}
    else:
        pars_dict = copy.deepcopy(pars_dict)

    for k, v in pars_dict.items():

        if not k in default_pars_dict:
            continue

        if not isinstance(v,dict):
            pars_dict[k] = {'name' : k, 'value' : v}

    pars_dict = utils.merge_dict(default_pars_dict,pars_dict)

    for k, v in pars_dict.items():
        pars_dict[k] = make_parameter_dict(v)

    return pars_dict

    
def make_parameter_dict(pdict, fixed_par=False, rescale=True):
    """
    Update a parameter dictionary.  This function will automatically
    set the parameter scale and bounds if they are not defined.
    Bounds are also adjusted to ensure that they encompass the
    parameter value.
    """
    o = copy.deepcopy(pdict)

    if 'scale' not in o or o['scale'] is None:

        if rescale:        
            value, scale = utils.scale_parameter(o['value'])
        else:
            value, scale = o['value'], 1.0

        o['value'] = value
        o['scale'] = scale
        if 'error' in o:
            o['error'] /= np.abs(scale)

    if 'min' not in o:
        o['min'] = o['value']*1E-3

    if 'max' not in o:
        o['max'] = o['value']*1E3

    if fixed_par:
        o['min'] = o['value']
        o['max'] = o['value']

    if float(o['min']) > float(o['value']):
        o['min'] = o['value']

    if float(o['max']) < float(o['value']):
        o['max'] = o['value']

#    for k, v in o.items():
#        o[k] = str(v)

    return o


def cast_pars_dict(pars_dict):
    """Cast the bool and float elements of a parameters dict to
    the appropriate python types.
    """
    
    o = {}

    for pname, pdict in pars_dict.items():

        o[pname] = {}

        for k,v in pdict.items():

            if k == 'free':
                o[pname][k] = bool(int(v))
            elif k == 'name':
                o[pname][k] = v
            else:
                o[pname][k] = float(v)

    return o
