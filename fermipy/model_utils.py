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
    if not hasattr(get_function_spec, 'fndict'):
        modelfile = os.path.join('$FERMIPY_ROOT',
                                 'data', 'models.yaml')
        modelfile = os.path.expandvars(modelfile)
        get_function_spec.fndict = yaml.load(open(modelfile))

    if not name in get_function_spec.fndict.keys():
        raise Exception('Invalid Function Name: %s' % name)

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
                         'Gaussian']:
        return 'SkyDirFunction'
    elif spatial_model in ['SpatialMap']:
        return 'SpatialMap'
    elif spatial_model in ['RadialGaussian', 'RadialDisk']:
        try:
            import pyLikelihood
            if hasattr(pyLikelihood, 'RadialGaussian'):
                return spatial_model
            else:
                return 'SpatialMap'
        except Exception:
            return spatial_model
    else:
        return spatial_model


def extract_pars_from_dict(name, src_dict):

    par_names = get_function_par_names(name)

    o = {}
    for k in par_names:

        o[k] = {}

        if not k in src_dict:
            continue

        v = src_dict.pop(k)

        if isinstance(v, dict):
            o[k] = v.copy()
        else:
            o[k] = {'name': k, 'value': v}

    return o


def create_pars_from_dict(name, pars_dict, rescale=True, update_bounds=False):
    """Create a dictionary for the parameters of a function.

    Parameters
    ----------
    name : str
        Name of the function.

    pars_dict : dict    
        Existing parameter dict that will be merged with the
        default dictionary created by this method.

    rescale : bool
        Rescale parameter values.

    """
    o = get_function_defaults(name)
    pars_dict = pars_dict.copy()

    for k in o.keys():

        if not k in pars_dict:
            continue

        v = pars_dict[k]

        if not isinstance(v, dict):
            v = {'name': k, 'value': v}

        o[k].update(v)

        kw = dict(update_bounds=update_bounds,
                  rescale=rescale)

        if 'min' in v or 'max' in v:
            kw['update_bounds'] = False

        if 'scale' in v:
            kw['rescale'] = False

        o[k] = make_parameter_dict(o[k], **kw)

    return o


def make_parameter_dict(pdict, fixed_par=False, rescale=True,
                        update_bounds=False):
    """
    Update a parameter dictionary.  This function will automatically
    set the parameter scale and bounds if they are not defined.
    Bounds are also adjusted to ensure that they encompass the
    parameter value.
    """
    o = copy.deepcopy(pdict)
    o.setdefault('scale', 1.0)

    if rescale:
        value, scale = utils.scale_parameter(o['value'] * o['scale'])
        o['value'] = np.abs(value) * np.sign(o['value'])
        o['scale'] = np.abs(scale) * np.sign(o['scale'])
        if 'error' in o:
            o['error'] /= np.abs(scale)

    if update_bounds:
        o['min'] = o['value'] * 1E-3
        o['max'] = o['value'] * 1E3

    if fixed_par:
        o['min'] = o['value']
        o['max'] = o['value']

    if float(o['min']) > float(o['value']):
        o['min'] = o['value']

    if float(o['max']) < float(o['value']):
        o['max'] = o['value']

    return o


def cast_pars_dict(pars_dict):
    """Cast the bool and float elements of a parameters dict to
    the appropriate python types.
    """

    o = {}

    for pname, pdict in pars_dict.items():

        o[pname] = {}

        for k, v in pdict.items():

            if k == 'free':
                o[pname][k] = bool(int(v))
            elif k == 'name':
                o[pname][k] = v
            else:
                o[pname][k] = float(v)

    return o


def pars_dict_to_vectors(function_name, pars_dict):

    o = {'param_names' : np.zeros(10, dtype='S32'),
         'param_values' : np.empty(10, dtype=float) * np.nan,
         'param_errors' : np.empty(10, dtype=float) * np.nan,
         }
    
    par_names = get_function_par_names(function_name)
    for i, p in enumerate(par_names):

        value = pars_dict[p]['value']*pars_dict[p]['scale']
        scale = pars_dict[p]['error']*pars_dict[p]['scale']        
        o['param_names'][i] = p
        o['param_values'][i] = value
        o['param_errors'][i] = scale
        
    return o
