# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
A few functions shared between real and simulated analyses
"""
from __future__ import absolute_import, division, print_function

import os
import sys

import numpy as np

from fermipy.utils import load_yaml

def baseline_roi_fit(gta, make_plots=False, minmax_npred=[1e3, np.inf]):
    """Do baseline fitting for a target Region of Interest

    Parameters
    ----------

    gta : `fermipy.gtaanalysis.GTAnalysis`
        The analysis object

    make_plots : bool
        Flag to make standard analysis plots

    minmax_npred : tuple or list
        Range of number of predicted coutns for which to free sources in initial fitting.

    """
    gta.free_sources(False)
    gta.write_roi('base_roi', make_plots=make_plots)
    
    gta.free_sources(True, minmax_npred=[1e3, np.inf])
    gta.optimize()
    gta.free_sources(False)      
    gta.print_roi()


def localize_sources(gta, **kwargs):    
    """Relocalize sources in the region of interest

    Parameters
    ----------

    gta : `fermipy.gtaanalysis.GTAnalysis`
        The analysis object

    kwargs : 
        These are passed to the gta.localize function

    """
    # Localize all point sources
    for src in sorted(gta.roi.sources, key=lambda t: t['ts'], reverse=True):
        #    for s in gta.roi.sources:
        
        if not src['SpatialModel'] == 'PointSource':
            continue
        if src['offset_roi_edge'] > -0.1:
            continue
        
        gta.localize(src.name, **kwargs)

    gta.optimize()
    gta.print_roi()


def add_source_get_correlated(gta, name, src_dict, correl_thresh=0.25, non_null_src=False):
    """Add a source and get the set of correlated sources

    Parameters
    ----------

    gta : `fermipy.gtaanalysis.GTAnalysis`
        The analysis object

    name : str
        Name of the source we are adding

    src_dict : dict
        Dictionary of the source parameters

    correl_thresh : float
        Threshold for considering a source to be correlated

    non_null_src : bool
        If True, don't zero the source

    Returns
    -------

    cdict : dict
        Dictionary with names and correlation factors of correlated sources 

    test_src_name : bool
        Name of the test source

    """
    if gta.roi.has_source(name):
        gta.zero_source(name)
        gta.update_source(name)
        test_src_name = "%s_test" % name
    else:
        test_src_name = name    

    gta.add_source(test_src_name, src_dict)
    gta.free_norm(test_src_name)
    gta.free_shape(test_src_name, free=False)
    fit_result = gta.fit(covar=True)

    mask = fit_result['is_norm']
    src_names = np.array(fit_result['src_names'])[mask]
    idx = (src_names == test_src_name).argmax()
    correl_vals = fit_result['correlation'][idx][mask]

    cdict = {}
    for src_name, correl_val in zip(src_names, correl_vals):
        if src_name == name:
            continue
        if np.fabs(correl_val) > 0.25:
            cdict[src_name] = correl_val

    if not non_null_src:
        gta.zero_source(test_src_name)

    gta.fit(covar=True)

    return cdict, test_src_name


def build_profile_dict(basedir, profile_name):
    """Get the name and source dictionary for the test source.
    
    Parameters
    ----------
    
    basedir : str
        Path to the analysis directory
        
    profile_name : str
        Key for the spatial from of the target

    Returns
    -------
    
    profile_name : str
        Name of for this particular profile

    src_name : str
        Name of the source for this particular profile
    
    profile_dict : dict
        Dictionary with the source parameters

    """
    profile_path = os.path.join(basedir, "profile_%s.yaml" % profile_name)
    profile_config = load_yaml(profile_path)
    src_name = profile_config['name']
    profile_dict = profile_config['source_model']
    return profile_name, src_name, profile_dict
