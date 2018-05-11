# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
A few functions shared between real and simulated analyses
"""
from __future__ import absolute_import, division, print_function

import numpy as np

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


def add_source_get_correlated(gta, name, src_dict, correl_thresh=0.25):
    
    gta.add_source(name, src_dict)
    fit_result = gta.fit(covar=True)
    mask = fit_result['is_norm']
    src_names = np.array(fit_result['src_names'])[mask]
    idx = (src_names == name).argmax()
    correl_vals = fit_result['correlation'][idx][mask]

    clist = []
    for src_name, correl_val in zip(src_names, correl_vals):
        if np.fabs(correl_val) > 0.25:
            clist.append(src_name)

    gta.zero_source(name)
    gta.fit(covar=True)

    return clist
