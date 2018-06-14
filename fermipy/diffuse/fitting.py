# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Scripts and tools to do all-sky diffuse fitting
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import numpy as np

from fermipy.utils import load_yaml, init_matplotlib_backend
init_matplotlib_backend('Agg')

from fermipy import plotting

from fermipy.jobs.utils import is_null, is_not_null
from fermipy.jobs.link import Link
from fermipy.jobs.scatter_gather import ScatterGather
from fermipy.jobs.slac_impl import make_nfs_path
from fermipy.jobs.analysis_utils import baseline_roi_fit, localize_sources,\
    add_source_get_correlated

from fermipy.diffuse import defaults

try:
    from fermipy.gtanalysis import GTAnalysis
    HAVE_ST = True
except ImportError:
    HAVE_ST = False



def build_srcdict(gta, prop):
    """Build a dictionary that maps from source name to the value of a source property

    Parameters
    ----------

    gta : `fermipy.GTAnalysis`
        The analysis object

    prop : str
        The name of the property we are mapping


    Returns
    -------

    odict : dict
        Dictionary that maps from source name to the value of the specified property

    """
    o = {}
    for s in gta.roi.sources:
        o[s.name] = s[prop]
    return o


def get_src_names(gta):
    """Build and return a list of source name

    Parameters
    ----------

    gta : `fermipy.GTAnalysis`
        The analysis object


    Returns
    -------

    l : list
       Names of the source

    """
    o = []
    for s in gta.roi.sources:
        o += [s.name]
    return sorted(o)


def set_wts_get_npred_wt(gta, maskname):
    """Set a weights file and get the weighted npred for all the sources

    Parameters
    ----------

    gta : `fermipy.GTAnalysis`
        The analysis object

    maskname : str
        The path to the file with the mask


    Returns
    -------

    odict : dict
        Dictionary mapping from source name to weighted npred

    """
    if is_null(maskname):
        maskname = None

    gta.set_weights_map(maskname)
    for name in gta.like.sourceNames():
        gta._init_source(name)
    gta._update_roi()
    return build_srcdict(gta, 'npred_wt')


def snapshot(gta, plotter, key, do_weighted=True, make_plots=True):
    """Take a snapshot of the ROI

    Parameters
    ----------

    gta : `fermipy.GTAnalysis`
        The analysis object

    plotter : `fermipy.plotting.AnalysisPlotter`
        The object that makes the plots

    key : str
        Key for this snapshot, used to create filenames

    do_weighted : bool
        If True, include weighted version of outputs

    make_plots : bool
        If True, make plots

    """
    gta.write_roi(key, save_model_map=True, make_plots=make_plots, save_weight_map=do_weighted)
    if make_plots:
        o = gta.residmap(key)
        plotter.make_residmap_plots(o, gta.roi)

        if do_weighted:
            gta.make_plots("%s_wt"%key, weighted=True)
            o = gta.residmap("%s_wt"%key, use_weights=True)
            plotter.make_residmap_plots(o, gta.roi)


def get_unchanged(src_list, npred_dict_new,
                  npred_dict_old,
                  npred_threshold=1e4,
                  frac_threshold=0.9):
    """Compare two dictionarys of npreds, and get the list of sources
    than have changed less that set thresholds

    Parameters
    ----------

    src_list : list
        List of sources to examine

    npred_dict_new : dict
        Dictionary mapping source name to npred for the current weights file

    npred_dict_old : dict
        Dictionary mapping source name to npred for the previous weights file

    npred_threshold : float
        Minimum value of npred above which to consider sources changed

    frac_threshold : float
        Value of npred_old / npred_new above which to consider sources unchanged


    Returns
    -------

    l : list
       Names of 'unchanged' sources

    """
    o = []
    for s in src_list:
        npred_new = npred_dict_new[s]
        if npred_new < npred_threshold:
            o += [s]
            continue
        if npred_dict_old is None:
            npred_old = 0.
        else:
            npred_old = npred_dict_old[s]
        frac = npred_old / npred_new
        if frac > frac_threshold:
            o += [s]
    return o



class FitDiffuse(Link):
    """Perform all-sky diffuse fits.

    This particular script does baseline fitting of an ROI.
    """
    appname = 'fermipy-fit-diffuse'
    linkname_default = 'fit-diffuse'
    usage = '%s [options]' % (appname)
    description = "All-sky diffuse fitting"

    default_options = dict(config=defaults.diffuse['config'],
                           roi_baseline=defaults.diffuse['roi_baseline'],
                           fit_strategy=defaults.diffuse['fit_strategy'],
                           input_pars=defaults.diffuse['input_pars'],
                           load_baseline=defaults.diffuse['load_baseline'],
                           make_plots=defaults.diffuse['make_plots'])

    __doc__ += Link.construct_docstring(default_options)

    def run_analysis(self, argv):
        """Run this analysis"""
        args = self._parser.parse_args(argv)

        if not HAVE_ST:
            raise RuntimeError(
                "Trying to run fermipy analysis, but don't have ST")

        if (args.load_baseline):
            gta = GTAnalysis.create(args.roi_baseline,
                                    args.config)
        else:
            gta = GTAnalysis(args.config,
                             logging={'verbosity': 3},
                             fileio={'workdir_regex': '\.xml$|\.npy$'})
            gta.setup()
            if is_not_null(args.input_pars):
                gta.load_parameters_from_yaml(args.input_pars)
            gta.write_roi(args.roi_baseline,
                          save_model_map=True,
                          save_weight_map=True,
                          make_plots=args.make_plots)

        src_list = get_src_names(gta)
        plotter = plotting.AnalysisPlotter(gta.config['plotting'],
                                           fileio=gta.config['fileio'],
                                           logging=gta.config['logging'])

        if is_null(args.fit_strategy):
            return

        fit_strategy = load_yaml(args.fit_strategy)
        npred_current = None
        npred_prev = None

        for fit_stage in fit_strategy:
            mask = fit_stage.get('mask', None)
            npred_threshold = fit_stage.get('npred_threshold', 1.0e4)
            frac_threshold = fit_stage.get('frac_threshold', 0.5)
            npred_frac = fit_stage.get('npred_frac', 0.9999)

            npred_current =  set_wts_get_npred_wt(gta, mask)
            skip_list_region = get_unchanged(src_list,
                                             npred_current,
                                             npred_prev,
                                             frac_threshold=frac_threshold)
            gta.optimize(npred_frac=npred_frac, 
                         npred_threshold=npred_threshold,
                         skip=skip_list_region)
            snapshot(gta, plotter, fit_stage['key'], make_plots=args.make_plots)
            npred_prev = npred_current
            npred_current = build_srcdict(gta, 'npred_wt')



class FitDiffuse_SG(ScatterGather):
    """Small class to generate configurations for this script

    This loops over all the models defined in the model list.
    """
    appname = 'fermipy-fit-diffuse-sg'
    usage = "%s [options]" % (appname)
    description = "Run analyses on a series of models"
    clientclass = FitDiffuse

    job_time = 6000

    default_options = dict(models=defaults.diffuse['models'],
                           config=defaults.diffuse['config'],
                           roi_baseline=defaults.diffuse['roi_baseline'],
                           fit_strategy=defaults.diffuse['fit_strategy'],
                           input_pars=defaults.diffuse['input_pars'],
                           load_baseline=defaults.diffuse['load_baseline'],
                           make_plots=defaults.diffuse['make_plots'])

    __doc__ += Link.construct_docstring(default_options)


    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        # Tweak the batch job args
        try:
            self._interface._lsf_args.update(dict(n=2))
        except AttributeError:
            pass

        models = load_yaml(args['models'])

        base_config = dict(fit_strategy=args['fit_strategy'],
                           input_pars=args['input_pars'],
                           load_baseline=args['load_baseline'],
                           make_plots=args['make_plots'])

        for modelkey in models:
            config_file = os.path.join('analysis', 'model_%s' % modelkey,
                                       args['config'])
            roi_baseline = os.path.join('analysis', 'model_%s' % modelkey,
                                       args['roi_baseline'])
            logfile = os.path.join('analysis', 'model_%s' % modelkey,
                                   'fit_%s.log' % modelkey)
            job_config = base_config.copy()
            job_config.update(dict(config=config_file,
                                   roi_baseline=roi_baseline,
                                   logfile=logfile))
            job_configs[modelkey] = job_config

        return job_configs



def register_classes():
    """Register these classes with the `LinkFactory` """
    FitDiffuse.register_class()
    FitDiffuse_SG.register_class()
