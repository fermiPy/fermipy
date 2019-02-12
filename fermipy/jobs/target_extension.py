#!/usr/bin/env python

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Module with classes for target extension analysis
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import numpy as np

from fermipy.utils import load_yaml, write_yaml, init_matplotlib_backend

from fermipy.jobs.utils import is_null, is_not_null
from fermipy.jobs.link import Link
from fermipy.jobs.analysis_utils import add_source_get_correlated, build_profile_dict
from fermipy.jobs.scatter_gather import ScatterGather

from fermipy.jobs.name_policy import NameFactory
from fermipy.jobs import defaults

init_matplotlib_backend('Agg')

try:
    from fermipy.gtanalysis import GTAnalysis
    HAVE_ST = True
except ImportError:
    HAVE_ST = False

NAME_FACTORY = NameFactory(basedir=('.'))



class AnalyzeExtension(Link):
    """Small class to wrap an analysis script.

    This particular script does target extension analysis
    with respect to the baseline ROI model.
    """
    appname = 'fermipy-analyze-extension'
    linkname_default = 'analyze-extension'
    usage = '%s [options]' % (appname)
    description = "Analyze extension for a single target"

    default_options = dict(config=defaults.common['config'],
                           roi_baseline=defaults.common['roi_baseline'],
                           make_plots=defaults.common['make_plots'],
                           target=defaults.common['target'])

    __doc__ += Link.construct_docstring(default_options)

    def run_analysis(self, argv):
        """Run this analysis"""
        args = self._parser.parse_args(argv)

        if not HAVE_ST:
            raise RuntimeError(
                "Trying to run fermipy analysis, but don't have ST")

        if is_not_null(args.roi_baseline):
            gta = GTAnalysis.create(args.roi_baseline, args.config)
        else:
            gta = GTAnalysis(args.config,
                             logging={'verbosity': 3},
                             fileio={'workdir_regex': '\.xml$|\.npy$'})
        gta.print_roi()
        
        test_source = args.target
        gta.sed(test_source, outfile='sed_%s.fits' % 'FL8Y', make_plots=True)
        gta.extension(test_source, make_plots=True)
        return gta


class AnalyzeExtension_SG(ScatterGather):
    """Small class to generate configurations for this script

    This loops over all the targets defined in the target list,
    and over all the profiles defined for each target.
    """
    appname = 'fermipy-analyze-extension-sg'
    usage = "%s [options]" % (appname)
    description = "Run analyses on a series of ROIs"
    clientclass = AnalyzeExtension

    job_time = 1500

    default_options = dict(ttype=defaults.common['ttype'],
                           targetlist=defaults.common['targetlist'],
                           config=defaults.common['config'],
                           roi_baseline=defaults.common['roi_baseline'],
                           make_plots=defaults.common['make_plots'])

    __doc__ += Link.construct_docstring(default_options)

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        ttype = args['ttype']
        (targets_yaml, sim) = NAME_FACTORY.resolve_targetfile(args)
        if sim is not None:
            raise ValueError("Found 'sim' argument on AnalyzeExtension_SG config.")
        if targets_yaml is None:
            return job_configs

        targets = load_yaml(targets_yaml)
        config_yaml = 'config.yaml'

        base_config = dict(roi_baseline=args['roi_baseline'],
                           make_plots=args['make_plots'])

        for target_name, target_list in targets.items():
            name_keys = dict(target_type=ttype,
                             target_name=target_name,
                             fullpath=True)
            target_dir = NAME_FACTORY.targetdir(**name_keys)
            config_path = os.path.join(target_dir, config_yaml)
            logfile = make_nfs_path(os.path.join(
                target_dir, "%s_%s.log" % (self.linkname, target_name)))
            job_config = base_config.copy()
            job_config.update(dict(config=config_path,
                                   logfile=logfile))
            job_configs[target_name] = job_config

        return job_configs


def register_classes():
    """Register these classes with the `LinkFactory` """
    AnalyzeExtension.register_class()
    AnalyzeExtension_SG.register_class()
