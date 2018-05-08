#!/usr/bin/env python

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Run gtsrcmaps for a single energy plane for a single source

This is useful to parallize the production of the source maps
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import numpy as np

from fermipy.utils import load_yaml, init_matplotlib_backend

from fermipy.jobs.utils import is_null, is_not_null
from fermipy.jobs.link import Link
from fermipy.jobs.scatter_gather import ScatterGather
from fermipy.jobs.slac_impl import make_nfs_path

from fermipy.jobs.name_policy import NameFactory
from fermipy.jobs import defaults

init_matplotlib_backend('Agg')

try:
    from fermipy.gtanalysis import GTAnalysis
    HAVE_ST = True
except ImportError:
    HAVE_ST = False

NAME_FACTORY = NameFactory(basedir=('.'))


class AnalyzeROI(Link):
    """Small class wrap an analysis script.

    This is useful for parallelizing analysis using the fermipy.jobs module.
    """
    appname = 'fermipy-analyze-roi'
    linkname_default = 'analyze-roi'
    usage = '%s [options]' % (appname)
    description = "Run analysis of a single ROI"

    default_options = dict(config=defaults.common['config'],
                           roi_baseline=defaults.common['roi_baseline'],
                           make_plots=defaults.common['make_plots'])

    def run_analysis(self, argv):
        """Run this analysis"""
        args = self._parser.parse_args(argv)

        if not HAVE_ST:
            raise RuntimeError(
                "Trying to run fermipy analysis, but don't have ST")

        gta = GTAnalysis(args.config, logging={'verbosity': 3},
                         fileio={'workdir_regex': '\.xml$|\.npy$'})

        gta.setup(overwrite=False)
        gta.free_sources(False)
        gta.write_roi('base_roi', make_plots=args.make_plots)

        gta.print_roi()
        gta.optimize()
        gta.print_roi()

        exclude = []

        # Localize all point sources
        for src in sorted(gta.roi.sources, key=lambda t: t['ts'], reverse=True):
            #    for s in gta.roi.sources:

            if not src['SpatialModel'] == 'PointSource':
                continue
            if src['offset_roi_edge'] > -0.1:
                continue

            if src.name in exclude:
                continue

            gta.localize(src.name, nstep=5, dtheta_max=0.5, update=True,
                         prefix='base', make_plots=args.make_plots)

        gta.optimize()
        gta.print_roi()

        gta.find_sources(sqrt_ts_threshold=5.0, search_skydir=gta.roi.skydir,
                         search_minmax_radius=[1.0, np.nan])
        gta.optimize()
        gta.print_roi()
        gta.print_params()

        gta.free_sources(skydir=gta.roi.skydir, distance=1.0, pars='norm')
        gta.fit()
        gta.print_roi()
        gta.print_params()

        gta.write_roi(args.roi_baseline, make_plots=args.make_plots)


class AnalyzeSED(Link):
    """Small class wrap an analysis script.

    This is useful for parallelizing analysis using the fermipy.jobs module.
    """
    appname = 'fermipy-analyze-sed'
    linkname_default = 'analyze-sed'
    usage = '%s [options]' % (appname)
    description = "Extract the SED for a single target"

    default_options = dict(config=defaults.common['config'],
                           roi_baseline=defaults.common['roi_baseline'],
                           skydirs=defaults.sims['skydirs'],
                           profiles=defaults.common['profiles'],
                           make_plots=defaults.common['make_plots'])

    @staticmethod
    def _build_profile_dict(basedir, profile_name):
        """
        """
        profile_path = os.path.join(basedir, "profile_%s.yaml" % profile_name)
        profile_config = load_yaml(profile_path)
        if profile_name != profile_config['name']:
            sys.stderr.write('Warning, profile name (%s) != name in %s (%s)\n' % (
                profile_name, profile_config['name'], profile_path))

        profile_dict = profile_config['source_model']
        return profile_name, profile_dict

    def run_analysis(self, argv):
        """Run this analysis"""
        args = self._parser.parse_args(argv)

        if not HAVE_ST:
            raise RuntimeError(
                "Trying to run fermipy analysis, but don't have ST")

        if is_null(args.skydirs):
            skydir_dict = None
        else:
            skydir_dict = load_yaml(args.skydirs)

        gta = GTAnalysis(args.config,
                         logging={'verbosity': 3},
                         fileio={'workdir_regex': '\.xml$|\.npy$'})
        gta.setup(overwrite=False)
        gta.load_roi('fit_baseline')
        gta.print_roi()

        basedir = os.path.dirname(args.config)
        # This should be a no-op, b/c it was done in the baseline analysis

        gta.free_sources(skydir=gta.roi.skydir, distance=1.0, pars='norm')

        for profile in args.profiles:
            if skydir_dict is None:
                skydir_keys = [None]
            else:
                skydir_keys = sorted(skydir_dict.keys())

            for skydir_key in skydir_keys:
                if skydir_key is None:
                    pkey, pdict = AnalyzeSED._build_profile_dict(
                        basedir, profile)
                else:
                    skydir_val = skydir_dict[skydir_key]
                    pkey, pdict = AnalyzeSED._build_profile_dict(
                        basedir, profile)
                    pdict['ra'] = skydir_val['ra']
                    pdict['dec'] = skydir_val['dec']
                    pkey += "_%06i" % skydir_key

                outfile = "sed_%s.fits" % pkey
                # test_case need to be a dict with spectrum and morphology
                gta.add_source(pkey, pdict)
                # refit the ROI
                gta.fit()
                # build the SED
                gta.sed(pkey, outfile=outfile, make_plots=args.make_plots)

                # remove the source
                gta.delete_source(pkey)
                # put the ROI back to how it was
                gta.load_xml('fit_baseline')

        return gta


class AnalyzeROI_SG(ScatterGather):
    """Small class to generate configurations for this script

    This adds the following arguments:
    """
    appname = 'fermipy-analyze-roi-sg'
    usage = "%s [options]" % (appname)
    description = "Run analyses on a series of ROIs"
    clientclass = AnalyzeROI

    job_time = 1500

    default_options = dict(ttype=defaults.common['ttype'],
                           targetlist=defaults.common['targetlist'],
                           config=defaults.common['config'],
                           roi_baseline=defaults.common['roi_baseline'],
                           make_plots=defaults.common['make_plots'])

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        ttype = args['ttype']
        (targets_yaml, sim) = NAME_FACTORY.resolve_targetfile(args)
        if sim is not None:
            raise ValueError("Found 'sim' argument on AnalyzeROI_SG config.")
        if targets_yaml is None:
            return job_configs

        config_yaml = 'config.yaml'
        config_override = args.get('config')
        if is_not_null(config_override):
            config_yaml = config_override

        targets = load_yaml(targets_yaml)
        roi_baseline = args['roi_baseline']
        make_plots = args['make_plots']

        for target_name in targets.keys():
            name_keys = dict(target_type=ttype,
                             target_name=target_name,
                             fullpath=True)
            target_dir = NAME_FACTORY.targetdir(**name_keys)
            config_path = os.path.join(target_dir, config_yaml)
            logfile = make_nfs_path(os.path.join(
                target_dir, "%s_%s.log" % (self.linkname, target_name)))
            job_config = dict(config=config_path,
                              roi_baseline=roi_baseline,
                              make_plots=make_plots,
                              logfile=logfile)
            job_configs[target_name] = job_config

        return job_configs


class AnalyzeSED_SG(ScatterGather):
    """Small class to generate configurations for this script

    This adds the following arguments:
    """
    appname = 'fermipy-analyze-sed-sg'
    usage = "%s [options]" % (appname)
    description = "Run analyses on a series of ROIs"
    clientclass = AnalyzeSED

    job_time = 1500

    default_options = dict(ttype=defaults.common['ttype'],
                           targetlist=defaults.common['targetlist'],
                           config=defaults.common['config'],
                           roi_baseline=defaults.common['roi_baseline'],
                           skydirs=defaults.sims['skydirs'],
                           make_plots=defaults.common['make_plots'])

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        ttype = args['ttype']
        (targets_yaml, sim) = NAME_FACTORY.resolve_targetfile(args)
        if sim is not None:
            raise ValueError("Found 'sim' argument on AnalyzeSED_SG config.")
        if targets_yaml is None:
            return job_configs

        targets = load_yaml(targets_yaml)
        config_yaml = 'config.yaml'

        if is_not_null(args['skydirs']):
            skydirs = args['skydirs']
        else:
            skydirs = None

        roi_baseline = args['roi_baseline']
        make_plots = args['make_plots']

        for target_name, target_list in targets.items():
            name_keys = dict(target_type=ttype,
                             target_name=target_name,
                             sim_name='random',
                             fullpath=True)
            if skydirs is None:
                target_dir = NAME_FACTORY.targetdir(**name_keys)
                skydir_path = None
            else:
                target_dir = NAME_FACTORY.sim_targetdir(**name_keys)
                skydir_path = os.path.join(target_dir, skydirs)
            config_path = os.path.join(target_dir, config_yaml)
            logfile = make_nfs_path(os.path.join(
                target_dir, "%s_%s.log" % (self.linkname, target_name)))
            job_config = dict(config=config_path,
                              roi_baseline=roi_baseline,
                              profiles=target_list,
                              skydirs=skydir_path,
                              make_plots=make_plots,
                              logfile=logfile)
            job_configs[target_name] = job_config

        return job_configs


def register_classes():
    """Register these classes with the `LinkFactory` """
    AnalyzeROI.register_class()
    AnalyzeROI_SG.register_class()
    AnalyzeSED.register_class()
    AnalyzeSED_SG.register_class()
