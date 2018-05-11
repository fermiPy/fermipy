#!/usr/bin/env python

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Run gtsrcmaps for a single energy plane for a single source

This is useful to parallize the production of the source maps
"""
from __future__ import absolute_import, division, print_function

import os
import sys
from shutil import copyfile

import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord, ICRS, Galactic
from gammapy.maps import WcsGeom

from fermipy.utils import load_yaml, write_yaml, init_matplotlib_backend

from fermipy import utils
from fermipy.jobs.utils import is_null, is_not_null
from fermipy.jobs.link import Link
from fermipy.jobs.scatter_gather import ScatterGather
from fermipy.jobs.slac_impl import make_nfs_path
from fermipy.jobs.analysis_utils import add_source_get_correlated

from fermipy.jobs.name_policy import NameFactory
from fermipy.jobs import defaults

init_matplotlib_backend('Agg')

try:
    from fermipy.gtanalysis import GTAnalysis
    HAVE_ST = True
except ImportError:
    HAVE_ST = False

NAME_FACTORY = NameFactory(basedir=('.'))


class CopyBaseROI(Link):
    """Small class to copy a baseline ROI to a simulation area

    This is useful for parallelizing analysis using the fermipy.jobs module.
    """
    appname = 'fermipy-copy-base-roi'
    linkname_default = 'copy-base-roi'
    usage = '%s [options]' % (appname)
    description = "Copy a baseline ROI to a simulation area"

    default_options = dict(ttype=defaults.common['ttype'],
                           target=defaults.common['target'],
                           roi_baseline=defaults.common['roi_baseline'],
                           extracopy=defaults.sims['extracopy'],
                           sim=defaults.sims['sim'])

    copyfiles = ['srcmap_00.fits']

    __doc__ += Link.construct_docstring(default_options)

    @classmethod
    def copy_analysis_files(cls, orig_dir, dest_dir, files):
        """ Copy a list of files from orig_dir to dest_dir"""
        for f in files:
            orig_path = os.path.join(orig_dir, f)
            dest_path = os.path.join(dest_dir, f)
            try:
                copyfile(orig_path, dest_path)
            except IOError:
                sys.stderr.write("WARNING: failed to copy %s\n" % orig_path)

    @classmethod
    def copy_target_dir(cls, orig_dir, dest_dir, roi_baseline, extracopy):
        """ Create and populate directoris for target analysis
        """
        try:
            os.makedirs(dest_dir)
        except OSError:
            pass

        copyfiles = ['%s.fits' % roi_baseline,
                     '%s.npy' % roi_baseline,
                     '%s_00.xml' % roi_baseline] + cls.copyfiles
        if isinstance(extracopy, list):
            copyfiles += extracopy

        cls.copy_analysis_files(orig_dir, dest_dir, copyfiles)

    def run_analysis(self, argv):
        """Run this analysis"""
        args = self._parser.parse_args(argv)

        name_keys = dict(target_type=args.ttype,
                         target_name=args.target,
                         sim_name=args.sim,
                         fullpath=True)

        orig_dir = NAME_FACTORY.targetdir(**name_keys)
        dest_dir = NAME_FACTORY.sim_targetdir(**name_keys)
        self.copy_target_dir(orig_dir, dest_dir,
                             args.roi_baseline, args.extracopy)


class CopyBaseROI_SG(ScatterGather):
    """Small class to generate configurations for this script

    This adds the following arguments:
    """
    appname = 'fermipy-copy-base-roi-sg'
    usage = "%s [options]" % (appname)
    description = "Run analyses on a series of ROIs"
    clientclass = CopyBaseROI

    job_time = 60

    default_options = dict(ttype=defaults.common['ttype'],
                           targetlist=defaults.common['targetlist'],
                           roi_baseline=defaults.common['roi_baseline'],
                           sim=defaults.sims['sim'],
                           extracopy=defaults.sims['extracopy'])

    __doc__ += Link.construct_docstring(default_options)

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        ttype = args['ttype']
        (sim_targets_yaml, sim) = NAME_FACTORY.resolve_targetfile(args)
        targets = load_yaml(sim_targets_yaml)

        base_config = dict(ttype=ttype,
                           roi_baseline=args['roi_baseline'],
                           extracopy = args['extracopy'],
                           sim=sim)
                           
        for target_name in targets.keys():
            targetdir = NAME_FACTORY.sim_targetdir(target_type=ttype,
                                                   target_name=target_name,
                                                   sim_name=sim)
            logfile = os.path.join(targetdir, 'copy_base_dir.log')
            job_config = base_config.copy()
            job_config.update(dict(target=target_name,
                                   logfile=logfile))
            job_configs[target_name] = job_config

        return job_configs


class RandomDirGen(Link):
    """Small class to generate random sky directions inside an ROI

    This is useful for parallelizing analysis using the fermipy.jobs module.
    """
    appname = 'fermipy-random-dir-gen'
    linkname_default = 'random-dir-gen'
    usage = '%s [options]' % (appname)
    description = "Generate random sky directions in an ROI"

    default_options = dict(config=defaults.common['config'],
                           rand_config=defaults.sims['rand_config'],
                           outfile=defaults.generic['outfile'])

    __doc__ += Link.construct_docstring(default_options)

    @staticmethod
    def _make_wcsgeom_from_config(config):
        """Build a `WCS.Geom` object from a fermipy coniguration file"""

        binning = config['binning']
        binsz = binning['binsz']
        coordsys = binning.get('coordsys', 'GAL')
        roiwidth = binning['roiwidth']
        proj = binning.get('proj', 'AIT')
        ra = config['selection']['ra']
        dec = config['selection']['dec']
        npix = int(np.round(roiwidth / binsz))
        skydir = SkyCoord(ra * u.deg, dec * u.deg)

        wcsgeom = WcsGeom.create(npix=npix, binsz=binsz,
                                 proj=proj, coordsys=coordsys,
                                 skydir=skydir)
        return wcsgeom

    @staticmethod
    def _build_skydir_dict(wcsgeom, rand_config):
        """Build a dictionary of random directions"""
        step_x = rand_config['step_x']
        step_y = rand_config['step_y']
        max_x = rand_config['max_x']
        max_y = rand_config['max_y']
        seed = rand_config['seed']
        nsims = rand_config['nsims']

        cdelt = wcsgeom.wcs.wcs.cdelt
        pixstep_x = step_x / cdelt[0]
        pixstep_y = -1. * step_y / cdelt[1]
        pixmax_x = max_x / cdelt[0]
        pixmax_y = max_y / cdelt[0]

        nstep_x = int(np.ceil(2. * pixmax_x / pixstep_x)) + 1
        nstep_y = int(np.ceil(2. * pixmax_y / pixstep_y)) + 1

        center = np.array(wcsgeom._center_pix)

        grid = np.meshgrid(np.linspace(-1 * pixmax_x, pixmax_x, nstep_x),
                           np.linspace(-1 * pixmax_y, pixmax_y, nstep_y))
        grid[0] += center[0]
        grid[1] += center[1]

        test_grid = wcsgeom.pix_to_coord(grid)
        glat_vals = test_grid[0].flat
        glon_vals = test_grid[1].flat
        conv_vals = SkyCoord(glat_vals * u.deg, glon_vals *
                             u.deg, frame=Galactic).transform_to(ICRS)

        ra_vals = conv_vals.ra.deg[seed:nsims]
        dec_vals = conv_vals.dec.deg[seed:nsims]

        o_dict = {}
        for i, (ra, dec) in enumerate(zip(ra_vals, dec_vals)):
            key = i + seed
            o_dict[key] = dict(ra=ra, dec=dec)
        return o_dict

    def run_analysis(self, argv):
        """Run this analysis"""
        args = self._parser.parse_args(argv)

        if is_null(args.config):
            raise ValueError("Config yaml file must be specified")
        if is_null(args.rand_config):
            raise ValueError(
                "Random direction config yaml file must be specified")
        config = load_yaml(args.config)
        rand_config = load_yaml(args.rand_config)

        wcsgeom = self._make_wcsgeom_from_config(config)
        dir_dict = self._build_skydir_dict(wcsgeom, rand_config)

        if is_not_null(args.outfile):
            write_yaml(dir_dict, args.outfile)


class SimulateROI(Link):
    """Small class wrap an analysis script.

    This is useful for parallelizing analysis using the fermipy.jobs module.
    """
    appname = 'fermipy-simulate-roi'
    linkname_default = 'simulate-roi'
    usage = '%s [options]' % (appname)
    description = "Run simulated analysis of a single ROI"

    default_options = dict(config=defaults.common['config'],
                           roi_baseline=defaults.common['roi_baseline'],
                           profiles=defaults.common['profiles'],
                           sim_profile=defaults.sims['sim_profile'],
                           sim=defaults.sims['sim'],
                           nsims=defaults.sims['nsims'],
                           seed=defaults.sims['seed'])

    __doc__ += Link.construct_docstring(default_options)

    @staticmethod
    def _run_simulation(gta, roi_baseline,
                        injected_name, test_sources, seed):
        """Simulate a realization of this analysis"""
        gta.load_roi('sim_baseline')
        gta.set_random_seed(seed)
        gta.simulate_roi()
        if injected_name:
            gta.zero_source(injected_name)
        
        gta.optimize()
        gta.find_sources(sqrt_ts_threshold=5.0, search_skydir=gta.roi.skydir,
                         search_minmax_radius=[1.0, np.nan])
        gta.optimize()
        gta.free_sources(skydir=gta.roi.skydir, distance=1.0, pars='norm')
        gta.fit(covar=True)
        gta.write_roi('sim_refit')

        for test_source in test_sources:
            test_source_name = test_source['name']
            sedfile = "sed_%s_%06i.fits" % (test_source_name, seed)
            correl_list = add_source_get_correlated(gta, test_source_name,
                                                    test_source['source_model'],
                                                    correl_thresh=0.25)
            gta.free_sources(False)
            for src_name in correl_list:
                gta.free_source(src_name, pars='norm')

            gta.sed(test_source_name, outfile=sedfile)
            # Set things back to how they were
            gta.delete_source(test_source_name)
            gta.load_xml('sim_refit')

    def run_analysis(self, argv):
        """Run this analysis"""
        args = self._parser.parse_args(argv)

        if not HAVE_ST:
            raise RuntimeError(
                "Trying to run fermipy analysis, but don't have ST")

        gta = GTAnalysis(args.config, logging={'verbosity': 3},
                         fileio={'workdir_regex': '\.xml$|\.npy$'})
        gta.load_roi(args.roi_baseline)

        workdir = os.path.dirname(args.config)
        simfile = os.path.join(workdir, 'sim_%s_%s.yaml' %
                               (args.sim, args.sim_profile))

        mcube_file = "%s_%s" % (args.sim, args.sim_profile)
        sim_config = utils.load_yaml(simfile)

        injected_source = sim_config.get('injected_source', None)
        if injected_source is not None:
            src_dict =  injected_source['source_model']
            src_dict['ra'] = gta.config['selection']['ra']
            src_dict['dec'] = gta.config['selection']['dec']
            injected_name = injected_source['name']
            gta.add_source(injected_name, src_dict)
            gta.write_model_map(mcube_file)
            mc_spec_dict = dict(true_counts=gta.model_counts_spectrum(injected_name),
                                energies=gta.energies,
                                model=src_dict)
            mcspec_file = os.path.join(gta.workdir,
                                       "mcspec_%s.yaml" % mcube_file)
            utils.write_yaml(mc_spec_dict, mcspec_file)
        else:
            injected_name = None

        gta.write_roi('sim_baseline')

        test_sources = []
        for profile in args.profiles:
            profile_path = os.path.join(workdir, 'profile_%s.yaml' % profile)
            test_source = load_yaml(profile_path)
            test_sources.append(test_source)
            first = args.seed
            last = first + args.nsims
            for seed in range(first, last):
                self._run_simulation(gta, args.roi_baseline,
                                     injected_name, test_sources, seed)


class RandomDirGen_SG(ScatterGather):
    """Small class to generate configurations for this script

    This adds the following arguments:
    """
    appname = 'fermipy-random-dir-gen-sg'
    usage = "%s [options]" % (appname)
    description = "Run analyses on a series of ROIs"
    clientclass = RandomDirGen

    job_time = 60

    default_options = dict(ttype=defaults.common['ttype'],
                           targetlist=defaults.common['targetlist'],
                           config=defaults.common['config'],
                           rand_config=defaults.sims['rand_config'],
                           sim=defaults.sims['sim'])

    __doc__ += Link.construct_docstring(default_options)

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        ttype = args['ttype']
        (targets_yaml, sim) = NAME_FACTORY.resolve_targetfile(args)
        if targets_yaml is None:
            return job_configs

        config_yaml = 'config.yaml'
        config_override = args.get('config')
        if is_not_null(config_override):
            config_yaml = config_override

        rand_yaml = NAME_FACTORY.resolve_randconfig(args)

        targets = load_yaml(targets_yaml)

        base_config = dict(rand_config=rand_yaml)

        for target_name in targets.keys():
            name_keys = dict(target_type=ttype,
                             target_name=target_name,
                             sim_name=sim,
                             fullpath=True)
            simdir = NAME_FACTORY.sim_targetdir(**name_keys)
            config_path = os.path.join(simdir, config_yaml)
            outfile = os.path.join(simdir, 'skydirs.yaml')
            logfile = make_nfs_path(outfile.replace('yaml', 'log'))
            job_config = base_config.copy()
            job_config.update(dict(config=config_path,
                                   outfile=outfile,
                                   logfile=logfile))
            job_configs[target_name] = job_config

        return job_configs


class SimulateROI_SG(ScatterGather):
    """Small class to generate configurations for this script

    This adds the following arguments:
    """
    appname = 'fermipy-simulate-roi-sg'
    usage = "%s [options]" % (appname)
    description = "Run analyses on a series of ROIs"
    clientclass = SimulateROI

    job_time = 1500

    default_options = dict(ttype=defaults.common['ttype'],
                           targetlist=defaults.common['targetlist'],
                           config=defaults.common['config'],
                           roi_baseline=defaults.common['roi_baseline'],
                           sim=defaults.sims['sim'],
                           sim_profile=defaults.sims['sim_profile'],
                           nsims=defaults.sims['nsims'],
                           seed=defaults.sims['seed'])

    __doc__ += Link.construct_docstring(default_options)

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        ttype = args['ttype']
        (targets_yaml, sim) = NAME_FACTORY.resolve_targetfile(args)
        if targets_yaml is None:
            return job_configs

        config_yaml = 'config.yaml'
        config_override = args.get('config')
        if is_not_null(config_override):
            config_yaml = config_override

        targets = load_yaml(targets_yaml)

        base_config = dict(sim_profile=args['sim_profile'],
                           roi_baseline=args['roi_baseline'],
                           sim=sim,
                           nsims=args['nsims'],
                           seed=args['seed'])

        for target_name, target_list in targets.items():
            name_keys = dict(target_type=ttype,
                             target_name=target_name,
                             sim_name=sim,
                             fullpath=True)
            simdir = NAME_FACTORY.sim_targetdir(**name_keys)
            config_path = os.path.join(simdir, config_yaml)
            logfile = make_nfs_path(os.path.join(
                simdir, "%s_%s.log" % (self.linkname, target_name)))
            job_config = base_config.copy()
            job_config.update(dict(config=config_path,
                                   logfile=logfile,
                                   profiles=target_list))
            job_configs[target_name] = job_config

        return job_configs


def register_classes():
    """Register these classes with the `LinkFactory` """
    CopyBaseROI.register_class()
    CopyBaseROI_SG.register_class()
    SimulateROI.register_class()
    SimulateROI_SG.register_class()
    RandomDirGen.register_class()
    RandomDirGen_SG.register_class()
