#!/usr/bin/env python

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Module with classes for simple target analysis and to 
paralleize those analyses.
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import numpy as np

from fermipy.utils import load_yaml, write_yaml, init_matplotlib_backend

from fermipy.jobs.utils import is_null, is_not_null
from fermipy.jobs.link import Link
from fermipy.jobs.scatter_gather import ScatterGather
from fermipy.jobs.slac_impl import make_nfs_path
from fermipy.jobs.analysis_utils import baseline_roi_fit, localize_sources,\
    add_source_get_correlated

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
    """Small class that wraps an analysis script.

    This particular script does baseline fitting of an ROI.
    """
    appname = 'fermipy-analyze-roi'
    linkname_default = 'analyze-roi'
    usage = '%s [options]' % (appname)
    description = "Run analysis of a single ROI"

    default_options = dict(config=defaults.common['config'],
                           roi_baseline=defaults.common['roi_baseline'],
                           make_plots=defaults.common['make_plots'])

    __doc__ += Link.construct_docstring(default_options)

    def run_analysis(self, argv):
        """Run this analysis"""
        args = self._parser.parse_args(argv)

        if not HAVE_ST:
            raise RuntimeError(
                "Trying to run fermipy analysis, but don't have ST")

        gta = GTAnalysis(args.config, logging={'verbosity': 3},
                         fileio={'workdir_regex': '\.xml$|\.npy$'})

        gta.setup(overwrite=False)

        #baseline_roi_fit(gta, make_plots=args.make_plots,
        #                 minmax_npred=[1e3, np.inf])

        gta.optimize()
	gta.print_model()
        gta.delete_sources(minmax_ts=[None,25.],exclude=['isodiff','galdiff',gta.roi.sources[0].name,'M31_IR'])
        #gta.delete_sources(minmax_ts=[None,25.],exclude=['isodiff','galdiff',gta.roi.sources[0].name])
        
        localize_sources(gta, nstep=5, dtheta_max=0.5, update=True,
                         prefix='base', make_plots=args.make_plots)
        
        #gta.find_sources(sqrt_ts_threshold=5.0, search_skydir=gta.roi.skydir,
        #                 search_minmax_radius=[0.50, np.nan])

	gta.find_sources(sqrt_ts_threshold=5.0,min_separation=0.3,tsmap_fitter='tsmap')
	gta.free_sources()
	gta.print_model()

        target = 'M31'
	#extension = 'pointlike'
        extension = 'noastro'
	#extension = 'best'
        #extension = 'lower'
	#extension = 'upper'
        #target = 'dSphs'
        #target = 'LMC'

        if target=='M31' or target=='M33':
	    if extension == 'pointlike':
                #gta.delete_source(gta.roi.sources[0].name)
                #gta.delete_source('FL8Y J0043.2+4114')
                gta.print_model()

	    if extension == 'noastro':
		#gta.delete_source(gta.roi.sources[0].name)
		gta.delete_source('FL8Y J0043.2+4114')
		gta.print_model()
		
	    if extension != 'noastro' and extension != 'pointlike':
            	gta.free_sources(skydir=gta.roi[gta.roi.sources[0].name].skydir,distance=[3.0],free=True)
            	gta.print_model()
            	gta.localize(gta.roi.sources[0].name, update=True,make_plots=True,dtheta_max=0.30)

            	gta.write_roi('roi_pointlike')
            	ext_disk = gta.extension(gta.roi.sources[0].name, update=True, sqrt_ts_threshold=3., 
                          width_max=3.,fit_position=True,spatial_model='RadialDisk')
            	TSext_disk = ext_disk['ts_ext']
            	size_disk = ext_disk['ext'] 
            	size_disk_hi = ext_disk['ext_err_hi'] 
            	size_disk_lo = ext_disk['ext_err_lo'] 
            	print('Extension Disk TSext=%.3f and extension=%.3f, error=(%.3f,%.3f)'%(TSext_disk,size_disk,size_disk_lo,size_disk_hi))

            	gta.load_roi('roi_pointlike')
            	ext_gauss = gta.extension(gta.roi.sources[0].name, update=True, sqrt_ts_threshold=3.,
                          width_max=3.,fit_position=True,spatial_model='RadialGaussian')
            	TSext_gauss = ext_gauss['ts_ext']
            	size_gauss = ext_gauss['ext']
            	size_gauss_hi = ext_gauss['ext_err_hi'] 
            	size_gauss_lo = ext_gauss['ext_err_lo']
            	print('Extension Gauss TSext=%.3f and extension=%.3f, error=(%.3f,%.3f)'%(TSext_gauss,size_gauss,size_gauss_lo,size_gauss_hi))

            	namesource = gta.roi.sources[0].name
            	glon0 = gta.config['selection']['glon']
            	glat0 = gta.config['selection']['glat']
            	gta.delete_source(gta.roi.sources[0].name)

            if extension == 'best':
                if TSext_gauss>TSext_disk and TSext_gauss>=9.0:
                    gta.add_source(namesource,{ 'glon' : glon0, 'glat' : glat0,
                    'SpectrumType' : 'PowerLaw', 'Index' : 2.0,
                    'Scale' : 1000, 'Prefactor' : 1e-11,
                    'SpatialModel': 'RadialGaussian', 'SpatialWidth': size_gauss })
                elif TSext_gauss<=TSext_disk and TSext_gauss>=9.0:
                    gta.add_source(namesource,{ 'glon' : glon0, 'glat' : glat0,
                    'SpectrumType' : 'PowerLaw', 'Index' : 2.0,
                    'Scale' : 1000, 'Prefactor' : 1e-11,
                    'SpatialModel': 'RadialDisk', 'SpatialWidth': size_disk })		
                else:
                    gta.load_roi('roi_pointlike')
		print('Spatial Template: ',gta.roi.sources[0]['SpatialType'],gta.roi.sources[0]['SpatialWidth'])		

            if extension == 'lower' or extension == 'upper':
                if TSext_gauss>TSext_disk and TSext_gauss>=9.0:
                    if extension == 'upper':
                        gta.add_source(namesource,{ 'glon' : glon0, 'glat' : glat0,
                        'SpectrumType' : 'PowerLaw', 'Index' : 2.0,
                        'Scale' : 1000, 'Prefactor' : 1e-11,
                        'SpatialModel': 'RadialGaussian', 'SpatialWidth': size_gauss+size_gauss_hi })
                    if extension == 'lower':
                        gta.add_source(namesource,{ 'glon' : glon0, 'glat' : glat0,
                        'SpectrumType' : 'PowerLaw', 'Index' : 2.0,
                        'Scale' : 1000, 'Prefactor' : 1e-11,
                        'SpatialModel': 'RadialGaussian', 'SpatialWidth': size_gauss-1.*size_gauss_lo })
                elif TSext_gauss<=TSext_disk and TSext_gauss>=9.0:
                    if extension == 'upper':
                        gta.add_source(namesource,{ 'glon' : glon0, 'glat' : glat0,
                        'SpectrumType' : 'PowerLaw', 'Index' : 2.0,
                        'Scale' : 1000, 'Prefactor' : 1e-11,
                        'SpatialModel': 'RadialDisk', 'SpatialWidth': size_disk+size_disk_hi })
                    if extension == 'lower':
                        gta.add_source(namesource,{ 'glon' : glon0, 'glat' : glat0,
                        'SpectrumType' : 'PowerLaw', 'Index' : 2.0,
                        'Scale' : 1000, 'Prefactor' : 1e-11,
                        'SpatialModel': 'RadialDisk', 'SpatialWidth': size_disk-1.*size_disk_lo })	
                else:
                    gta.load_roi('roi_pointlike')
		print('Spatial Template: ',gta.roi.sources[0]['SpatialType'],gta.roi.sources[0]['SpatialWidth'])

	    print('Spatial Template: ',gta.roi.sources[0]['SpatialType'],gta.roi.sources[0]['SpatialWidth'])	    
            gta.print_model()
            gta.free_sources()
            gta.optimize()
            gta.fit(covar=True)
            gta.print_model()
            gta.print_model()
	
        else:
            gta.optimize()
            gta.print_roi()
            gta.print_params()

            gta.free_sources()
            gta.fit(covar=True)
            gta.print_roi()
            gta.print_params()

        #gta.write_roi(args.roi_baseline, make_plots=args.make_plots)
	gta.write_roi(args.roi_baseline, make_plots=True)

        if target=='dSphs':
            namesource = 'dSphs'
            glon0 = gta.config['selection']['glon']
            glat0 = gta.config['selection']['glat']
            gta.add_source(namesource,{ 'glon' : glon0, 'glat' : glat0,
                 'SpectrumType' : 'PowerLaw', 'Index' : 2.0,
                 'Scale' : 1000, 'Prefactor' : 1e-11,
                 'SpatialModel': 'PointSource'})
            gta.free_sources()
            gta.optimize()
            gta.fit()
            gta.print_model()
            gta.print_model()
            gta.write_roi('dSphs_fit',make_plots=True,save_model_map=True)

        gta.tsmap(prefix='TSmap_final',make_plots=True,write_fits=True,write_npy=True)
        gta.delete_source(gta.roi.sources[0].name)
        gta.tsmap(prefix='TSmap_final_nosource',make_plots=True,write_fits=True,write_npy=True)


class AnalyzeSED(Link):
    """Small class to wrap an analysis script.

    This particular script fits an SED for a target source
    with respect to the baseline ROI model.
    """
    appname = 'fermipy-analyze-sed'
    linkname_default = 'analyze-sed'
    usage = '%s [options]' % (appname)
    description = "Extract the SED for a single target"

    default_options = dict(config=defaults.common['config'],
                           roi_baseline=defaults.common['roi_baseline'],
                           skydirs=defaults.sims['skydirs'],
                           profiles=defaults.common['profiles'],
                           make_plots=defaults.common['make_plots'],
                           astro_bkgs=(None, "Astrophysical background sources", list))


    __doc__ += Link.construct_docstring(default_options)

    @staticmethod
    def _build_profile_dict(basedir, profile_name):
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
            Name of source to use for this particular profile

        profile_dict : dict
            Dictionary with the source parameters

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
        #gta.setup(overwrite=False)
        gta.load_roi(args.roi_baseline)
        gta.print_roi()

        basedir = os.path.dirname(args.config)
        # This should be a no-op, b/c it was done in the baseline analysis

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

                # Add the source and get the list of correlated soruces
                correl_dict = add_source_get_correlated(gta, pkey,
                                                        pdict, correl_thresh=0.25)
                
                # Write the list of correlated sources
                correl_yaml = os.path.join(basedir, "correl_%s.yaml" % pkey)
                write_yaml(correl_dict, correl_yaml)

                gta.free_sources(False)
                for src_name in correl_dict.keys():
                    gta.free_source(src_name, pars='norm')

                # build the SED
                gta.sed(pkey, outfile=outfile, make_plots=args.make_plots)

                # remove the source
                gta.delete_source(pkey)
                # put the ROI back to how it was
                gta.load_xml(args.roi_baseline)

        return gta


class AnalyzeROI_SG(ScatterGather):
    """Small class to generate configurations for the `AnalyzeROI` class.

    This loops over all the targets defined in the target list.
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

    __doc__ += Link.construct_docstring(default_options)

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
        base_config = dict(roi_baseline=args['roi_baseline'],
                           make_plots=args['make_plots'])

        for target_name in targets.keys():
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


class AnalyzeSED_SG(ScatterGather):
    """Small class to generate configurations for this script

    This loops over all the targets defined in the target list,
    and over all the profiles defined for each target.
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

    __doc__ += Link.construct_docstring(default_options)

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

        base_config = dict(roi_baseline=args['roi_baseline'],
                           make_plots=args['make_plots'])

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
            job_config = base_config.copy()
            job_config.update(dict(config=config_path,
                                   profiles=target_list,
                                   skydirs=skydir_path,
                                   logfile=logfile))
            job_configs[target_name] = job_config

        return job_configs


def register_classes():
    """Register these classes with the `LinkFactory` """
    AnalyzeROI.register_class()
    AnalyzeROI_SG.register_class()
    AnalyzeSED.register_class()
    AnalyzeSED_SG.register_class()
