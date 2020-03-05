#!/usr/bin/env python

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Scripts to run the all-sky diffuse analysis
"""
from __future__ import absolute_import, division, print_function

from fermipy.utils import load_yaml

from fermipy.jobs.job_archive import JobStatus, JobDetails
from fermipy.jobs.link import Link
from fermipy.jobs.chain import Chain, purge_dict
from fermipy.jobs.target_analysis import AnalyzeROI_SG, AnalyzeSED_SG
from fermipy.jobs.target_collect import CollectSED_SG
from fermipy.jobs.target_sim import CopyBaseROI_SG, SimulateROI_SG,\
    RandomDirGen_SG
from fermipy.jobs.target_plotting import PlotCastro_SG


from fermipy.jobs import defaults

def _get_plot_config(plotting_dict, key):
    """ Get the configuration for a type of plot """
    sub_dict = plotting_dict.get(key, None)
    if sub_dict is None:
        return None
    return purge_dict(sub_dict)



class PipelineData(Chain):
    """Chain together the steps of the dSphs pipeline

    This chain consists of:

    analyze-roi : `AnalyzeROI_SG`
        Do the baseline analysis for each target in the target list.

    analyze-sed : `AnalyzeSED_SG`
        Extract the SED for each profile for each target in the target list.

    Optional plotting modules includde

    plot-castro : `PlotCastro`
        Make 'Castro' plots of the SEDs for each profile for each target
        in the target list.
    """

    appname = 'fermipy-pipeline-data'
    linkname_default = 'pipeline-data'
    usage = '%s [options]' % (appname)
    description = 'Data analysis pipeline'

    default_options = dict(config=defaults.common['config'],
                           sim=defaults.sims['sim'],
                           dry_run=defaults.common['dry_run'])

    __doc__ += Link.construct_docstring(default_options)

    def _map_arguments(self, args):
        """Map from the top-level arguments to the arguments provided to
        the indiviudal links """
        config_yaml = args['config']
        config_dict = load_yaml(config_yaml)
        ttype = config_dict.get('ttype')
        config_localpath = config_dict.get('config_localpath', None)
        specfile = config_dict.get('specfile')
        targetlist = config_dict.get('targetlist')
        data_plotting = config_dict.get('data_plotting')

        self._set_link('analyze-roi',
                       AnalyzeROI_SG,
                       ttype=ttype,
                       targetlist=targetlist,
                       config=config_localpath)
        self._set_link('analyze-sed',
                       AnalyzeSED_SG,
                       ttype=ttype,
                       targetlist=targetlist,
                       config=config_localpath)

        config_plot_castro = _get_plot_config(data_plotting, 'plot-castro')
        if config_plot_castro is not None:
            self._set_link('plot-castro-sg',
                           PlotCastro_SG,
                           ttype=ttype,
                           targetlist=targetlist,
                           **config_plot_castro)



class PipelineSim(Chain):
    """Chain together the steps of the dSphs pipeline for simulations

    This chain consists of:

    copy-base-roi : `CopyBaseROI_SG`
        Copy the baseline analysis directory files for each target.

    simulate-roi : `AnalyzeROI_SG`
        Simulate the SED analysis for each target and profile in the target list.

    collect-sed : `CollectSED_SG`
        Collect and summarize the SED results for all the simulations.

    """
    appname = 'fermipy-pipeline-sim'
    linkname_default = 'pipeline-sim'
    usage = '%s [options]' % (appname)
    description = 'Do stuff'

    default_options = dict(config=defaults.common['config'],
                           sim=defaults.sims['sim'],
                           dry_run=defaults.common['dry_run'])

    __doc__ += Link.construct_docstring(default_options)

    def _map_arguments(self, args):
        """Map from the top-level arguments to the arguments provided to
        the indiviudal links """
        config_yaml = args['config']
        config_dict = load_yaml(config_yaml)

        sim_name = args['sim']
        sim_dict = config_dict['sims'][sim_name]

        ttype = config_dict.get('ttype')
        config_template = config_dict.get('config_template', None)
        config_localpath = config_dict.get('config_localpath', None)
        specfile = config_dict.get('specfile')
        targetlist = config_dict.get('targetlist')

        sim_values = config_dict['sim_defaults']
        sim_values.update(sim_dict)

        sim_profile = sim_values['profile']
        seed = sim_values.get('seed', 0)
        nsims = sim_values.get('nsims', 20)
        nsims_job = sim_values.get('nsims_job', 0)
        non_null_src = sim_values.get('non_null_src', False)
        do_find_src = sim_values.get('do_find_src', False)

        sim_plotting = config_dict.get('sim_plotting')
        plot_channels_default = config_dict.get('plot_channels', [])

        self._set_link('copy-base-roi',
                       CopyBaseROI_SG,
                       ttype=ttype,
                       targetlist=targetlist,
                       rosterlist=rosterlist,
                       sim=sim_name,
                       config=config_template)
        self._set_link('simulate-roi',
                       SimulateROI_SG,
                       ttype=ttype,
                       sim=sim_name,
                       sim_profile=sim_profile,
                       targetlist=targetlist,
                       config=config_localpath,
                       seed=seed, nsims=nsims,
                       non_null_src=non_null_src,
                       do_find_src=do_find_src,
                       nsims_job=nsims_job)
        self._set_link('collect-sed',
                       CollectSED_SG,
                       ttype=ttype,
                       sim=sim_name,
                       config=config_localpath,
                       targetlist=targetlist,
                       seed=seed, nsims=nsims)


class PipelineRandom(Chain):
    """Chain together the steps of the dSphs pipeline for random
    direction studies.

    This chain consists of:

    copy-base-roi : `CopyBaseROI_SG`
        Copy the baseline analysis directory files for each target.

    random-dir-gen : `RandomDirGen_SG`
        Select random directions inside the ROI and generate approriate target files.

    analyze-sed : `AnalyzeSED_SG`
        Extract the SED for each profile for each target in the target list.

    collect-sed : `CollectSED_SG`
        Collect and summarize the SED results for all the simulations.


    """
    appname = 'dmpipe-pipeline-random'
    linkname_default = 'pipeline-random'
    usage = '%s [options]' % (appname)
    description = 'Data analysis pipeline for random directions'

    default_options = dict(config=defaults.common['config'],
                           dry_run=defaults.common['dry_run'])

    __doc__ += Link.construct_docstring(default_options)

    def _map_arguments(self, args):
        """Map from the top-level arguments to the arguments provided to
        the indiviudal links """
        config_yaml = args['config']
        config_dict = load_yaml(config_yaml)

        ttype = config_dict.get('ttype')
        config_template = config_dict.get('config_template', None)
        config_localpath = config_dict.get('config_localpath', None)
        specfile = config_dict.get('specfile')
        targetlist = config_dict.get('targetlist')
        random = config_dict.get('random')

        rand_dirs = random.get('rand_dirs')
        sim_values = config_dict['sim_defaults']
        seed = sim_values.get('seed', 0)
        nsims = sim_values.get('nsims', 20)

        rand_plotting = config_dict.get('rand_plotting')
        plot_channels_default = config_dict.get('plot_channels', [])

        self._set_link('copy-base-roi',
                       CopyBaseROI_SG,
                       ttype=ttype,
                       targetlist=targetlist,
                       rosterlist=rosterlist,
                       sim='random',
                       config=config_template)
        self._set_link('random-dir-gen',
                       RandomDirGen_SG,
                       ttype=ttype,
                       rand_config=rand_dirs,
                       targetlist=targetlist,
                       sim='random',
                       config=config_localpath)
        self._set_link('analyze-sed',
                       AnalyzeSED_SG,
                       ttype=ttype,
                       sim='random',
                       skydirs='skydirs.yaml',
                       targetlist=targetlist,
                       config=config_localpath,
                       seed=seed, nsims=nsims)
        self._set_link('collect-sed',
                       CollectSED_SG,
                       ttype=ttype,
                       sim='random',
                       config=config_localpath,
                       targetlist=targetlist,
                       seed=seed, nsims=nsims)


class Pipeline(Chain):
    """Top level DM pipeline analysis chain.

    This chain consists of:

    prepare-targets : `PrepareTargets`
        Make the input files need for all the targets in the analysis.

    spec-table : `SpecTable`
        Build the FITS table with the DM spectra for all the channels
        being analyzed.

    data : `PipelineData`
        Data analysis pipeline

    sim_{sim_name} : `PipelineSim`
        Simulation pipeline for each simulation scenario

    random : `PipelineRandom`
        Analysis pipeline for random direction control studies

    final-plots : `PlotFinalLimits_SG`
        Make the final analysis results plots

    """
    appname = 'fermipy-pipeline'
    linkname_default = 'pipeline'
    usage = '%s [options]' % (appname)
    description = 'Data analysis pipeline'

    default_options = dict(config=defaults.common['config'],
                           dry_run=defaults.common['dry_run'])

    def __init__(self, **kwargs):
        """C'tor
        """
        super(Pipeline, self).__init__(**kwargs)
        self._preconfigured = False


    def preconfigure(self, config_yaml):
        """ Run any links needed to build files
        that are used in _map_arguments """
        if self._preconfigured:
            return
        config_dict = load_yaml(config_yaml)
        ttype = config_dict.get('ttype')
        self.link_prefix = "%s." % ttype
        config_template = config_dict.get('config_template', None)
        rosters = config_dict.get('rosters')
        alias_dict = config_dict.get('alias_dict', None)
        spatial_models = config_dict.get('spatial_models')
        sims = config_dict.get('sims', {})
        sim_names = []
        sim_names += sims.keys()
        if 'random' in config_dict:
            sim_names += ['random']

        self._set_link('prepare-targets',
                       PrepareTargets,
                       ttype=ttype,
                       rosters=rosters,
                       spatial_models=spatial_models,
                       alias_dict=alias_dict,
                       sims=sim_names,
                       config=config_template)
        link = self['prepare-targets']

        key = JobDetails.make_fullkey(link.full_linkname)
        if not link.jobs:
            raise ValueError("No Jobs")
        link_status = link.check_job_status(key)
        if link_status == JobStatus.done:
            self._preconfigured = True
            return
        elif link_status == JobStatus.failed:
            link.clean_jobs()
        link.run_with_log()
        self._preconfigured = True

    def _map_arguments(self, args):
        """Map from the top-level arguments to the arguments provided to
        the indiviudal links """

        config_yaml = args['config']
        config_dict = load_yaml(config_yaml)
        ttype = config_dict.get('ttype')
        config_template = config_dict.get('config_template', None)
        rosters = config_dict.get('rosters')
        rosterlist = config_dict.get('rosterlist')
        spatial_models = config_dict.get('spatial_models')
        specfile = config_dict.get('specfile')
        sims = config_dict.get('sims', {})
        sim_names = []
        sim_names += sims.keys()
        if 'random' in config_dict:
            sim_names += ['random']

        plot_channels = config_dict.get('plot_channels', [])

        dry_run = args.get('dry_run', False)

        self._set_link('prepare-targets',
                       PrepareTargets,
                       ttype=ttype,
                       rosters=rosters,
                       spatial_models=spatial_models,
                       sims=sim_names,
                       config=config_template)

        self._set_link('data',
                       PipelineData,
                       link_prefix='data.',
                       config=config_yaml,
                       dry_run=dry_run)

        final_plot_sims = []

        for sim in sims.keys():
            if sim in ['null']:
                final_plot_sims.append(sim)
            linkname = 'sim_%s' % sim
            self._set_link(linkname,
                           PipelineSim,
                           link_prefix='%s.' % linkname,
                           config=config_yaml,
                           sim=sim,
                           dry_run=dry_run)

        if 'random' in config_dict:
            final_plot_sims.append('random')
            self._set_link('random',
                           PipelineRandom,
                           link_prefix='random.',
                           config=config_yaml,
                           dry_run=dry_run)

