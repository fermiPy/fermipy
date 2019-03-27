#!/usr/bin/env python
#

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Top level scripts to make castro plot and limits plots in index / norm space
"""
from __future__ import absolute_import, division, print_function

import os
from os.path import splitext
import numpy as np

from astropy.table import Table

from fermipy.utils import init_matplotlib_backend, load_yaml
from fermipy.jobs.utils import is_not_null
from fermipy.jobs.link import Link
from fermipy.jobs.scatter_gather import ScatterGather
from fermipy.jobs.slac_impl import make_nfs_path

from .stack_castro import StackCastroData
from .stack_spec_table import StackSpecTable
from .stack_plotting_utils import plot_stack_castro
from .stack_plotting_utils import plot_stack_spectra_by_index, plot_stack_spectra_by_spec
from .stack_plotting_utils import plot_limits_from_arrays, plot_mc_truth

from .name_policy import NameFactory
from . import defaults

init_matplotlib_backend()
NAME_FACTORY = NameFactory(basedir='.')



def get_ul_bands(table, prefix):
    """ Get the upper limit bands a table

    Parameters
    ----------

    table : `astropy.table.Table`
        Table to get the limits from.

    prefix : str
        Prefix to append to the column names for the limits


    Returns
    -------

    output : dict
        A dictionary with the limits bands

    """
    o = dict(q02=np.squeeze(table["%s_q02" % prefix]),
             q16=np.squeeze(table["%s_q16" % prefix]),
             q84=np.squeeze(table["%s_q84" % prefix]),
             q97=np.squeeze(table["%s_q97" % prefix]),
             median=np.squeeze(table["%s_median" % prefix]))
    return o


class PlotStackSpectra(Link):
    """Small class to plot the stacked spectra from pre-computed tables.

    """
    appname = 'stack-plot-stack-spectra'
    linkname_default = 'plot-stack-spectra'
    usage = '%s [options]' % (appname)
    description = "Plot the stacked spectra stored in pre-computed tables"

    default_options = dict(infile=defaults.generic['infile'],
                           outfile=defaults.generic['outfile'],
                           spec=defaults.common['spec'],
                           index=defaults.common['index'],
                           spec_type=defaults.common['spec_type'])

    __doc__ += Link.construct_docstring(default_options)

    def run_analysis(self, argv):
        """Run this analysis"""
        args = self._parser.parse_args(argv)

        stack_spec_table = StackSpecTable.create_from_fits(args.infile)
        stack_plot_by_index = plot_stack_spectra_by_index(
            stack_spec_table, spec=args.spec, spec_type=args.spec_type)
        stack_plot_by_spec = plot_stack_spectra_by_spec(
            stack_spec_table, index=args.index, spec_type=args.spec_type)

        if args.outfile:
            stack_plot_by_index[0].savefig(
                args.outfile.replace(
                    '.png', '_%s.png' %
                    args.spec))
            stack_plot_by_spec[0].savefig(
                args.outfile.replace(
                    '.png', '_%1.FGeV.png' %
                    args.index))


class PlotLimits(Link):
    """Small class to Plot Stack limits on norm v. index

    """
    appname = 'stack-plot-limits'
    linkname_default = 'plot-limits'
    usage = '%s [options]' % (appname)
    description = "Plot stacked limits on norm v. index"

    default_options = dict(infile=defaults.generic['infile'],
                           outfile=defaults.generic['outfile'],
                           spec=defaults.common['spec'],
                           bands=defaults.collect['bands'],
                           sim=defaults.sims['sim'])

    __doc__ += Link.construct_docstring(default_options)

    def run_analysis(self, argv):
        """Run this analysis"""
        args = self._parser.parse_args(argv)

        limit_col = 'ul_0.95'
        ylims = (1.0e-2, 1.0e2)

        if is_not_null(args.infile):
            tab_m = Table.read(args.infile, hdu="indices")
            tab_s = Table.read(args.infile, hdu=args.spec)
            xvals = tab_m['indices'][0]
            yvals = tab_s[limit_col][0]
            ldict = dict(limits=(xvals, yvals))
        else:
            ldict = {}

        if is_not_null(args.bands):
            tab_b = Table.read(args.bands, hdu=args.spec)
            tab_bm = Table.read(args.bands, hdu="indices")
            bands = get_ul_bands(tab_b, limit_col)
            bands['indices'] = tab_bm['indices'][0]
        else:
            bands = None

        if is_not_null(args.sim):
            sim_srcs = load_yaml(args.sim)
            injected_src = sim_srcs.get('injected_source', None)
        else:
            injected_src = None

        xlims = (1e1, 1e4)

        stack_plot = plot_limits_from_arrays(ldict, xlims, ylims, bands)

        if injected_src is not None:
            mc_model = injected_src['source_model']
            plot_mc_truth(stack_plot[1], mc_model)

        if args.outfile:
            stack_plot[0].savefig(args.outfile)
            return None
        return stack_plot


class PlotMLEs(Link):
    """Small class to Plot maximum likelihood estimate norm v. index

    """
    appname = 'stack-plot-mles'
    linkname_default = 'plot-mles'
    usage = '%s [options]' % (appname)
    description = "Plot maximum likelihood estimate on norm v. index"

    default_options = dict(infile=defaults.generic['infile'],
                           outfile=defaults.generic['outfile'],
                           spec=defaults.common['spec'],
                           bands=defaults.collect['bands'],
                           sim=defaults.sims['sim'])

    __doc__ += Link.construct_docstring(default_options)

    def run_analysis(self, argv):
        """Run this analysis"""
        args = self._parser.parse_args(argv)

        limit_col = 'ul_0.95'
        ylims = (1e-2, 1e+2)

        if is_not_null(args.infile):
            tab_m = Table.read(args.infile, hdu="indices")
            tab_s = Table.read(args.infile, hdu=args.spec)
            xvals = tab_m['indices'][0]
            yvals = tab_s[limit_col][0]
            ldict = dict(limits=(xvals, yvals))
        else:
            ldict = {}

        if is_not_null(args.bands):
            tab_b = Table.read(args.bands, hdu=args.spec)
            tab_bm = Table.read(args.bands, hdu="indices")
            bands = get_ul_bands(tab_b, 'mles')
            bands['indices'] = tab_bm['indices'][0]
        else:
            bands = None

        if is_not_null(args.sim):
            sim_srcs = load_yaml(args.sim)
            injected_src = sim_srcs.get('injected_source', None)
        else:
            injected_src = None

        xlims = (1e1, 1e4)

        stack_plot = plot_limits_from_arrays(ldict, xlims, ylims, bands)

        if injected_src is not None:
            mc_model = injected_src['source_model']
            plot_mc_truth(stack_plot[1], mc_model)

        if args.outfile:
            stack_plot[0].savefig(args.outfile)
            return None
        return stack_plot




class PlotStack(Link):
    """Small class to plot the likelihood vs norm and spectral index

    """
    appname = 'stack-plot-stack'
    linkname_default = 'plot-stack'
    usage = "%s [options]" % (appname)
    description = "Plot the likelihood vs norm and index"

    default_options = dict(infile=defaults.generic['infile'],
                           outfile=defaults.generic['outfile'],
                           spec=defaults.common['spec'],
                           global_min=defaults.common['global_min'])

    __doc__ += Link.construct_docstring(default_options)

    def run_analysis(self, argv):
        """Run this analysis"""
        args = self._parser.parse_args(argv)
        exttype = splitext(args.infile)[-1]
        if exttype in ['.fits']:
            stack_castro = StackCastroData.create_from_fitsfile(args.infile, args.spec, norm_type='stack')
        else:
            raise ValueError("Can not read file type %s for SED" % exttype)

        stack_plot = plot_stack_castro(stack_castro, global_min=args.global_min)
        if args.outfile:
            stack_plot[0].savefig(args.outfile)
            return None
        return stack_plot


class PlotLimits_SG(ScatterGather):
    """Small class to generate configurations for `PlotLimits`

    This does a triple nested loop over targets, profiles and j-factor priors
    """
    appname = 'stack-plot-limits-sg'
    usage = "%s [options]" % (appname)
    description = "Make castro plots for set of targets"
    clientclass = PlotLimits

    job_time = 60

    default_options = dict(ttype=defaults.common['ttype'],
                           targetlist=defaults.common['targetlist'],
                           specs=defaults.common['specs'],
                           astro_priors=defaults.common['astro_priors'],
                           dry_run=defaults.common['dry_run'])

    __doc__ += Link.construct_docstring(default_options)

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        ttype = args['ttype']
        (targets_yaml, sim) = NAME_FACTORY.resolve_targetfile(args)
        if targets_yaml is None:
            return job_configs

        astro_priors = args['astro_priors']
        specs = args['specs']

        base_config = dict(bands=None,
                           sim=sim)

        targets = load_yaml(targets_yaml)
        for target_name, target_list in targets.items():
            for targ_prof in target_list:
                prof_specs = specs
                for astro_prior in astro_priors:
                    name_keys = dict(target_type=ttype,
                                     target_name=target_name,
                                     profile=targ_prof,
                                     astro_prior=astro_prior,
                                     fullpath=True)
                    input_path = NAME_FACTORY.stack_limitsfile(**name_keys)
                    for spec in prof_specs:
                        targ_key = "%s:%s:%s:%s" % (
                            target_name, targ_prof, astro_prior, spec)

                        output_path = input_path.replace(
                            '.fits', '_%s.png' % spec)
                        logfile = make_nfs_path(
                            output_path.replace('.png', '.log'))
                        job_config = base_config.copy()
                        job_config.update(dict(infile=input_path,
                                               outfile=output_path,
                                               astro_prior=astro_prior,
                                               logfile=logfile,
                                               spec=spec))
                        job_configs[targ_key] = job_config

        return job_configs


class PlotStackedLimits_SG(ScatterGather):
    """Small class to generate configurations for `PlotStackedLimits`

    This does a double nested loop over rosters and j-factor priors
    """
    appname = 'stack-plot-stacked-limits-sg'
    usage = "%s [options]" % (appname)
    description = "Make castro plots for set of targets"
    clientclass = PlotLimits

    job_time = 60

    default_options = dict(ttype=defaults.common['ttype'],
                           rosterlist=defaults.common['rosterlist'],
                           bands=defaults.collect['bands'],
                           specs=defaults.common['specs'],
                           astro_priors=defaults.common['astro_priors'],
                           sim=defaults.sims['sim'],
                           nsims=defaults.sims['nsims'],
                           seed=defaults.sims['seed'],
                           dry_run=defaults.common['dry_run'])

    __doc__ += Link.construct_docstring(default_options)

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        ttype = args['ttype']
        (roster_yaml, sim) = NAME_FACTORY.resolve_rosterfile(args)
        if roster_yaml is None:
            return job_configs

        roster_dict = load_yaml(roster_yaml)

        astro_priors = args['astro_priors']
        specs = args['specs']

        for roster_name in roster_dict.keys():
            rost_specs = specs
            for astro_prior in astro_priors:
                name_keys = dict(target_type=ttype,
                                 roster_name=roster_name,
                                 astro_prior=astro_prior,
                                 sim_name=sim,
                                 fullpath=True)
                for spec in rost_specs:
                    targ_key = "%s:%s:%s" % (roster_name, astro_prior, spec)
                    if sim is not None:
                        seedlist = range(
                            args['seed'], args['seed'] + args['nsims'])
                        sim_path = os.path.join('config', 'sim_%s.yaml' % sim)
                    else:
                        seedlist = [None]
                        sim_path = None

                    for seed in seedlist:
                        if seed is not None:
                            name_keys['seed'] = "%06i" % seed
                            input_path = NAME_FACTORY.sim_stackedlimitsfile(
                                **name_keys)
                            full_targ_key = "%s_%06i" % (targ_key, seed)
                        else:
                            input_path = NAME_FACTORY.stackedlimitsfile(
                                **name_keys)
                            full_targ_key = targ_key

                        output_path = input_path.replace(
                            '.fits', '_%s.png' % spec)
                        logfile = make_nfs_path(
                            output_path.replace('.png', '.log'))
                        job_config = dict(infile=input_path,
                                          outfile=output_path,
                                          astro_prior=astro_prior,
                                          logfile=logfile,
                                          sim=sim_path,
                                          spec=spec)
                        job_configs[full_targ_key] = job_config

        return job_configs


class PlotStack_SG(ScatterGather):
    """Small class to generate configurations for `PlotStack`

    This does a quadruple nested loop over targets, profiles,
    j-factor priors and specs
    """
    appname = 'stack-plot-stack-sg'
    usage = "%s [options]" % (appname)
    description = "Make castro plots for set of targets"
    clientclass = PlotStack

    job_time = 60

    default_options = dict(ttype=defaults.common['ttype'],
                           targetlist=defaults.common['targetlist'],
                           specs=defaults.common['specs'],
                           astro_priors=defaults.common['astro_priors'],
                           global_min=defaults.common['global_min'],
                           dry_run=defaults.common['dry_run'])

    __doc__ += Link.construct_docstring(default_options)

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        ttype = args['ttype']
        targetfile_info = NAME_FACTORY.resolve_targetfile(args)
        targets_yaml = targetfile_info[0]
        if targets_yaml is None:
            return job_configs

        targets = load_yaml(targets_yaml)

        astro_priors = args['astro_priors']
        specs = args['specs']
        global_min = args['global_min']

        for target_name, target_list in targets.items():
            for targ_prof in target_list:
                prof_specs = specs
                for astro_prior in astro_priors:
                    name_keys = dict(target_type=ttype,
                                     target_name=target_name,
                                     profile=targ_prof,
                                     astro_prior=astro_prior,
                                     fullpath=True)
                    input_path = NAME_FACTORY.stack_likefile(**name_keys)
                    for spec in prof_specs:
                        targ_key = "%s:%s:%s:%s" % (
                            target_name, targ_prof, astro_prior, spec)
                        output_path = input_path.replace(
                            '.fits', '_%s.png' % spec)
                        logfile = make_nfs_path(
                            output_path.replace('.png', '.log'))
                        job_config = dict(infile=input_path,
                                          outfile=output_path,
                                          astro_prior=astro_prior,
                                          logfile=logfile,
                                          global_min=global_min,
                                          spec=spec)
                        job_configs[targ_key] = job_config

        return job_configs


class PlotStackedStack_SG(ScatterGather):
    """Small class to generate configurations for `PlotStack`

    This does a triple loop over rosters, j-factor priors and specs
    """
    appname = 'stack-plot-stacked-stack-sg'
    usage = "%s [options]" % (appname)
    description = "Make castro plots for set of targets"
    clientclass = PlotStack

    job_time = 60

    default_options = dict(ttype=defaults.common['ttype'],
                           rosterlist=defaults.common['rosterlist'],
                           specs=defaults.common['specs'],
                           astro_priors=defaults.common['astro_priors'],
                           sim=defaults.sims['sim'],
                           nsims=defaults.sims['nsims'],
                           seed=defaults.sims['seed'],
                           global_min=defaults.common['global_min'],
                           dry_run=defaults.common['dry_run'])

    __doc__ += Link.construct_docstring(default_options)

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        ttype = args['ttype']
        (roster_yaml, sim) = NAME_FACTORY.resolve_rosterfile(args)
        if roster_yaml is None:
            return job_configs

        roster_dict = load_yaml(roster_yaml)

        astro_priors = args['astro_priors']
        specs = args['specs']
        global_min = args['global_min']

        for roster_name in roster_dict.keys():
            rost_specs = specs
            for astro_prior in astro_priors:
                name_keys = dict(target_type=ttype,
                                 roster_name=roster_name,
                                 astro_prior=astro_prior,
                                 sim_name=sim,
                                 fullpath=True)

                for spec in rost_specs:
                    targ_key = "%s:%s:%s" % (roster_name, astro_prior, spec)

                    if sim is not None:
                        seedlist = range(
                            args['seed'], args['seed'] + args['nsims'])
                    else:
                        seedlist = [None]

                    for seed in seedlist:
                        if seed is not None:
                            name_keys['seed'] = "%06i" % seed
                            input_path = NAME_FACTORY.sim_resultsfile(
                                **name_keys)
                            full_targ_key = "%s_%06i" % (targ_key, seed)
                        else:
                            input_path = NAME_FACTORY.resultsfile(**name_keys)
                            full_targ_key = targ_key

                        output_path = input_path.replace(
                            '.fits', '_%s.png' % spec)
                        logfile = make_nfs_path(
                            output_path.replace('.png', '.log'))
                        job_config = dict(infile=input_path,
                                          outfile=output_path,
                                          astro_prior=astro_prior,
                                          logfile=logfile,
                                          global_min=global_min,
                                          spec=spec)
                        job_configs[full_targ_key] = job_config

        return job_configs



class PlotControlLimits_SG(ScatterGather):
    """Small class to generate configurations for `PlotLimits`

    This does a quadruple loop over rosters, j-factor priors, specs, and expectation bands
    """
    appname = 'stack-plot-control-limits-sg'
    usage = "%s [options]" % (appname)
    description = "Make limits plots for positve controls"
    clientclass = PlotLimits

    job_time = 60

    default_options = dict(ttype=defaults.common['ttype'],
                           rosterlist=defaults.common['targetlist'],
                           specs=defaults.common['specs'],
                           astro_priors=defaults.common['astro_priors'],
                           sim=defaults.sims['sim'],
                           dry_run=defaults.common['dry_run'])

    __doc__ += Link.construct_docstring(default_options)

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        ttype = args['ttype']

        try:
            os.makedirs(os.path.join(ttype, 'results'))
        except OSError:
            pass

        (roster_yaml, sim) = NAME_FACTORY.resolve_rosterfile(args)
        if roster_yaml is None:
            return job_configs

        roster_dict = load_yaml(roster_yaml)

        astro_priors = args['astro_priors']
        specs = args['specs']

        sim_path = os.path.join('config', 'sim_%s.yaml' % sim)

        for roster_name in roster_dict.keys():
            rost_specs = specs
            for astro_prior in astro_priors:
                name_keys = dict(target_type=ttype,
                                 roster_name=roster_name,
                                 astro_prior=astro_prior,
                                 sim_name=sim,
                                 seed='summary',
                                 fullpath=True)
                bands_path = NAME_FACTORY.sim_stackedlimitsfile(**name_keys)

                for spec in rost_specs:
                    targ_key = "%s:%s:%s:%s" % (roster_name, astro_prior, sim, spec)
                    output_path = os.path.join(ttype, 'results',
                                               "control_%s_%s_%s_%s.png" % (roster_name, astro_prior, sim, spec))
                    logfile = make_nfs_path(output_path.replace('.png', '.log'))
                    job_config = dict(bands=bands_path,
                                      outfile=output_path,
                                      sim=sim_path,
                                      logfile=logfile,
                                      spec=spec)
                    job_configs[targ_key] = job_config
        return job_configs


class PlotControlMLEs_SG(ScatterGather):
    """Small class to generate configurations for `PlotMLEs`

    This does a quadruple loop over rosters, j-factor priors, specs, and expectation bands
    """
    appname = 'stack-plot-control-mles-sg'
    usage = "%s [options]" % (appname)
    description = "Make mle plots for positve controls"
    clientclass = PlotMLEs

    job_time = 60

    default_options = dict(ttype=defaults.common['ttype'],
                           rosterlist=defaults.common['targetlist'],
                           specs=defaults.common['specs'],
                           astro_priors=defaults.common['astro_priors'],
                           sim=defaults.sims['sim'],
                           dry_run=defaults.common['dry_run'])

    __doc__ += Link.construct_docstring(default_options)

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        ttype = args['ttype']

        try:
            os.makedirs(os.path.join(ttype, 'results'))
        except OSError:
            pass

        (roster_yaml, sim) = NAME_FACTORY.resolve_rosterfile(args)
        if roster_yaml is None:
            return job_configs

        roster_dict = load_yaml(roster_yaml)

        astro_priors = args['astro_priors']
        specs = args['specs']

        sim_path = os.path.join('config', 'sim_%s.yaml' % sim)

        for roster_name in roster_dict.keys():
            rost_specs = specs
            for astro_prior in astro_priors:
                name_keys = dict(target_type=ttype,
                                 roster_name=roster_name,
                                 astro_prior=astro_prior,
                                 sim_name=sim,
                                 seed='summary',
                                 fullpath=True)
                bands_path = NAME_FACTORY.sim_stackedlimitsfile(**name_keys)

                for spec in rost_specs:
                    targ_key = "%s:%s:%s:%s" % (roster_name, astro_prior, sim, spec)
                    output_path = os.path.join(ttype, 'results',
                                               "control_mle_%s_%s_%s_%s.png" % (roster_name, astro_prior, sim, spec))
                    logfile = make_nfs_path(output_path.replace('.png', '.log'))
                    job_config = dict(bands=bands_path,
                                      outfile=output_path,
                                      sim=sim_path,
                                      logfile=logfile,
                                      spec=spec)
                    job_configs[targ_key] = job_config
        return job_configs



class PlotFinalLimits_SG(ScatterGather):
    """Small class to generate configurations for `PlotLimits`

    This does a quadruple loop over rosters, j-factor priors, specs, and expectation bands
    """
    appname = 'stack-plot-final-limits-sg'
    usage = "%s [options]" % (appname)
    description = "Make final limits plots"
    clientclass = PlotLimits

    job_time = 60

    default_options = dict(ttype=defaults.common['ttype'],
                           rosterlist=defaults.common['rosterlist'],
                           specs=defaults.common['specs'],
                           astro_priors=defaults.common['astro_priors'],
                           sims=defaults.sims['sims'],
                           dry_run=defaults.common['dry_run'])

    __doc__ += Link.construct_docstring(default_options)

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        ttype = args['ttype']
        (roster_yaml, sim) = NAME_FACTORY.resolve_rosterfile(args)
        if roster_yaml is None:
            return job_configs
        if sim is not None:
            raise ValueError("Sim argument set of plotting data results")

        roster_dict = load_yaml(roster_yaml)

        astro_priors = args['astro_priors']
        specs = args['specs']

        sims = args['sims']
        for roster_name in roster_dict.keys():
            rost_specs = specs
            for astro_prior in astro_priors:
                name_keys = dict(target_type=ttype,
                                 roster_name=roster_name,
                                 astro_prior=astro_prior,
                                 fullpath=True)
                input_path = NAME_FACTORY.stackedlimitsfile(**name_keys)
                for sim in sims:
                    name_keys.update(sim_name=sim,
                                     seed='summary')
                    bands_path = NAME_FACTORY.sim_stackedlimitsfile(**name_keys)

                    for spec in rost_specs:
                        targ_key = "%s:%s:%s:%s" % (roster_name, astro_prior, sim, spec)
                        output_path = os.path.join(ttype, 'results',
                                                   "final_%s_%s_%s_%s.png" % (roster_name, astro_prior, sim, spec))
                        logfile = make_nfs_path(output_path.replace('.png', '.log'))
                        job_config = dict(infile=input_path,
                                          outfile=output_path,
                                          bands=bands_path,
                                          logfile=logfile,
                                          spec=spec)
                        job_configs[targ_key] = job_config

        return job_configs






def register_classes():
    """Register these classes with the `LinkFactory` """
    PlotStackSpectra.register_class()
    PlotLimits.register_class()
    PlotLimits_SG.register_class()
    PlotMLEs.register_class()
    PlotStack.register_class()
    PlotStack_SG.register_class()
    PlotStackedStack_SG.register_class()
    PlotStackedLimits_SG.register_class()
    PlotControlLimits_SG.register_class()
    PlotControlMLEs_SG.register_class()
    PlotFinalLimits_SG.register_class()
