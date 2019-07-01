#!/usr/bin/env python

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Collect information for simulated realizations of an analysis
"""
from __future__ import absolute_import, division, print_function

#from dmsky.roster import RosterLibrary
from astropy.table import Table

from fermipy.utils import load_yaml, init_matplotlib_backend
from fermipy import fits_utils

from fermipy.jobs.utils import is_null, is_not_null
from fermipy.jobs.link import Link
from fermipy.jobs.scatter_gather import ScatterGather
from fermipy.jobs.slac_impl import make_nfs_path
from fermipy.jobs.target_collect import vstack_tables, add_summary_stats_to_table

from . import defaults
from .name_policy import NameFactory

init_matplotlib_backend('Agg')

NAME_FACTORY = NameFactory(basedir=('.'))


def summarize_limits_results(limit_table):
    """Build a stats summary table for a table that has all the SED results """
    del_cols = ['ul_0.68', 'ul_0.95', 'mles']
    stats_cols = ['ul_0.95', 'mles']

    table_out = Table(limit_table[0])
    table_out.remove_columns(del_cols)
    add_summary_stats_to_table(limit_table, table_out, stats_cols)
    return table_out


class CollectLimits(Link):
    """Small class to collect limit results from a series of simulations.

    """
    appname = 'stack-collect-limits'
    linkname_default = 'collect-limits'
    usage = '%s [options]' % (appname)
    description = "Collect Limits from simulations"

    default_options = dict(limitfile=defaults.generic['limitfile'],
                           specconfig=defaults.common['specconfig'],
                           outfile=defaults.generic['outfile'],
                           summaryfile=defaults.generic['summaryfile'],
                           nsims=defaults.sims['nsims'],
                           seed=defaults.sims['seed'],
                           dry_run=defaults.common['dry_run'])

    __doc__ += Link.construct_docstring(default_options)


    def run_analysis(self, argv):
        """Run this analysis"""
        args = self._parser.parse_args(argv)

        limitfile = args.limitfile
        first = args.seed
        last = first + args.nsims
        flist = [limitfile.replace("_SEED.fits", "_%06i.fits" % seed)\
                     for seed in range(first, last)]

        spec_config = load_yaml(args.specconfig)
        specs = spec_config['specs']
        sum_specs = specs.copy()

        outfile = args.outfile
        summaryfile = args.summaryfile

        hdus = sum_specs + ['INDICES']

        out_tables, out_names = vstack_tables(flist, hdus)

        if is_not_null(outfile):
            fits_utils.write_tables_to_fits(outfile,
                                            out_tables,
                                            namelist=out_names)

        if is_not_null(summaryfile):
            summary_tables = []
            for ot in out_tables[0:-1]:
                summary_table = summarize_limits_results(ot)
                summary_tables.append(summary_table)
            summary_tables.append(Table(out_tables[-1][0]))
            fits_utils.write_tables_to_fits(summaryfile,
                                            summary_tables,
                                            namelist=out_names)


class CollectLimits_SG(ScatterGather):
    """Small class to generate configurations for `CollectLimits`

    This does a triple loop over all targets, profiles and j-factor priors.
    """
    appname = 'stack-collect-limits-sg'
    usage = "%s [options]" % (appname)
    description = "Run analyses on a series of ROIs"
    clientclass = CollectLimits

    job_time = 120

    default_options = dict(ttype=defaults.common['ttype'],
                           targetlist=defaults.common['targetlist'],
                           specconifg=defaults.common['specconfig'],
                           astro_priors=defaults.common['astro_priors'],
                           sim=defaults.sims['sim'],
                           nsims=defaults.sims['nsims'],
                           seed=defaults.sims['seed'],
                           write_full=defaults.collect['write_full'],
                           dry_run=defaults.common['dry_run'])

    __doc__ += Link.construct_docstring(default_options)

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        ttype = args['ttype']
        (targets_yaml, sim) = NAME_FACTORY.resolve_targetfile(
            args, require_sim_name=True)
        if targets_yaml is None:
            return job_configs

        specconfig = NAME_FACTORY.resolve_specconfig(args)

        astro_priors = args['astro_priors']
        write_full = args.get('write_full', False)

        targets = load_yaml(targets_yaml)
        base_config = dict(nsims=args['nsims'],
                           seed=args['seed'],
                           specconfig=specconfig)

        for target_name, profile_list in targets.items():
            for profile in profile_list:
                for astro_prior in astro_priors:
                    if is_null(astro_prior):
                        astro_prior = 'none'
                    full_key = "%s:%s:%s:%s" % (
                        target_name, profile, sim, astro_prior)
                    name_keys = dict(target_type=ttype,
                                     target_name=target_name,
                                     sim_name=sim,
                                     profile=profile,
                                     astro_prior=astro_prior,
                                     fullpath=True)
                    limitfile = NAME_FACTORY.sim_stack_limitsfile(**name_keys)
                    first = args['seed']
                    last = first + args['nsims'] - 1
                    outfile = limitfile.replace(
                        '_SEED.fits', '_collected_%06i_%06i.fits' %
                        (first, last))
                    logfile = make_nfs_path(outfile.replace('.fits', '.log'))
                    if not write_full:
                        outfile = None
                    summaryfile = limitfile.replace(
                        '_SEED.fits', '_summary_%06i_%06i.fits' %
                        (first, last))
                    job_config = base_config.copy()
                    job_config.update(dict(limitfile=limitfile,
                                           astro_prior=astro_prior,
                                           outfile=outfile,
                                           summaryfile=summaryfile,
                                           logfile=logfile))
                    job_configs[full_key] = job_config

        return job_configs


class CollectStackedLimits_SG(ScatterGather):
    """Small class to generate configurations for this script

    This adds the following arguments:
    """
    appname = 'stack-collect-stacked-limits-sg'
    usage = "%s [options]" % (appname)
    description = "Run analyses on a series of ROIs"
    clientclass = CollectLimits

    job_time = 120

    default_options = dict(ttype=defaults.common['ttype'],
                           rosterlist=defaults.common['targetlist'],
                           astro_priors=defaults.common['astro_priors'],
                           sim=defaults.sims['sim'],
                           nsims=defaults.sims['nsims'],
                           seed=defaults.sims['seed'],
                           write_full=defaults.collect['write_full'],
                           write_summary=defaults.collect['write_summary'],
                           dry_run=defaults.common['dry_run'])

    __doc__ += Link.construct_docstring(default_options)

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        ttype = args['ttype']
        (roster_yaml, sim) = NAME_FACTORY.resolve_rosterfile(
            args, require_sim_name=True)
        if roster_yaml is None:
            return job_configs

        specconfig = NAME_FACTORY.resolve_specconfig(args)

        astro_priors = args['astro_priors']
        write_full = args['write_full']
        first = args['seed']
        last = first + args['nsims'] - 1

        base_config = dict(nsims=args['nsims'],
                           seed=args['seed'])

        roster_dict = load_yaml(roster_yaml)
        for roster_name in roster_dict.keys():
            for astro_prior in astro_priors:
                if is_null(astro_prior):
                    astro_prior = 'none'
                full_key = "%s:%s:%s" % (roster_name, sim, astro_prior)
                name_keys = dict(target_type=ttype,
                                 roster_name=roster_name,
                                 sim_name=sim,
                                 astro_prior=astro_prior,
                                 fullpath=True)

                limitfile = NAME_FACTORY.sim_stackedlimitsfile(**name_keys)
                outfile = limitfile.replace(
                    '_SEED.fits', '_collected_%06i_%06i.fits' %
                    (first, last))
                logfile = make_nfs_path(outfile.replace('.fits', '.log'))
                if not write_full:
                    outfile = None
                summaryfile = limitfile.replace('_SEED.fits', '_summary.fits')

                job_config = base_config.copy()
                job_config.update(dict(limitfile=limitfile,
                                       specconfig=specconfig,
                                       astro_prior=astro_prior,
                                       outfile=outfile,
                                       summaryfile=summaryfile,
                                       logfile=logfile))
                job_configs[full_key] = job_config

        return job_configs


def register_classes():
    """Register these classes with the `LinkFactory` """
    CollectLimits.register_class()
    CollectLimits_SG.register_class()
    CollectStackedLimits_SG.register_class()
