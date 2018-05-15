#!/usr/bin/env python

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Collect information for simulated realizations of an analysis
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import yaml
import numpy as np


from astropy.table import Table, Column, vstack

from fermipy.utils import load_yaml, init_matplotlib_backend

from fermipy.jobs.utils import is_not_null
from fermipy.jobs.link import Link
from fermipy.jobs.scatter_gather import ScatterGather
from fermipy.jobs.slac_impl import make_nfs_path

from fermipy.jobs.name_policy import NameFactory
from fermipy.jobs import defaults

init_matplotlib_backend('Agg')


NAME_FACTORY = NameFactory(basedir=('.'))


def _get_enum_bins(configfile):
    """Get the number of energy bin in the SED

    Parameters
    ----------

    configfile : str
        Fermipy configuration file.

    Returns
    -------

    nbins : int
        The number of energy bins

    """
    config = yaml.safe_load(open(configfile))

    emin = config['selection']['emin']
    emax = config['selection']['emax']
    log_emin = np.log10(emin)
    log_emax = np.log10(emax)
    ndec = log_emax - log_emin
    binsperdec = config['binning']['binsperdec']
    nebins = int(np.ceil(binsperdec * ndec))

    return nebins



def fill_output_table(filelist, hdu, collist, nbins):
    """Fill the arrays from the files in filelist

    Parameters
    ----------

    filelist : list
        List of the files to get data from.

    hdu : str
        Name of the HDU containing the table with the input data.

    colllist : list
        List of the column names

    nbins : int
        Number of bins in the input data arrays

    Returns
    -------

    table : astropy.table.Table
        A table with all the requested data extracted.

    """
    nfiles = len(filelist)
    shape = (nbins, nfiles)
    outdict = {}
    for c in collist:
        outdict[c['name']] = np.ndarray(shape)

    sys.stdout.write('Working on %i files: ' % nfiles)
    sys.stdout.flush()
    for i, f in enumerate(filelist):
        sys.stdout.write('.')
        sys.stdout.flush()
        tab = Table.read(f, hdu)
        for c in collist:
            cname = c['name']
            outdict[cname][:, i] = tab[cname].data
    sys.stdout.write('!\n')
    outcols = []
    for c in collist:
        cname = c['name']
        if 'unit' in c:
            col = Column(data=outdict[cname], name=cname,
                         dtype=np.float, shape=nfiles, unit=c['unit'])
        else:
            col = Column(data=outdict[cname], name=cname,
                         dtype=np.float, shape=nfiles)
        outcols.append(col)
    tab = Table(data=outcols)
    return tab


def vstack_tables(filelist, hdus):
    """vstack a set of HDUs from a set of files

    Parameters
    ----------

    filelist : list
        List of the files to get data from.

    hdus : list
        Names of the HDU containing the table with the input data.

    Returns
    -------

    out_tables : list
        A list with the table with all the requested data extracted.

    out_names : list
        A list with the names of the tables.

    """
    nfiles = len(filelist)
    out_tables = []
    out_names = []
    for hdu in hdus:
        sys.stdout.write('Working on %i files for %s: ' % (nfiles, hdu))
        sys.stdout.flush()
        tlist = []
        for f in filelist:
            try:
                tab = Table.read(f, hdu)
                tlist.append(tab)
                sys.stdout.write('.')
            except KeyError:
                sys.stdout.write('x')
            sys.stdout.flush()
        sys.stdout.write('!\n')
        if tlist:
            out_table = vstack(tlist)
            out_tables.append(out_table)
            out_names.append(hdu)
    return (out_tables, out_names)


def collect_summary_stats(data):
    """Collect summary statisitics from an array

    This creates a dictionry of output arrays of summary
    statistics, with the input array dimension reducted by one.

    Parameters
    ----------

    data : `numpy.ndarray`
        Array with the collected input data


    Returns
    -------

    output : dict
        Dictionary of `np.ndarray` with the summary data.
        These include mean, std, median, and 4 quantiles (0.025, 0.16, 0.86, 0.975).

    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    median = np.median(data, axis=0)
    q02, q16, q84, q97 = np.percentile(data, [2.5, 16, 84, 97.5], axis=0)

    o = dict(mean=mean,
             std=std,
             median=median,
             q02=q02,
             q16=q16,
             q84=q84,
             q97=q97)

    return o


def add_summary_stats_to_table(table_in, table_out, colnames):
    """Collect summary statisitics from an input table and add them to an output table

     Parameters
    ----------

    table_in : `astropy.table.Table`
        Table with the input data.

    table_out : `astropy.table.Table`
        Table with the output data.

    colnames : list
        List of the column names to get summary statistics for.

    """
    for col in colnames:
        col_in = table_in[col]
        stats = collect_summary_stats(col_in.data)
        for k, v in stats.items():
            out_name = "%s_%s" % (col, k)
            col_out = Column(data=np.vstack(
                [v]), name=out_name, dtype=col_in.dtype, shape=v.shape, unit=col_in.unit)
            table_out.add_column(col_out)


def summarize_sed_results(sed_table):
    """Build a stats summary table for a table that has all the SED results """
    del_cols = ['dnde', 'dnde_err', 'dnde_errp', 'dnde_errn', 'dnde_ul',
                'e2dnde', 'e2dnde_err', 'e2dnde_errp', 'e2dnde_errn', 'e2dnde_ul',
                'norm', 'norm_err', 'norm_errp', 'norm_errn', 'norm_ul',
                'ts']
    stats_cols = ['dnde', 'dnde_ul',
                  'e2dnde', 'e2dnde_ul',
                  'norm', 'norm_ul']

    table_out = Table(sed_table[0])
    table_out.remove_columns(del_cols)
    add_summary_stats_to_table(sed_table, table_out, stats_cols)
    return table_out


class CollectSED(Link):
    """Small class to collect SED results from a series of simulations.

    """
    appname = 'fermipy-collect-sed'
    linkname_default = 'collect-sed'
    usage = '%s [options]' % (appname)
    description = "Collect SED results from simulations"

    default_options = dict(sed_file=defaults.common['sed_file'],
                           outfile=defaults.generic['outfile'],
                           config=defaults.common['config'],
                           summaryfile=defaults.generic['summaryfile'],
                           nsims=defaults.sims['nsims'],
                           enumbins=(12, 'Number of energy bins', int),
                           seed=defaults.sims['seed'],
                           dry_run=defaults.common['dry_run'])

    collist = [dict(name='e_min', unit='MeV'),
               dict(name='e_ref', unit='MeV'),
               dict(name='e_max', unit='MeV'),
               dict(name='ref_dnde_e_min', unit='cm-2 MeV-1 ph s-1'),
               dict(name='ref_dnde_e_max', unit='cm-2 MeV-1 ph s-1'),
               dict(name='ref_dnde', unit='cm-2 MeV-1 ph s-1'),
               dict(name='ref_flux', unit='cm-2 ph s-1'),
               dict(name='ref_eflux', unit='cm-2 MeV s-1'),
               dict(name='ref_npred'),
               dict(name='dnde', unit='cm-2 MeV-1 ph s-1'),
               dict(name='dnde_err', unit='cm-2 MeV-1 ph s-1'),
               dict(name='dnde_errp', unit='cm-2 MeV-1 ph s-1'),
               dict(name='dnde_errn', unit='cm-2 MeV-1 ph s-1'),
               dict(name='dnde_ul', unit='cm-2 MeV-1 ph s-1'),
               dict(name='e2dnde', unit='cm-2 MeV s-1'),
               dict(name='e2dnde_err', unit='cm-2 MeV s-1'),
               dict(name='e2dnde_errp', unit='cm-2 MeV s-1'),
               dict(name='e2dnde_errn', unit='cm-2 MeV s-1'),
               dict(name='e2dnde_ul', unit='cm-2 MeV s-1'),
               dict(name='norm'),
               dict(name='norm_err'),
               dict(name='norm_errp'),
               dict(name='norm_errn'),
               dict(name='norm_ul'),
               dict(name='ts')]

    __doc__ += Link.construct_docstring(default_options)

    def run_analysis(self, argv):
        """Run this analysis"""
        args = self._parser.parse_args(argv)


        sedfile = args.sed_file

        if is_not_null(args.config):
            configfile = os.path.join(os.path.dirname(sedfile), args.config)
        else:
            configfile = os.path.join(os.path.dirname(sedfile), 'config.yaml')

        nbins = _get_enum_bins(configfile)

        first = args.seed
        last = first + args.nsims
        flist = [sedfile.replace("_SEED.fits", "_%06i.fits" % seed)
                 for seed in range(first, last)]
        outfile = args.outfile
        summaryfile = args.summaryfile


        outtable = fill_output_table(
            flist, "SED", CollectSED.collist, nbins=nbins)

        if is_not_null(outfile):
            outtable.write(outfile)

        if is_not_null(summaryfile):
            summary = summarize_sed_results(outtable)
            summary.write(summaryfile)


class CollectSED_SG(ScatterGather):
    """Small class to generate configurations for `CollectSED`

    This loops over all the targets defined in the target list
    """
    appname = 'fermipy-collect-sed-sg'
    usage = "%s [options]" % (appname)
    description = "Collect SED data from a set of simulations for a series of ROIs"
    clientclass = CollectSED

    job_time = 120

    default_options = dict(ttype=defaults.common['ttype'],
                           targetlist=defaults.common['targetlist'],
                           config=defaults.common['config'],
                           sim=defaults.sims['sim'],
                           nsims=defaults.sims['nsims'],
                           seed=defaults.sims['seed'],
                           write_full=defaults.collect['write_full'],
                           write_summary=defaults.collect['write_summary'])

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

        write_full = args['write_full']

        targets = load_yaml(targets_yaml)

        base_config = dict(config=args['config'],
                           nsims=args['nsims'],
                           seed=args['seed'])

        first = args['seed']
        last = first + args['nsims'] - 1

        for target_name, profile_list in targets.items():
            for profile in profile_list:
                full_key = "%s:%s:%s" % (target_name, profile, sim)
                name_keys = dict(target_type=ttype,
                                 target_name=target_name,
                                 sim_name=sim,
                                 profile=profile,
                                 fullpath=True)
                sed_file = NAME_FACTORY.sim_sedfile(**name_keys)
                outfile = sed_file.replace(
                    '_SEED.fits', '_collected_%06i_%06i.fits' % (first, last))
                logfile = make_nfs_path(outfile.replace('.fits', '.log'))
                if not write_full:
                    outfile = None
                summaryfile = sed_file.replace(
                    '_SEED.fits', '_summary_%06i_%06i.fits' % (first, last))
                job_config = base_config.copy()
                job_config.update(dict(sed_file=sed_file,
                                       outfile=outfile,
                                       summaryfile=summaryfile,
                                       logfile=logfile))
                job_configs[full_key] = job_config

        return job_configs


def register_classes():
    """Register these classes with the `LinkFactory` """
    CollectSED.register_class()
    CollectSED_SG.register_class()
