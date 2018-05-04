#!/usr/bin/env python
#

# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Top level script to make a castro plot in mass / sigmav space
"""
from __future__ import absolute_import, division, print_function


from fermipy.utils import init_matplotlib_backend, load_yaml

from fermipy.jobs.link import Link
from fermipy.jobs.scatter_gather import ScatterGather
from fermipy.jobs.slac_impl import make_nfs_path

from fermipy.castro import CastroData
from fermipy.sed_plotting import plotCastro

from fermipy.jobs.name_policy import NameFactory
from fermipy.jobs import defaults

init_matplotlib_backend()
NAME_FACTORY = NameFactory(basedir='.')


class PlotCastro(Link):
    """Small class wrap an analysis script.

    This is useful for parallelizing analysis using the fermipy.jobs module.
    """
    appname = 'fermipy-plot-castro'
    linkname_default = 'plot-castro'
    usage = '%s [options]' % (appname)
    description = "Plot likelihood v. flux normalization and energy"

    default_options = dict(infile=defaults.generic['infile'],
                           outfile=defaults.generic['outfile'])

    def __init__(self, **kwargs):
        """C'tor
        """
        linkname, init_dict = self._init_dict(**kwargs)
        super(PlotCastro, self).__init__(linkname, **init_dict)

    def run_analysis(self, argv):
        """Run this analysis"""
        args = self._parser.parse_args(argv)
        castro_data = CastroData.create_from_sedfile(args.infile)
        ylims = [1e-8, 1e-5]

        plot = plotCastro(castro_data, ylims)
        if args.outfile:
            plot[0].savefig(args.outfile)
            return None
        return plot


class PlotCastro_SG(ScatterGather):
    """Small class to generate configurations for this script

    This adds the following arguments:
    """
    appname = 'fermipy-plot-castro-sg'
    usage = "%s [options]" % (appname)
    description = "Make castro plots for set of targets"
    clientclass = PlotCastro

    job_time = 60

    default_options = dict(ttype=defaults.common['ttype'],
                           targetlist=defaults.common['targetlist'])

    def __init__(self, link, **kwargs):
        """C'tor
        """
        super(PlotCastro_SG, self).__init__(link,
                                            options=kwargs.get('options',
                                                               self.default_options.copy()))

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        ttype = args['ttype']
        (targets_yaml, sim) = NAME_FACTORY.resolve_targetfile(args)
        if targets_yaml is None:
            return job_configs

        targets = load_yaml(targets_yaml)

        for target_name, target_list in targets.items():
            for targ_prof in target_list:
                name_keys = dict(target_type=ttype,
                                 target_name=target_name,
                                 profile=targ_prof,
                                 fullpath=True)
                targ_key = "%s_%s" % (target_name, targ_prof)
                input_path = NAME_FACTORY.sedfile(**name_keys)
                output_path = input_path.replace('.fits', '.png')
                logfile = make_nfs_path(input_path.replace('.fits', '.log'))
                job_config = dict(infile=input_path,
                                  outfile=output_path,
                                  logfile=logfile)
                job_configs[targ_key] = job_config

        return job_configs


def register_classes():
    """Register these classes with the `LinkFactory` """
    PlotCastro.register_class()
    PlotCastro_SG.register_class()
