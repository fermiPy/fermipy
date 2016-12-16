# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Merge source maps to build composite sources
"""
from __future__ import absolute_import, division, print_function

import os
import sys

import argparse

import BinnedAnalysis as BinnedAnalysis
import pyLikelihood as pyLike

from fermipy.jobs.chain import Link
from fermipy.jobs.scatter_gather import ConfigMaker
from fermipy.jobs.lsf_impl import build_sg_from_link
from fermipy.diffuse.name_policy import NameFactory
from fermipy.diffuse.binning import Component
from fermipy.diffuse.catalog_src_manager import make_catalog_comp_dict

NAME_FACTORY = NameFactory()


class GtMergeSourceMaps(object):
    """Small class to merge source maps for composite sources.

    This is useful for parallelizing source map creation.
    """
    NULL_MODEL = 'srcmdls/null.xml'

    def __init__(self):
        """C'tor
        """
        self.parser = GtMergeSourceMaps._make_parser()
        self.link = GtMergeSourceMaps._make_link()

    @staticmethod
    def _make_parser():
        """Make an argument parser for this class """
        usage = "gt_merge_srcmaps.py [options]"
        description = "Run gtsrcmaps for one or more energy planes for a single source"

        parser = argparse.ArgumentParser(usage=usage, description=description)
        parser.add_argument('--irfs', type=str, default='CALDB',
                            help='Instrument response functions')
        parser.add_argument('--expcube', type=str, default=None,
                            help='Input Livetime cube file')
        parser.add_argument('--srcmaps', type=str, default=None,
                            help='Input source maps file')
        parser.add_argument('--bexpmap', type=str, default=None,
                            help='Input binned exposure map file')
        parser.add_argument('--srcmdl', type=str, default=None,
                            help='Input source model xml file')
        parser.add_argument('--merged', type=str, default=None,
                            help='Name of merged source')
        parser.add_argument('--outfile', type=str, default=None,
                            help='Output source map file')
        parser.add_argument('--outxml', type=str, default=None,
                            help='Output source model xml file')
        parser.add_argument('--gzip', action='store_true',
                            help='Compress output file')
        return parser

    @staticmethod
    def _make_link():
        link = Link('merge-srcmaps',
                    appname='fermipy-merge-srcmaps',
                    options=dict(irfs=None, expcube=None, srcmaps=None,
                                 bexpmap=None, srcmdl=None, merged=None,
                                 outfile=None, outxml=None, gzip=True),
                    flags=['gzip'],
                    input_file_args=['expcube', 'cmap', 'bexpmap', 'srcmdl'],
                    output_file_args=['outfile', 'outxml'])
        return link

    def run(self, argv):
        """Run this analysis"""
        args = self.parser.parse_args(argv)
        obs = BinnedAnalysis.BinnedObs(irfs=args.irfs,
                                       expCube=args.expcube,
                                       srcMaps=args.srcmaps,
                                       binnedExpMap=args.bexpmap)

        like = BinnedAnalysis.BinnedAnalysis(obs,
                                             optimizer='MINUIT',
                                             srcModel=GtMergeSourceMaps.NULL_MODEL,
                                             wmap=None)

        like.logLike.set_use_single_fixed_map(False)

        print ("Reading xml model from %s" % args.srcmdl)
        source_factory = pyLike.SourceFactory(obs.observation)
        source_factory.readXml(args.srcmdl, BinnedAnalysis._funcFactory, False, True, True)
        strv = pyLike.StringVector()
        source_factory.fetchSrcNames(strv)
        source_names = [strv[i] for i in range(strv.size())]

        missing_sources = []
        srcs_to_merge = []
        for source_name in source_names:
            try:
                source = source_factory.releaseSource(source_name)
                like.addSource(source)
                srcs_to_merge.append(source_name)
            except KeyError:
                missing_sources.append(source_name)

        comp = like.mergeSources(args.merged, source_names, 'ConstantValue')
        like.logLike.getSourceMap(comp.getName())

        print ("Missed sources: ", missing_sources)

        print ("Writing output source map file %s" % args.outfile)
        like.logLike.saveSourceMaps(args.outfile, False, False)
        if args.gzip:
            os.system("gzip -9 %s" % args.outfile)

        print ("Writing output xml file %s" % args.outxml)
        like.writeXml(args.outxml)


class ConfigMaker_MergeSrcmaps(ConfigMaker):
    """Small class to generate configurations for this script
    """

    def __init__(self, link):
        """C'tor
        """
        ConfigMaker.__init__(self)
        self.link = link

    def add_arguments(self, parser, action):
        """Hook to add arguments to the command line argparser

        Parameters:
        ----------------
        parser : `argparse.ArgumentParser'
            Object we are filling

        action : str
            String specifing what we want to do

        This adds the following arguments:
        --comp     : binning component definition yaml file
        --data     : datset definition yaml file
        --irf_ver  : IRF verions string (e.g., 'V6')
        --sources  : Catalog model component definition yaml file'
        """
        parser.add_argument('--comp', type=str, default=None,
                            help='Yaml file with component definitions')
        parser.add_argument('--data', type=str, default=None,
                            help='Yaml file with dataset definitions')
        parser.add_argument('--irf_ver', type=str, default='V6',
                            help='Version of IRFs to use')
        parser.add_argument('--sources', type=str, default=None,
                            help='File with source merging configuration')

    def make_base_config(self, args):
        """Hook to build a baseline job configuration

        Parameters:
        ----------------
        args : `argparse.Namespace'
            Command line arguments, see add_arguments
        """
        self.link.update_args(args.__dict__)
        return self.link.args

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        input_config = {}
        job_configs = {}

        components = Component.build_from_yamlfile(args.comp)
        NAME_FACTORY.update_base_dict(args.data)
        ret_dict = make_catalog_comp_dict(sources=args.sources, basedir='.')
        comp_info_dict = ret_dict['comp_info_dict']

        for split_ver, split_dict in comp_info_dict.items():
            for source_key, source_dict in split_dict.items():

                print (split_ver, source_key, source_dict.model_type)
                if source_dict.model_type != 'CompositeSource':
                    continue

                for comp in components:
                    zcut = "zmax%i" % comp.zmax
                    key = "%s_%s" % (source_key, comp.make_key('{ebin_name}_{evtype_name}'))
                    name_keys = dict(zcut=zcut,
                                     sourcekey=source_key,
                                     ebin=comp.ebin_name,
                                     psftype=comp.evtype_name,
                                     coordsys='GAL',
                                     irf_ver=args.irf_ver)
                    nested_name_keys = dict(zcut=zcut,
                                            sourcekey=source_dict.catalog_info.catalog_name,
                                            ebin=comp.ebin_name,
                                            psftype=comp.evtype_name,
                                            coordsys='GAL',
                                            irf_ver=args.irf_ver)

                    job_configs[key] = dict(srcmaps=NAME_FACTORY.srcmaps(**nested_name_keys),
                                            expcube=NAME_FACTORY.ltcube(**name_keys),
                                            irfs=NAME_FACTORY.irfs(**name_keys),
                                            bexpmap=NAME_FACTORY.bexpcube(**name_keys),
                                            srcmdl=NAME_FACTORY.nested_srcmdl_xml(**name_keys),
                                            merged=source_key,
                                            outfile=NAME_FACTORY.srcmaps(**name_keys),
                                            outxml=NAME_FACTORY.srcmdl_xml(**name_keys))

        output_config = {}
        return input_config, job_configs, output_config


def build_scatter_gather():
    """Build and return a ScatterGather object that can invoke this script"""
    gtmerge = GtMergeSourceMaps()
    link = gtmerge.link

    lsf_args = {'W': 1500,
                'R': 'rhel60'}

    usage = "fermipy-merge-srcmaps.py [options] input"
    description = "Prepare data for diffuse all-sky analysis"

    config_maker = ConfigMaker_MergeSrcmaps(link)
    lsf_sg = build_sg_from_link(link, config_maker,
                                lsf_args=lsf_args,
                                usage=usage,
                                description=description)
    return lsf_sg


def main_single():
    """Entry point for command line use for single job """
    gtsmp = GtMergeSourceMaps()
    gtsmp.run(sys.argv[1:])


def main_batch():
    """Entry point for command line use  for dispatching batch jobs """
    lsf_sg = build_scatter_gather()
    lsf_sg(sys.argv)

if __name__ == '__main__':
    main_single()
