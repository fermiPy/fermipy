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

from fermipy.jobs.file_archive import FileFlags
from fermipy.jobs.chain import add_argument, Link
from fermipy.jobs.scatter_gather import ConfigMaker
from fermipy.jobs.lsf_impl import build_sg_from_link
from fermipy.diffuse.name_policy import NameFactory
from fermipy.diffuse.binning import Component
from fermipy.diffuse.catalog_src_manager import make_catalog_comp_dict
from fermipy.diffuse import defaults as diffuse_defaults

NAME_FACTORY = NameFactory()


class GtMergeSourceMaps(object):
    """Small class to merge source maps for composite sources.

    This is useful for parallelizing source map creation.
    """
    NULL_MODEL = 'srcmdls/null.xml'

    default_options = dict(irfs=diffuse_defaults.gtopts['irfs'],
                           expcube=diffuse_defaults.gtopts['expcube'],
                           bexpmap=diffuse_defaults.gtopts['bexpmap'],
                           srcmaps=diffuse_defaults.gtopts['srcmaps'],
                           srcmdl=diffuse_defaults.gtopts['srcmdl'],
                           outfile=diffuse_defaults.gtopts['outfile'],
                           merged=(None, 'Name of merged source', str),
                           outxml=(None, 'Output source model xml file', str),
                           gzip=(False, 'Compress output file', bool))

    def __init__(self, **kwargs):
        """C'tor
        """
        self.parser = GtMergeSourceMaps.make_parser()
        self.link = GtMergeSourceMaps.make_link(**kwargs)

    @staticmethod
    def make_parser():
        """Make an argument parser for this class """
        usage = "gt_merge_srcmaps.py [options]"
        description = "Run gtsrcmaps for one or more energy planes for a single source"

        parser = argparse.ArgumentParser(usage=usage, description=description)
        for key, val in GtMergeSourceMaps.default_options.items():
            add_argument(parser, key, val)
        return parser

    @staticmethod
    def make_link(**kwargs):
        """Make a `fermipy.jobs.Link object to run `GtMergeSourceMaps` """
        link = Link(kwargs.pop('linkname', 'merge-srcmaps'),
                    appname='fermipy-merge-srcmaps',
                    options=GtMergeSourceMaps.default_options.copy(),
                    file_args=dict(expcube=FileFlags.input_mask,
                                   cmap=FileFlags.input_mask,
                                   bexpmap=FileFlags.input_mask,
                                   srcmdl=FileFlags.input_mask,
                                   outfile=FileFlags.output_mask,
                                   outxml=FileFlags.output_mask),
                    **kwargs)
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

        print ("Merged %i sources into %s"%(len(srcs_to_merge), comp.getName()))
        if len(missing_sources) > 0:
            print ("Missed sources: ", missing_sources)

        print ("Writing output source map file %s" % args.outfile)
        like.logLike.saveSourceMaps(args.outfile, False, False)
        if args.gzip:
            os.system("gzip -9 %s" % args.outfile)

        print ("Writing output xml file %s" % args.outxml)
        like.writeXml(args.outxml)


class ConfigMaker_MergeSrcmaps(ConfigMaker):
    """Small class to generate configurations for this script

    This adds the following arguments:
    --comp     : binning component definition yaml file
    --data     : datset definition yaml file
    --irf_ver  : IRF verions string (e.g., 'V6')
    --sources  : Catalog model component definition yaml file'
    """
    default_options = dict(comp=diffuse_defaults.diffuse['binning_yaml'],
                           data=diffuse_defaults.diffuse['dataset_yaml'],
                           irf_ver=diffuse_defaults.diffuse['irf_ver'],
                           sources=diffuse_defaults.diffuse['catalog_comp_yaml'])

    def __init__(self, link, **kwargs):
        """C'tor
        """
        ConfigMaker.__init__(self, link,
                             options=kwargs.get('options',
                                                ConfigMaker_MergeSrcmaps.default_options.copy()))

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        input_config = {}
        job_configs = {}

        components = Component.build_from_yamlfile(args['comp'])
        NAME_FACTORY.update_base_dict(args['data'])
        ret_dict = make_catalog_comp_dict(sources=args['sources'], basedir='.')
        comp_info_dict = ret_dict['comp_info_dict']

        for split_ver, split_dict in comp_info_dict.items():
            for source_key, source_dict in split_dict.items():

                print (split_ver, source_key, source_dict.model_type)
                full_key = "%s_%s"%(split_ver, source_key)
                if source_dict.model_type != 'CompositeSource':
                    continue

                for comp in components:
                    zcut = "zmax%i" % comp.zmax
                    key = "%s_%s" % (source_key, comp.make_key('{ebin_name}_{evtype_name}'))
                    name_keys = dict(zcut=zcut,
                                     sourcekey=full_key,
                                     ebin=comp.ebin_name,
                                     psftype=comp.evtype_name,
                                     coordsys='GAL',
                                     irf_ver=args['irf_ver'])
                    nested_name_keys = dict(zcut=zcut,
                                            sourcekey=source_dict.catalog_info.catalog_name,
                                            ebin=comp.ebin_name,
                                            psftype=comp.evtype_name,
                                            coordsys='GAL',
                                            irf_ver=args['irf_ver'])
                    outfile = NAME_FACTORY.srcmaps(**name_keys)
                    
                    job_configs[key] = dict(srcmaps=NAME_FACTORY.srcmaps(**nested_name_keys),
                                            expcube=NAME_FACTORY.ltcube(**name_keys),
                                            irfs=NAME_FACTORY.irfs(**name_keys),
                                            bexpmap=NAME_FACTORY.bexpcube(**name_keys),
                                            srcmdl=NAME_FACTORY.srcmdl_xml(**name_keys),
                                            merged=source_key,
                                            outfile=outfile,
                                            outxml=NAME_FACTORY.nested_srcmdl_xml(**name_keys),
                                            logfile=outfile.replace('.fits', '.log'))

        output_config = {}
        return input_config, job_configs, output_config

def create_link_merge_srcmaps(**kwargs):
    """Build and return a `Link` object that can invoke GtAssembleModel"""
    gtmerge = GtMergeSourceMaps(**kwargs)
    return gtmerge.link

def create_sg_merge_srcmaps(**kwargs):
    """Build and return a ScatterGather object that can invoke this script"""
    gtmerge = GtMergeSourceMaps()
    link = gtmerge.link
    link.linkname = kwargs.pop('linkname', link.linkname)
    appname = kwargs.pop('appname', 'fermipy-merge-srcmaps-sg')

    lsf_args = {'W': 6000,
                'R': 'rhel60'}

    usage = "%s [options]"%(appname)
    description = "Prepare data for diffuse all-sky analysis"

    config_maker = ConfigMaker_MergeSrcmaps(link)
    lsf_sg = build_sg_from_link(link, config_maker,
                                lsf_args=lsf_args,
                                usage=usage,
                                description=description,
                                appname=appname,
                                **kwargs)
    return lsf_sg


def main_single():
    """Entry point for command line use for single job """
    gtsmp = GtMergeSourceMaps()
    gtsmp.run(sys.argv[1:])


def main_batch():
    """Entry point for command line use  for dispatching batch jobs """
    lsf_sg = create_sg_merge_srcmaps()
    lsf_sg(sys.argv)

if __name__ == '__main__':
    main_single()
