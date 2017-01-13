# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Run gtsrcmaps for a single energy plane for a single source

This is useful to parallize the production of the source maps
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import argparse

import xml.etree.cElementTree as ElementTree

import BinnedAnalysis as BinnedAnalysis
import pyLikelihood as pyLike

from fermipy import utils
from fermipy.jobs.chain import FileFlags, Link
from fermipy.jobs.scatter_gather import ConfigMaker
from fermipy.jobs.lsf_impl import build_sg_from_link
from fermipy.diffuse.name_policy import NameFactory
from fermipy.diffuse.binning import Component
from fermipy.diffuse.diffuse_src_manager import make_diffuse_comp_info_dict
from fermipy.diffuse.source_factory import make_sources


NAME_FACTORY = NameFactory()
HPX_ORDER_TO_KSTEP = {5: -1, 6: -1, 7: -1, 8: 2, 9: 1}


class GtSrcmapPartial(object):
    """Small class to create srcmaps for only once source in a model,
    and optionally for only some of the energy layers.

    This is useful for parallelizing source map creation.
    """
    NULL_MODEL = 'srcmdls/null.xml'

    def __init__(self):
        """C'tor
        """
        self.parser = GtSrcmapPartial._make_parser()
        self.link = GtSrcmapPartial._make_link()

    @staticmethod
    def _make_parser():
        """Make an argument parser for this class """
        usage = "fermipy-srcmaps-diffuse [options]"
        description = "Run gtsrcmaps for one or more energy planes for a single source"

        parser = argparse.ArgumentParser(usage=usage, description=description)
        parser.add_argument('--irfs', type=str, default='CALDB',
                            help='Instrument response functions')
        parser.add_argument('--expcube', type=str, default=None,
                            help='Input Livetime cube file')
        parser.add_argument('--cmap', type=str, default=None,
                            help='Input counts map file')
        parser.add_argument('--bexpmap', type=str, default=None,
                            help='Input binned exposure map file')
        parser.add_argument('--srcmdl', type=str, default=None,
                            help='Input source model xml file')
        parser.add_argument('--source', type=str, default=None,
                            help='Input source')
        parser.add_argument('--outfile', type=str, default=None,
                            help='Output file')
        parser.add_argument('--kmin', type=int, default=0,
                            help='Minimum Energy Bin')
        parser.add_argument('--kmax', type=int, default=-1,
                            help='Maximum Energy Bin (-1 for all bins)')
        parser.add_argument('--gzip', action='store_true',
                            help='Compress output file')
        return parser

    @staticmethod
    def _make_link():
        link = Link('srcmaps-diffuse',
                    appname='fermipy-srcmaps-diffuse',
                    options=dict(irfs=('CALDB', 'IRFS to use', str),
                                 expcube=(None, "Input livetime cube file", str),
                                 bexpmap=(None, "Input binned exposure cube file", str),
                                 cmap=(None, "Input counts map file", str),
                                 srcmdl=(None, "Input source model xml file", str),
                                 source=(None, "Name of source", str),
                                 outfile=(None, "Output source map file", str),
                                 kmin=(0, "Minimum energy bin", int),
                                 kmin=(-1, "Maximum energy bin", int),
                                 gzip=(False, "Compress output", bool)),
                    file_args=dict(expcube=FileFlags.input_mask, 
                                  cmap=FileFlags.input_mask,
                                  bexpmap=FileFlags.input_mask,
                                  srcmdl=FileFlags.input_mask,
                                  outfile=FileFlags.output_mask))
        return link

    def run(self, argv):
        """Run this analysis"""
        args = self.parser.parse_args(argv)
        obs = BinnedAnalysis.BinnedObs(irfs=args.irfs,
                                       expCube=args.expcube,
                                       srcMaps=args.cmap,
                                       binnedExpMap=args.bexpmap)

        like = BinnedAnalysis.BinnedAnalysis(obs,
                                             optimizer='MINUIT',
                                             srcModel=GtSrcmapPartial.NULL_MODEL,
                                             wmap=None)

        source_factory = pyLike.SourceFactory(obs.observation)
        source_factory.readXml(args.srcmdl, BinnedAnalysis._funcFactory,
                               False, True, True)

        source = source_factory.releaseSource(args.source)

        try:
            diffuse_source = pyLike.DiffuseSource.cast(source)
            diffuse_source.mapBaseObject().projmap().setExtrapolation(False)
        except:
            pass

        like.logLike.saveSourceMap_partial(args.outfile, source, args.kmin, args.kmax)

        if args.gzip:
            os.system("gzip -9 %s" % args.outfile)


class ConfigMaker_SrcmapPartial(ConfigMaker):
    """Small class to generate configurations for this script
    
    This adds the following arguments:
    --comp     : binning component definition yaml file
    --data     : datset definition yaml file
    --irf_ver  : IRF verions string (e.g., 'V6')
    --diffuse  : Diffuse model component definition yaml file'
    --make_xml : Write xml files for the individual components
    """
    default_options = dict(comp=diffuse_defaults.diffuse['binning_yaml'],
                           data=diffuse_defaults.diffuse['dataset_yaml'],
                           irf_ver=diffuse_defaults.diffuse['irf_ver'],
                           diffuse=diffuse_defaults.diffuse['diffuse_comp_yaml'],
                           make_xml=(False, 'Write xml files needed to make source maps', bool))

    def __init__(self, link, **kwargs):
        """C'tor
        """
        ConfigMaker.__init__(self, link,
                             options=kwargs.get('options',default_options.copy()))

    @staticmethod
    def _write_xml(xmlfile, srcs):
        """Save the ROI model as an XML """
        root = ElementTree.Element('source_library')
        root.set('title', 'source_library')

        for src in srcs:
            src.write_xml(root)

        output_file = open(xmlfile, 'w')
        output_file.write(utils.prettify_xml(root))

    @staticmethod
    def _handle_component(sourcekey, comp_dict):
        """Make the source objects and write the xml for a component
        """
        if comp_dict.comp_key is None:
            fullkey = sourcekey
        else:
            fullkey = "%s_%s" % (sourcekey, comp_dict.comp_key)
        srcdict = make_sources(fullkey, comp_dict)
        if comp_dict.model_type == 'IsoSource':
            print ("Writing xml for %s to %s: %s %s" % (fullkey,
                                                        comp_dict.srcmdl_name,
                                                        comp_dict.model_type,
                                                        comp_dict.Spectral_Filename))
        elif comp_dict.model_type == 'MapCubeSource':
            print ("Writing xml for %s to %s: %s %s" % (fullkey,
                                                        comp_dict.srcmdl_name,
                                                        comp_dict.model_type,
                                                        comp_dict.Spatial_Filename))
        ConfigMaker_SrcmapPartial._write_xml(comp_dict.srcmdl_name, srcdict.values())

    @staticmethod
    def _make_xml_files(diffuse_comp_info_dict):
        """Make all the xml file for individual components
        """
        for sourcekey in sorted(diffuse_comp_info_dict.keys()):
            comp_info = diffuse_comp_info_dict[sourcekey]
            if comp_info.components is None:
                ConfigMaker_SrcmapPartial._handle_component(sourcekey, comp_info)
            else:
                for sub_comp_info in comp_info.components.values():
                    ConfigMaker_SrcmapPartial._handle_component(sourcekey, sub_comp_info)

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        input_config = {}
        job_configs = {}

        components = Component.build_from_yamlfile(args['comp'])
        NAME_FACTORY.update_base_dict(args['data'])

        ret_dict = make_diffuse_comp_info_dict(components=components,
                                               diffuse=args['diffuse'],
                                               basedir='.')
        diffuse_comp_info_dict = ret_dict['comp_info_dict']
        if args['make_xml']:
            ConfigMaker_SrcmapPartial._make_xml_files(diffuse_comp_info_dict)

        for diffuse_comp_info_key in sorted(diffuse_comp_info_dict.keys()):
            diffuse_comp_info_value = diffuse_comp_info_dict[diffuse_comp_info_key]
            for comp in components:
                zcut = "zmax%i" % comp.zmax
                key = comp.make_key('{ebin_name}_{evtype_name}')
                if diffuse_comp_info_value.components is None:
                    sub_comp_info = diffuse_comp_info_value
                else:
                    sub_comp_info = diffuse_comp_info_value.get_component_info(comp)
                name_keys = dict(zcut=zcut,
                                 sourcekey=sub_comp_info.sourcekey,
                                 ebin=comp.ebin_name,
                                 psftype=comp.evtype_name,
                                 coordsys='GAL',
                                 irf_ver=args['irf_ver'])

                kmin = 0
                kmax = comp.enumbins + 1
                outfile_base = NAME_FACTORY.srcmaps(**name_keys)
                kstep = HPX_ORDER_TO_KSTEP[comp.hpx_order]
                base_dict = dict(cmap=NAME_FACTORY.ccube(**name_keys),
                                 expcube=NAME_FACTORY.ltcube(**name_keys),
                                 irfs=NAME_FACTORY.irfs(**name_keys),
                                 bexpmap=NAME_FACTORY.bexpcube(**name_keys),
                                 srcmdl=sub_comp_info.srcmdl_name,
                                 source=sub_comp_info.source_name,
                                 evtype=comp.evtype)

                if kstep < 0:
                    kstep = kmax
                    for k in range(kmin, kmax, kstep):
                        full_key = "%s_%s_%02i" % (diffuse_comp_info_key, key, k)
                        khi = min(kmax, k + kstep)

                        full_dict = base_dict.copy()
                        full_dict.update(dict(outfile=\
                                                  outfile_base.replace('.fits', '_%02i.fits' % k),
                                              kmin=k, kmax=khi,
                                              logfile=\
                                                  outfile_base.replace('.fits', '_%02i.log' % k)))
                        job_configs[full_key] = full_dict

        output_config = {}
        return input_config, job_configs, output_config


def build_scatter_gather():
    """Build and return a ScatterGather object that can invoke this script"""
    gtsmp = GtSrcmapPartial()
    link = gtsmp.link

    lsf_args = {'W': 1500,
                'R': 'rhel60'}

    usage = "fermipy-srcmaps-diffuse-sg [options] input"
    description = "Build source maps for diffuse model components"

    config_maker = ConfigMaker_SrcmapPartial(link)
    lsf_sg = build_sg_from_link(link, config_maker,
                                lsf_args=lsf_args,
                                usage=usage,
                                description=description)
    return lsf_sg


def main_single():
    """Entry point for command line use for single job """
    gtsmp = GtSrcmapPartial()
    gtsmp.run(sys.argv[1:])


def main_batch():
    """Entry point for command line use  for dispatching batch jobs """
    lsf_sg = build_scatter_gather()
    lsf_sg(sys.argv)


if __name__ == '__main__':
    main_single()
