# Licensed under a 3-clause BSD style license - see LICENSE.rst
""" 
Merge source maps to build composite sources
"""
from __future__ import absolute_import, division, print_function

import os
import sys

import yaml
import argparse

from astropy.io import fits
from fermipy.skymap import HpxMap

from fermipy.jobs.scatter_gather import ConfigMaker
from fermipy.jobs.lsf_impl import build_sg_from_link
from fermipy.jobs.chain import Link
from fermipy.diffuse.binning import Component
from fermipy.diffuse.name_policy import NameFactory

NAME_FACTORY = NameFactory()


class GtAssembleSourceMaps(object):
    """Small class to assemple source map files for fermipy analysis.

    This is useful for re-merging after parallelizing source map creation.
    """

    def __init__(self):
        """C'tor
        """
        self.parser = GtAssembleSourceMaps._make_parser()
        self.link = GtAssembleSourceMaps._make_link()

    @staticmethod
    def _make_parser():
        """Make an argument parser for this class """
        usage = "gt_assemble_srcmaps.py [options]"
        description = "Copy source maps from the library to a analysis directory"

        parser = argparse.ArgumentParser(usage=usage, description=description)
        parser.add_argument('-i', '--input', type=str, default=None,
                            help='Input yaml file')
        parser.add_argument('--comp', type=str, default=None,
                            help='Component')
        parser.add_argument('--hpx_order', type=int, default=7,
                            help='Maximum healpix order')
        return parser

    @staticmethod
    def _make_link():
        link = Link('assemble-model',
                    appname='fermipy-assemble-model',
                    options=dict(input=None, comp=None, hpx_order=None),
                    flags=['gzip'],
                    input_file_args=['input'])
        return link

    @staticmethod
    def _copy_ccube(ccube, outsrcmap, hpx_order):
        """
        """
        try:
            hdulist_in = fits.open(ccube)
        except IOError:
            hdulist_in = fits.open("%s.gz" % ccube)

        hpx_order_in = hdulist_in[1].header['ORDER']

        if hpx_order_in > hpx_order:
            hpxmap = HpxMap.create_from_hdulist(hdulist_in)
            hpxmap_out = hpxmap.ud_grade(hpx_order, preserve_counts=True)
            hpxlist_out = hdulist_in
            hpxlist_out['SKYMAP'] = hpxmap_out.create_image_hdu()
            hpxlist_out.writeto(outsrcmap)
            return hpx_order
        else:
            os.system('cp %s.gz %s.gz' % (ccube, outsrcmap))
            os.system('gunzip -f %s.gz' % (outsrcmap))
        return None

    @staticmethod
    def _open_outsrcmap(outsrcmap):
        """
        """
        outhdulist = fits.open(outsrcmap, 'append')
        return outhdulist

    @staticmethod
    def _append_hdus(hdulist, srcmap_file, source_names, hpx_order):
        """
        """
        print ("  Extracting %i sources from %s" % (len(source_names), srcmap_file))
        try:
            hdulist_in = fits.open(srcmap_file)
        except IOError:
            try:
                hdulist_in = fits.open('%s.gz' % srcmap_file)
            except IOError:
                print ("  Missing file %s" % srcmap_file)
                return

        for source_name in source_names:
            sys.stdout.write('.')
            sys.stdout.flush()
            if hpx_order is None:
                hdulist.append(hdulist_in[source_name])
            else:
                try:
                    hpxmap = HpxMap.create_from_hdulist(hdulist_in, hdu=source_name)
                except IndexError:
                    print ("  Index error on source %s in file %s" % (source_name, srcmap_file))
                    continue
                hpxmap_out = hpxmap.ud_grade(hpx_order, preserve_counts=True)
                hdulist.append(hpxmap_out.create_image_hdu(name=source_name))

        hdulist.flush()
        hdulist_in.close()

    @staticmethod
    def _assemble_component(compname, compinfo, hpx_order):
        """
        """
        print ("Working on component %s" % compname)
        ccube = compinfo['ccube']
        outsrcmap = compinfo['outsrcmap']
        outsrcmap += '.fits'
        source_dict = compinfo['source_dict']

        hpx_order = GtAssembleSourceMaps._copy_ccube(ccube, outsrcmap, hpx_order)
        hdulist = GtAssembleSourceMaps._open_outsrcmap(outsrcmap)

        for comp_name in sorted(source_dict.keys()):
            source_info = source_dict[comp_name]
            source_names = source_info['source_names']
            srcmap_file = source_info['srcmap_file']
            GtAssembleSourceMaps._append_hdus(hdulist, srcmap_file,
                                              source_names, hpx_order)
        print ("Done")

    def run(self, argv):
        args = self.parse_args(argv)
        manifest = yaml.safe_load(open(args.input))

        key = args.comp
        value = manifest[key]
        GtAssembleSourceMaps._assemble_component(key, value, args.hpx_order)


class ConfigMaker_AssembleSrcmaps(ConfigMaker):
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
        --hpx_order: Maximum HEALPix order to use
       --irf_ver   : IRF verions string (e.g., 'V6')
        models     : Names of models to assemble source maps for
        """
        parser.add_argument('--comp', type=str, default=None,
                            help='Yaml file with component definitions')
        parser.add_argument('--data', type=str, default=None,
                            help='Yaml file with dataset definitions')
        parser.add_argument('--hpx_order', type=int, default=7,
                            help='Maximum HEALPix order to use')
        parser.add_argument('--irf_ver', type=str, default="V6",
                            help='IRF Version')
        parser.add_argument('models', nargs='+', help='Names of input models')

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

        for modelkey in args.models:
            manifest = os.path.join('analysis', 'model_%s' % modelkey,
                                    'srcmap_manifest_%s.yaml' % modelkey)
            for comp in components:
                zcut = "zmax%i" % comp.zmax
                key = comp.make_key('{ebin_name}_{evtype_name}')
                outfile = NAME_FACTORY.merged_srcmaps(modelkey=modelkey,
                                                      component=key,
                                                      coordsys='GAL',
                                                      irf_ver=args.irf_ver)
                logfile = outfile.replace('.fits', '.log')
                job_configs[key] = dict(input=manifest,
                                        comp=key,
                                        logfile=logfile)
        output_config = {}
        return input_config, job_configs, output_config


def build_scatter_gather():
    """Build and return a ScatterGather object that can invoke this script"""
    gtassemble = GtAssembleSourceMaps()
    link = gtassemble.link

    lsf_args = {'W': 1500,
                'R': 'rhel60'}

    usage = "fermipy-assemble-model-sg [options]"
    description = "Copy source maps from the library to a analysis directory"

    config_maker = ConfigMaker_AssembleSrcmaps(link)
    lsf_sg = build_sg_from_link(link, config_maker,
                                lsf_args=lsf_args,
                                usage=usage,
                                description=description)
    return lsf_sg


def main_single():
    """Entry point for command line use for single job """
    gtsmp = GtAssembleSourceMaps()
    gtsmp.run(sys.argv[1:])


def main_batch():
    """Entry point for command line use  for dispatching batch jobs """
    lsf_sg = build_scatter_gather()
    lsf_sg(sys.argv)

if __name__ == '__main__':
    main_single()
