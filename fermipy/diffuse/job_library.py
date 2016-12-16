# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Module to collect configuration to run specific jobs
"""
from __future__ import absolute_import, division, print_function

import os
import sys

from fermipy.jobs.chain import Link
from fermipy.jobs.gtlink import Gtlink
from fermipy.jobs.scatter_gather import ConfigMaker
from fermipy.jobs.lsf_impl import build_sg_from_link
from fermipy.diffuse.name_policy import NameFactory
from fermipy.diffuse.binning import Component
from fermipy.diffuse.diffuse_src_manager import make_ring_dicts,\
    make_diffuse_comp_info_dict
from fermipy.diffuse.catalog_src_manager import make_catalog_comp_dict

NAME_FACTORY = NameFactory()

GTEXPCUBE2 = Gtlink('gtexpcube2',
                    options=dict(irfs='CALDB', hpx_order=6,
                                 infile=None, cmap=None,
                                 outfile=None, coordsys='GAL'),
                    input_file_args=['infile', 'cmap'],
                    output_file_args=['outfile'])

GTSRCMAPS = Gtlink('gtsrcmaps',
                   options=dict(irfs='CALDB', expcube=None,
                                bexpmap=None, cmap=None,
                                srcmdl=None, outfile=None),
                   flags=['gzip'],
                   input_file_args=['expcube', 'cmap', 'bexpmap', 'srcmdl'],
                   output_file_args=['outfile'])

FERMIPYCOADD = Link('fermipy-coadd',
                    appname='fermipy-coadd',
                    options=dict(args=[], output=None),
                    output_file_args=['output'])

FERMIPYVSTACK = Link('fermipy-vstack',
                     appname='fermipy-vstack',
                     options=dict(output=None, hdu=None, args=None,
                                  gzip=True),
                     flags=['gzip'],
                     input_file_args=[],
                     output_file_args=['output'])


class ConfigMaker_Gtexpcube2(ConfigMaker):
    """Small class to generate configurations for gtexpcube2"""

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
        --coordsys : Coordinate system ['GAL' | 'CEL']
        --hpx_order: HEALPix order parameter
        """
        parser.add_argument('--comp', type=str, default=None,
                            help='Yaml file with component definitions')
        parser.add_argument('--data', type=str, default=None,
                            help='Yaml file with dataset definitions')
        parser.add_argument('--irf_ver', type=str, default='V6',
                            help='Version of IRFs to use')
        parser.add_argument('--hpx_order', type=int, default=6,
                            help='HEALPix order parameter')
        parser.add_argument('--coordsys', type=str, default='CEL',
                            help='coordinate system')

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

        for comp in components:
            zcut = "zmax%i" % comp.zmax
            key = comp.make_key('{ebin_name}_{evtype_name}')
            name_keys = dict(zcut=zcut,
                             ebin=comp.ebin_name,
                             psftype=comp.evtype_name,
                             coordsys=args.coordsys,
                             irf_ver=args.irf_ver)
            outfile = NAME_FACTORY.bexpcube(**name_keys)
            job_configs[key] = dict(cmap=NAME_FACTORY.ccube(**name_keys).replace('.fits',
                                                                                 '.fits.gz'),
                                    infile=NAME_FACTORY.ltcube(**name_keys),
                                    outfile=outfile,
                                    irfs=NAME_FACTORY.irfs(**name_keys),
                                    hpx_order=min(comp.hpx_order, args.hpx_order),
                                    evtype=comp.evtype,
                                    logfile=outfile.replace('.fits', '.log'))

        output_config = {}
        return input_config, job_configs, output_config


class ConfigMaker_SrcmapsCatalog(ConfigMaker):
    """Small class to generate configurations for gtsrcmaps for catalog sources"""

    def __init__(self, link):
        """C'tor
        """
        ConfigMaker.__init__(self)
        self.link = link

    @staticmethod
    def _make_xml_files(catalog_info_dict, comp_info_dict):
        """Make all the xml file for individual components
        """
        for val in catalog_info_dict.values():
            print ("%s : %06i" % (val.srcmdl_name, len(val.roi_model.sources)))
            #val.roi_model.write_xml(val.srcmdl_name)

        for val in comp_info_dict.values():
            for val2 in val.values():
                print ("%s : %06i" % (val2.srcmdl_name, len(val2.roi_model.sources)))
                #val2.roi_model.write_xml(val2.srcmdl_name)

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
        --sources  : Yaml file with input source model definitions
        --make_xml : Write xml files for the individual components
        """
        parser.add_argument('--comp', type=str, default=None,
                            help='Yaml file with component definitions')
        parser.add_argument('--data', type=str, default=None,
                            help='Yaml file with dataset definitions')
        parser.add_argument('--irf_ver', type=str, default='V6',
                            help='Version of IRFs to use')
        parser.add_argument('--sources', type=str, default=None,
                            help='Yaml file with input source model definitions')
        parser.add_argument('--make_xml', action='store_true',
                            help='Write xml files needed to make source maps')

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
        catalog_info_dict = ret_dict['catalog_info_dict']
        comp_info_dict = ret_dict['comp_info_dict']

        if args.make_xml:
            ConfigMaker_SrcmapsCatalog._make_xml_files(catalog_info_dict, comp_info_dict)

        for catalog_name, catalog_info in catalog_info_dict.items():
            for comp in components:
                zcut = "zmax%i" % comp.zmax
                key = comp.make_key('{ebin_name}_{evtype_name}')
                name_keys = dict(zcut=zcut,
                                 sourcekey=catalog_name,
                                 ebin=comp.ebin_name,
                                 psftype=comp.evtype_name,
                                 coordsys='GAL',
                                 irf_ver=args.irf_ver)
                outfile = NAME_FACTORY.srcmaps(**name_keys)
                logfile = outfile.replace('.fits', '.log')
                job_configs[key] = dict(cmap=NAME_FACTORY.ccube(**name_keys),
                                        expcube=NAME_FACTORY.ltcube(**name_keys),
                                        irfs=NAME_FACTORY.irfs(**name_keys),
                                        bexpmap=NAME_FACTORY.bexpcube(**name_keys),
                                        outfile=outfile,
                                        logfile=logfile,
                                        srcmdl=catalog_info.srcmdl_name,
                                        evtype=comp.evtype)

        output_config = {}
        return input_config, job_configs, output_config


class ConfigMaker_SumRings(ConfigMaker):
    """Small class to generate configurations for fermipy-coadd
    to sum galprop ring gasmaps
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

        --diffuse  : Diffuse model component definition yaml file'
        --outdir   : Output directory
        """
        parser.add_argument('--diffuse', type=str, default=None,
                            help='Diffuse model component definition yaml file')
        parser.add_argument('--outdir', type=str, default=None,
                            help='Output directory')

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

        gmm = make_ring_dicts(diffuse=args.diffuse, basedir='.')

        for galkey in gmm.galkeys():
            ring_dict = gmm.ring_dict(galkey)
            for ring_key, ring_info in ring_dict.items():
                output_file = ring_info.merged_gasmap
                file_string = ""
                for fname in ring_info.files:
                    file_string += " %s" % fname
                job_configs[ring_key] = dict(output=output_file,
                                             args=file_string,
                                             logfile=output_file.replace('.fits', '.log'))

        output_config = {}
        return input_config, job_configs, output_config


class ConfigMaker_Vstack(ConfigMaker):
    """Small class to generate configurations for fermipy-vstack
    to merge source maps
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
        --diffuse  : Diffuse model component definition yaml file'
        """
        parser.add_argument('--comp', type=str, default=None,
                            help='Yaml file with component definitions')
        parser.add_argument('--data', type=str, default=None,
                            help='Yaml file with dataset definitions')
        parser.add_argument('--irf_ver', type=str, default='V6',
                            help='Version of IRFs to use')
        parser.add_argument('--diffuse', type=str, default=None,
                            help='Yaml file with input diffuse model definitions')

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

        ret_dict = make_diffuse_comp_info_dict(components=components,
                                               diffuse=args.diffuse,
                                               basedir=NAME_FACTORY.base_dict['basedir'])
        diffuse_comp_info_dict = ret_dict['comp_info_dict']

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
                                 irf_ver=args.irf_ver)

                outfile = NAME_FACTORY.srcmaps(**name_keys)
                outfile_tokens = os.path.splitext(outfile)
                infile_regexp = "%s_*.fits*" % outfile_tokens[0]
                full_key = "%s_%s" % (sub_comp_info.sourcekey, key)

                job_configs[full_key] = dict(output=outfile,
                                             args=infile_regexp,
                                             hdu=sub_comp_info.source_name,
                                             logfile=outfile.replace('.fits', '.log'))

        output_config = {}
        return input_config, job_configs, output_config


def create_sg_gtexpcube2():
    """Build and return a ScatterGather object that can invoke gtexpcube2"""
    link = GTEXPCUBE2

    lsf_args = {'W': 1500,
                'R': 'rhel60'}

    usage = "fermipy-gtexcube2-sg [options] input"
    description = "Run gtexpcube2 for a series of event types."

    config_maker = ConfigMaker_Gtexpcube2(link)
    lsf_sg = build_sg_from_link(link, config_maker,
                                lsf_args=lsf_args,
                                usage=usage,
                                description=description)
    return lsf_sg


def create_sg_gtsrcmaps_catalog():
    """Build and return a ScatterGather object that can invoke gtsrcmaps for catalog sources"""
    link = GTSRCMAPS

    lsf_args = {'W': 1500,
                'R': 'rhel60'}

    usage = "fermipy-srcmaps-catalog-sg [options] input"
    description = "Run gtsrcmaps for catalog sources"

    config_maker = ConfigMaker_SrcmapsCatalog(link)
    lsf_sg = build_sg_from_link(link, config_maker,
                                lsf_args=lsf_args,
                                usage=usage,
                                description=description)
    return lsf_sg


def create_sg_sum_ring_gasmaps():
    """Build and return a ScatterGather object that can invoke fermipy-coadd"""
    link = FERMIPYCOADD

    lsf_args = {'W': 1500,
                'R': 'rhel60'}

    usage = "fermipy-sum-ring-gasmaps-sg [options] input"
    description = "Sum gasmaps to build diffuse model components"

    config_maker = ConfigMaker_SumRings(link)
    lsf_sg = build_sg_from_link(link, config_maker,
                                lsf_args=lsf_args,
                                usage=usage,
                                description=description)
    return lsf_sg


def create_sg_vstack_diffuse():
    """Build and return a ScatterGather object that can invoke fermipy-vstack"""
    link = FERMIPYVSTACK

    lsf_args = {'W': 1500,
                'R': 'rhel60'}

    usage = "fermipy-vstack-diffuse-sg [options] input"
    description = "Sum gasmaps to build diffuse model components"

    config_maker = ConfigMaker_Vstack(link)
    lsf_sg = build_sg_from_link(link, config_maker,
                                lsf_args=lsf_args,
                                usage=usage,
                                description=description)
    return lsf_sg


def invoke_sg_gtexpcube2():
    """Entry point for command line use for dispatching batch jobs """
    lsf_sg = create_sg_gtexpcube2()
    lsf_sg(sys.argv)


def invoke_sg_gtsrcmaps_catalog():
    """Entry point for command line use for dispatching batch jobs """
    lsf_sg = create_sg_gtsrcmaps_catalog()
    lsf_sg(sys.argv)


def invoke_sg_sum_ring_gasmaps():
    """Entry point for command line use for dispatching batch jobs """
    lsf_sg = create_sg_sum_ring_gasmaps()
    lsf_sg(sys.argv)


def invoke_sg_vstack_diffuse():
    """Entry point for command line use for dispatching batch jobs """
    lsf_sg = create_sg_vstack_diffuse()
    lsf_sg(sys.argv)
