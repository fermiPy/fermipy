# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Module to collect configuration to run specific jobs
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import argparse

from fermipy.jobs.file_archive import FileFlags
from fermipy.jobs.chain import Link, Chain
from fermipy.jobs.gtlink import Gtlink
from fermipy.jobs.scatter_gather import ConfigMaker
from fermipy.jobs.lsf_impl import build_sg_from_link
from fermipy.diffuse.name_policy import NameFactory
from fermipy.diffuse.binning import Component
from fermipy.diffuse import defaults as diffuse_defaults

NAME_FACTORY = NameFactory()


def create_link_gtexphpsun(**kwargs):
    """Make a `fermipy.jobs.Gtlink` object to run gtexphpsun """
    gtlink = Gtlink(linkname=kwargs.pop('linkname', 'gtexphpsun'),
                    appname='gtexphpsun',
                    options=dict(irfs=diffuse_defaults.gtopts['irfs'],
                                 evtype=(3, "Event type selection", int),
                                 emin=(100., "Start energy (MeV) of first bin", float),
                                 emax=(1000000., "Stop energy (MeV) of last bin", float),
                                 enumbins=(12,"Number of logarithmically-spaced energy bins", int),
                                 binsz=(1.,"Image scale (in degrees/pixel)", float),                                           
                                 infile=(None, "Input livetime cube file", str),
                                 outfile=diffuse_defaults.gtopts['outfile']),
                    file_args=dict(infile=FileFlags.input_mask,
                                   outfile=FileFlags.output_mask),
                    **kwargs)
    return gtlink

def create_link_gtsuntemp(**kwargs):
    """Make a `fermipy.jobs.Gtlink` object to run gtsuntemp """
    gtlink = Gtlink(linkname=kwargs.pop('linkname', 'gtsuntemp'),
                    appname='gtsuntemp',
                    options=dict(expsun=(None, "Exposure binned in healpix and solar angles", str),
                                 avgexp=(None, "Binned exposure", str),
                                 sunprof=(None, "Fits file containing solar intensity profile", str),
                                 cmap=("none", "Counts map file", str),
                                 irfs=diffuse_defaults.gtopts['irfs'],
                                 evtype=(3, "Event type selection", int),
                                 coordsys=("GAL", "Coordinate system (CEL - celestial, GAL -galactic)", str),
                                 emin=(100., "Start energy (MeV) of first bin", float),
                                 emax=(1000000., "Stop energy (MeV) of last bin", float),
                                 enumbins=(12,"Number of logarithmically-spaced energy bins", int),
                                 nxpix=(1440, "Size of the X axis in pixels", int),
                                 nypix=(720, "Size of the Y axis in pixels", int),
                                 binsz=(0.25, "Image scale (in degrees/pixel)", float),
                                 xref=(0., "First coordinate of image center in degrees (RA or GLON)", float),
                                 yref=(0., "Second coordinate of image center in degrees (DEC or GLAT)", float),
                                 axisrot=(0., "Rotation angle of image axis, in degrees", float),
                                 proj=("CAR", "Projection method e.g. AIT|ARC|CAR|GLS|MER|NCP|SIN|STG|TAN", str),
                                 outfile=diffuse_defaults.gtopts['outfile']),
                    file_args=dict(expsun=FileFlags.input_mask,
                                   avgexp=FileFlags.input_mask,
                                   sunprof=FileFlags.input_mask,
                                   outfile=FileFlags.output_mask),
                    **kwargs)
    return gtlink





class ConfigMaker_Gtexphpsun(ConfigMaker):
    """Small class to generate configurations for gtexphpsun 

    This takes the following arguments:
    --comp     : binning component definition yaml file
    --data     : datset definition yaml file
    --irf_ver  : IRF verions string (e.g., 'V6')
    """
    default_options = dict(comp=diffuse_defaults.sun_moon['binning_yaml'],
                           data=diffuse_defaults.sun_moon['dataset_yaml'],
                           irf_ver=diffuse_defaults.sun_moon['irf_ver'])

    def __init__(self, link, **kwargs):
        """C'tor
        """
        ConfigMaker.__init__(self, link,
                             options=kwargs.get('options', self.default_options.copy()))

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        input_config = {}
        job_configs = {}

        components = Component.build_from_yamlfile(args['comp'])
        NAME_FACTORY.update_base_dict(args['data'])

        for comp in components:
            zcut = "zmax%i" % comp.zmax
            key = comp.make_key('{ebin_name}_{evtype_name}')
            name_keys = dict(zcut=zcut,
                             ebin=comp.ebin_name,
                             psftype=comp.evtype_name,
                             irf_ver=args['irf_ver'],
                             fullpath=True)
            outfile = NAME_FACTORY.bexpcube_sun(**name_keys)
            job_configs[key] = dict(infile=NAME_FACTORY.ltcube_sun(**name_keys),
                                    outfile=outfile,
                                    irfs=NAME_FACTORY.irfs(**name_keys),
                                    evtype=comp.evtype,
                                    emin=comp.emin,
                                    emax=comp.emax,
                                    enumbins=comp.enumbins,
                                    logfile=outfile.replace('.fits', '.log'))

        output_config = {}
        return input_config, job_configs, output_config


class ConfigMaker_Gtsuntemp(ConfigMaker):
    """Small class to generate configurations for gtsuntemp

    This takes the following arguments:
    --comp       : binning component definition yaml file
    --data       : datset definition yaml file
    --irf_ver    : IRF verions string (e.g., 'V6')
    --sourcekeys : Keys for sources to make template for
    """
    default_options = dict(comp=diffuse_defaults.sun_moon['binning_yaml'],
                           data=diffuse_defaults.sun_moon['dataset_yaml'],
                           sourcekeys=diffuse_defaults.sun_moon['sourcekeys'],
                           irf_ver=diffuse_defaults.sun_moon['irf_ver'])

    def __init__(self, link, **kwargs):
        """C'tor
        """
        ConfigMaker.__init__(self, link,
                             options=kwargs.get('options', self.default_options.copy()))

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        input_config = {}
        job_configs = {}

        components = Component.build_from_yamlfile(args['comp'])
        NAME_FACTORY.update_base_dict(args['data'])

        for comp in components:
            for sourcekey in args['sourcekeys']:
                zcut = "zmax%i" % comp.zmax
                key = comp.make_key('{ebin_name}_{evtype_name}') + "_%s"%sourcekey
                name_keys = dict(zcut=zcut,
                                 ebin=comp.ebin_name,
                                 psftype=comp.evtype_name,
                                 irf_ver=args['irf_ver'],
                                 sourcekey=sourcekey,
                                 fullpath=True)
                outfile = NAME_FACTORY.template_sunmoon(**name_keys)
                job_configs[key] = dict(expsun=NAME_FACTORY.bexpcube_sun(**name_keys),
                                        avgexp=NAME_FACTORY.bexpcube(**name_keys),
                                        sunprof=NAME_FACTORY.angprofile(**name_keys),
                                        cmap='none',
                                        outfile=outfile,
                                        irfs=NAME_FACTORY.irfs(**name_keys),
                                        evtype=comp.evtype,
                                        emin=comp.emin,
                                        emax=comp.emax,
                                        enumbins=comp.enumbins,
                                        logfile=outfile.replace('.fits', '.log'))

        output_config = {}
        return input_config, job_configs, output_config



def create_sg_Gtexphpsun(**kwargs):
    """Build and return a ScatterGather object that can invoke gtexphpsun"""
    appname = kwargs.pop('appname', 'fermipy-gtexphpsun-sg')
    link = create_link_gtexphpsun(**kwargs)
    linkname = kwargs.pop('linkname', link.linkname)

    lsf_args = {'W': 1500,
                'R': 'rhel60'}

    usage = "%s [options]"%(appname)
    description = "Run gtexpcube2 for a series of event types."

    config_maker = ConfigMaker_Gtexphpsun(link)
    lsf_sg = build_sg_from_link(link, config_maker,
                                lsf_args=lsf_args,
                                usage=usage,
                                description=description,
                                linkname=linkname,
                                appname=appname,
                                **kwargs)
    return lsf_sg


def create_sg_Gtsuntemp(**kwargs):
    """Build and return a ScatterGather object that can invoke gtexphpsun"""
    appname = kwargs.pop('appname', 'fermipy-gtsuntemp-sg')
    link = create_link_gtsuntemp(**kwargs)
    linkname = kwargs.pop('linkname', link.linkname)

    lsf_args = {'W': 1500,
                'R': 'rhel60'}

    usage = "%s [options]"%(appname)
    description = "Run gtexpcube2 for a series of event types."

    config_maker = ConfigMaker_Gtsuntemp(link)
    lsf_sg = build_sg_from_link(link, config_maker,
                                lsf_args=lsf_args,
                                usage=usage,
                                description=description,
                                linkname=linkname,
                                appname=appname,
                                **kwargs)
    return lsf_sg



class SunMoonChain(Chain):
    """Small class to construct sun and moon templates
    """    
    def __init__(self, linkname):
        """C'tor
        """
        link_gtexphpsun = create_sg_Gtexphpsun(linkname="%s.gtexphpsun"%linkname,
                                               mapping={'data':'dataset_yaml',
                                                        'comp':'binning_yaml'})
        link_gtsuntemp = create_sg_Gtsuntemp(linkname="%s.gtsuntemp"%linkname,
                                             mapping={'data':'dataset_yaml',
                                                      'comp':'binning_yaml'})
        options = diffuse_defaults.sun_moon.copy()
        options['dry_run'] = (False, 'Print commands but do not run', bool)
        Chain.__init__(self, linkname,
                       appname='FIXME',
                       links=[link_gtexphpsun, link_gtsuntemp],
                       options=options,
                       parser=SunMoonChain._make_parser())

    @staticmethod
    def _make_parser():
        """Make an argument parser for this chain """
        usage = "FIXME [options]"
        description = "Build sun and moon templates"

        parser = argparse.ArgumentParser(usage=usage, description=description)
        return parser
 
    def run_argparser(self, argv):
        """Initialize a link with a set of arguments using argparser
        """
        args = Link.run_argparser(self, argv)
        for link in self._links.values():
            link.run_link(stream=sys.stdout, dry_run=True)
        return args


def create_chain_sun_moon(**kwargs):
    """Build and return a `ResidualCRChain` object """
    ret_chain = SunMoonChain(linkname=kwargs.pop('linkname', 'SunMoon'))
    return ret_chain


def invoke_sg_Gtexphpsun():
    """Entry point for command line use for dispatching batch jobs """
    lsf_sg = create_sg_Gtexphpsun()
    lsf_sg(sys.argv)


def invoke_sg_Gtsuntemp():
    """Entry point for command line use for dispatching batch jobs """
    lsf_sg = create_sg_Gtsuntemp()
    lsf_sg(sys.argv)


def main_chain():
    """Energy point for running the entire Cosmic-ray analysis """
    the_chain = SunMoonChain('SunMoon')
    args = the_chain.run_argparser(sys.argv[1:])
    the_chain.run_chain(sys.stdout, args.dry_run)
    the_chain.finalize(args.dry_run)


if __name__ == '__main__':
    main_chain()
