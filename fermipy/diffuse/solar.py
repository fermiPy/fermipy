# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Module to collect configuration to run specific jobs
"""
from __future__ import absolute_import, division, print_function

from fermipy.utils import load_yaml
from fermipy.jobs.file_archive import FileFlags
from fermipy.jobs.chain import Chain
from fermipy.jobs.gtlink import Gtlink
from fermipy.jobs.scatter_gather import ScatterGather
from fermipy.jobs.slac_impl import make_nfs_path

from fermipy.diffuse.name_policy import NameFactory
from fermipy.diffuse.binning import Component
from fermipy.diffuse import defaults as diffuse_defaults

NAME_FACTORY = NameFactory()


class Gtlink_exphpsun(Gtlink):
    """Small wrapper to run gtexphpsun """

    appname = 'gtexphpsun'
    linkname_default = 'gtexphpsun'
    usage = '%s [options]' % (appname)
    description = "Link to run %s" % (appname)

    default_options = dict(irfs=diffuse_defaults.gtopts['irfs'],
                           evtype=(3, "Event type selection", int),
                           emin=(100., "Start energy (MeV) of first bin", float),
                           emax=(1000000., "Stop energy (MeV) of last bin", float),
                           enumbins=(12, "Number of logarithmically-spaced energy bins", int),
                           binsz=(1., "Image scale (in degrees/pixel)", float),
                           infile=(None, "Input livetime cube file", str),
                           outfile=diffuse_defaults.gtopts['outfile'])
    default_file_args = dict(infile=FileFlags.input_mask,
                             outfile=FileFlags.output_mask)


class Gtlink_suntemp(Gtlink):
    """Small wrapper to run gtsuntemp """

    appname = 'gtsuntemp'
    linkname_default = 'gtsuntemp'
    usage = '%s [options]' % (appname)
    description = "Link to run %s" % (appname)

    default_options = dict(expsun=(None, "Exposure binned in healpix and solar angles", str),
                           avgexp=(None, "Binned exposure", str),
                           sunprof=(None, "Fits file containing solar intensity profile", str),
                           cmap=("none", "Counts map file", str),
                           irfs=diffuse_defaults.gtopts['irfs'],
                           evtype=(3, "Event type selection", int),
                           coordsys=("GAL",
                                     "Coordinate system (CEL - celestial, GAL -galactic)", str),
                           emin=(100., "Start energy (MeV) of first bin", float),
                           emax=(1000000., "Stop energy (MeV) of last bin", float),
                           enumbins=(12, "Number of logarithmically-spaced energy bins", int),
                           nxpix=(1440, "Size of the X axis in pixels", int),
                           nypix=(720, "Size of the Y axis in pixels", int),
                           binsz=(0.25, "Image scale (in degrees/pixel)", float),
                           xref=(0.,
                                 "First coordinate of image center in degrees (RA or GLON)", float),
                           yref=(0.,
                                 "Second coordinate of image center in degrees (DEC or GLAT)",
                                 float),
                           axisrot=(0., "Rotation angle of image axis, in degrees", float),
                           proj=("CAR",
                                 "Projection method e.g. AIT|ARC|CAR|GLS|MER|NCP|SIN|STG|TAN", str),
                           outfile=diffuse_defaults.gtopts['outfile']),
    default_file_args = dict(expsun=FileFlags.input_mask,
                             avgexp=FileFlags.input_mask,
                             sunprof=FileFlags.input_mask,
                             outfile=FileFlags.output_mask)


class Gtexphpsun_SG(ScatterGather):
    """Small class to generate configurations for gtexphpsun

    This takes the following arguments:
    --comp     : binning component definition yaml file
    --data     : datset definition yaml file
    """
    appname = 'fermipy-gtexphpsun-sg'
    usage = "%s [options]" % (appname)
    description = "Submit gtexphpsun jobs in parallel"
    clientclass = Gtlink_exphpsun

    job_time = 300

    default_options = dict(comp=diffuse_defaults.diffuse['comp'],
                           data=diffuse_defaults.diffuse['data'])

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        components = Component.build_from_yamlfile(args['comp'])
        NAME_FACTORY.update_base_dict(args['data'])

        # FIXME
        mktime = 'nosm'

        for comp in components:
            zcut = "zmax%i" % comp.zmax
            key = comp.make_key('{ebin_name}_{evtype_name}')
            name_keys = dict(zcut=zcut,
                             ebin=comp.ebin_name,
                             psftype=comp.evtype_name,
                             irf_ver=NAME_FACTORY.irf_ver(),
                             mktime=mktime,
                             fullpath=True)
            outfile = NAME_FACTORY.bexpcube_sun(**name_keys)
            job_configs[key] = dict(infile=NAME_FACTORY.ltcube_sun(**name_keys),
                                    outfile=outfile,
                                    irfs=NAME_FACTORY.irfs(**name_keys),
                                    evtype=comp.evtype,
                                    emin=comp.emin,
                                    emax=comp.emax,
                                    enumbins=comp.enumbins,
                                    logfile=make_nfs_path(outfile.replace('.fits', '.log')))

        return job_configs


class Gtsuntemp_SG(ScatterGather):
    """Small class to generate configurations for gtsuntemp

    This takes the following arguments:
    --comp       : binning component definition yaml file
    --data       : datset definition yaml file
    --sourcekeys : Keys for sources to make template for
    """
    appname = 'fermipy-gtsuntemp-sg'
    usage = "%s [options]" % (appname)
    description = "Submit gtsuntemp jobs in parallel"
    clientclass = Gtlink_exphpsun

    job_time = 300

    default_options = dict(comp=diffuse_defaults.diffuse['comp'],
                           data=diffuse_defaults.diffuse['data'],
                           sourcekeys=diffuse_defaults.sun_moon['sourcekeys'])

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        components = Component.build_from_yamlfile(args['comp'])
        NAME_FACTORY.update_base_dict(args['data'])

        # FIXME
        mktime = 'nosm'

        for comp in components:
            for sourcekey in args['sourcekeys']:
                zcut = "zmax%i" % comp.zmax
                key = comp.make_key('{ebin_name}_{evtype_name}') + "_%s" % sourcekey
                name_keys = dict(zcut=zcut,
                                 ebin=comp.ebin_name,
                                 psftype=comp.evtype_name,
                                 irf_ver=NAME_FACTORY.irf_ver(),
                                 sourcekey=sourcekey,
                                 mktime=mktime,
                                 coordsys=comp.coordsys,
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

        return job_configs


class SunMoonChain(Chain):
    """Small class to construct sun and moon templates
    """
    appname = 'fermipy-sunmoon-chain'
    linkname_default = 'summoon'
    usage = '%s [options]' % (appname)
    description = 'Run sun and moon template construction'

    default_options = dict(config=diffuse_defaults.diffuse['config'])

    def __init__(self, **kwargs):
        """C'tor
        """
        super(SunMoonChain, self).__init__(**kwargs)
        self.comp_dict = None

    def _map_arguments(self, input_dict):
        """Map from the top-level arguments to the arguments provided to
        the indiviudal links """

        config_yaml = input_dict['config']
        config_dict = load_yaml(config_yaml)

        data = config_dict.get('data')
        comp = config_dict.get('comp')
        sourcekeys = config_dict.get('sourcekeys')

        self._set_link('exphpsun', Gtexphpsun_SG,
                             comp=comp, data=data)

        self._set_link('suntemp', Gtsuntemp_SG,
                             comp=comp, data=data,
                             sourcekeys=sourcekeys)



def register_classes():
    """Register these classes with the `LinkFactory` """
    Gtlink_exphpsun.register_class()
    Gtlink_suntemp.register_class()
    Gtexphpsun_SG.register_class()
    Gtsuntemp_SG.register_class()
    SunMoonChain.register_class()
