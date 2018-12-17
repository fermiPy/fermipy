# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Module to collect configuration to run specific jobs
"""
from __future__ import absolute_import, division, print_function

from fermipy.utils import load_yaml
from fermipy.jobs.file_archive import FileFlags
from fermipy.jobs.chain import Chain
from fermipy.jobs.link import Link
from fermipy.jobs.gtlink import Gtlink
from fermipy.jobs.scatter_gather import ScatterGather
from fermipy.jobs.slac_impl import make_nfs_path

from fermipy.diffuse.name_policy import NameFactory
from fermipy.diffuse.binning import Component
from fermipy.diffuse import defaults as diffuse_defaults

NAME_FACTORY = NameFactory()

class Gtlink_expcube2_wcs(Gtlink):
    """Small wrapper to run gtexpcube2 """

    appname = 'gtexpcube2'
    linkname_default = 'gtexpcube2'
    usage = '%s [options]' % (appname)
    description = "Link to run %s" % (appname)

    default_options = dict(irfs=diffuse_defaults.gtopts['irfs'],
                           evtype=diffuse_defaults.gtopts['evtype'],
                           cmap=('none', "Input counts cube template", str),
                           emin=(100., "Start energy (MeV) of first bin", float),
                           emax=(1000000., "Stop energy (MeV) of last bin", float),
                           enumbins=(12, "Number of logarithmically-spaced energy bins", int),
                           binsz=(0.25, "Image scale (in degrees/pixel)", float),
                           xref=(0.,
                                 "First coordinate of image center in degrees (RA or GLON)", float),
                           yref=(0.,
                                 "Second coordinate of image center in degrees (DEC or GLAT)",
                                 float),
                           axisrot=(0., "Rotation angle of image axis, in degrees", float),
                           proj=("CAR",
                                 "Projection method e.g. AIT|ARC|CAR|GLS|MER|NCP|SIN|STG|TAN", str),
                           nxpix=(1440, "Size of the X axis in pixels", int),
                           nypix=(720, "Size of the Y axis in pixels", int),
                           infile=(None, "Input livetime cube file", str),
                           outfile=diffuse_defaults.gtopts['outfile'],
                           coordsys=('GAL', "Coordinate system", str))
    default_file_args = dict(infile=FileFlags.input_mask,
                             cmap=FileFlags.input_mask,
                             outfile=FileFlags.output_mask)

    __doc__ += Link.construct_docstring(default_options)

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

    __doc__ += Link.construct_docstring(default_options)


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
                           outfile=diffuse_defaults.gtopts['outfile'])
    default_file_args = dict(expsun=FileFlags.input_mask,
                             avgexp=FileFlags.input_mask,
                             sunprof=FileFlags.input_mask,
                             outfile=FileFlags.output_mask)

    __doc__ += Link.construct_docstring(default_options)


class Gtexpcube2wcs_SG(ScatterGather):
    """Small class to generate configurations for gtexphpsun

    """
    appname = 'fermipy-gtexpcube2wcs-sg'
    usage = "%s [options]" % (appname)
    description = "Submit gtexpcube2 jobs in parallel"
    clientclass = Gtlink_expcube2_wcs

    job_time = 300

    default_options = dict(comp=diffuse_defaults.diffuse['comp'],
                           data=diffuse_defaults.diffuse['data'],
                           mktimefilter=diffuse_defaults.diffuse['mktimefilter'],
                           binsz=(1.0, "Image scale (in degrees/pixel)", float),
                           nxpix=(360, "Size of the X axis in pixels", int),
                           nypix=(180, "Size of the Y axis in pixels", int))

    __doc__ += Link.construct_docstring(default_options)

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        components = Component.build_from_yamlfile(args['comp'])
        NAME_FACTORY.update_base_dict(args['data'])

        mktime = args['mktimefilter']

        base_config = dict(nxpix=args['nxpix'],
                           nypix=args['nypix'],
                           binsz=args['binsz'])

        for comp in components:
            zcut = "zmax%i" % comp.zmax
            key = comp.make_key('{ebin_name}_{evtype_name}')
            name_keys = dict(zcut=zcut,
                             ebin=comp.ebin_name,
                             psftype=comp.evtype_name,
                             irf_ver=NAME_FACTORY.irf_ver(),
                             coordsys=comp.coordsys,
                             mktime=mktime,
                             fullpath=True)
            outfile = NAME_FACTORY.bexpcube(**name_keys)
            ltcube = NAME_FACTORY.ltcube(**name_keys)
            full_config = base_config.copy()
            full_config.update(dict(infile=ltcube,
                                    outfile=outfile,
                                    irfs=NAME_FACTORY.irfs(**name_keys),
                                    evtype=comp.evtype,
                                    emin=comp.emin,
                                    emax=comp.emax,
                                    enumbins=comp.enumbins,
                                    logfile=make_nfs_path(outfile.replace('.fits', '.log'))))
            job_configs[key] = full_config

        return job_configs


class Gtexphpsun_SG(ScatterGather):
    """Small class to generate configurations for gtexphpsun

    """
    appname = 'fermipy-gtexphpsun-sg'
    usage = "%s [options]" % (appname)
    description = "Submit gtexphpsun jobs in parallel"
    clientclass = Gtlink_exphpsun

    job_time = 300

    default_options = dict(comp=diffuse_defaults.diffuse['comp'],
                           data=diffuse_defaults.diffuse['data'],
                           mktimefilter=diffuse_defaults.diffuse['mktimefilter'])

    __doc__ += Link.construct_docstring(default_options)

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        components = Component.build_from_yamlfile(args['comp'])
        NAME_FACTORY.update_base_dict(args['data'])

        mktime = args['mktimefilter']

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
            ltcube_sun = NAME_FACTORY.ltcube_sun(**name_keys)
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

    """
    appname = 'fermipy-gtsuntemp-sg'
    usage = "%s [options]" % (appname)
    description = "Submit gtsuntemp jobs in parallel"
    clientclass = Gtlink_suntemp

    job_time = 300

    default_options = dict(comp=diffuse_defaults.diffuse['comp'],
                           data=diffuse_defaults.diffuse['data'],
                           mktimefilter=diffuse_defaults.diffuse['mktimefilter'],
                           sourcekeys=diffuse_defaults.sun_moon['sourcekeys'])

    __doc__ += Link.construct_docstring(default_options)

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        components = Component.build_from_yamlfile(args['comp'])
        NAME_FACTORY.update_base_dict(args['data'])

        # FIXME
        mktime = args['mktimefilter']

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
    """Chain to construct sun and moon templates

    This chain consists of:

    exphpsun :  `Gtexphpsun_SG`
        Build the sun-centered exposure cubes

    suntemp : `Gtsuntemp_SG`
        Build the templates

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

        mktimefilter = config_dict.get('mktimefilter')

        self._set_link('expcube2', Gtexpcube2wcs_SG,
                       comp=comp, data=data,
                       mktimefilter=mktimefilter)

        self._set_link('exphpsun', Gtexphpsun_SG,
                       comp=comp, data=data,
                       mktimefilter=mktimefilter)

        self._set_link('suntemp', Gtsuntemp_SG,
                       comp=comp, data=data,
                       mktimefilter=mktimefilter,
                       sourcekeys=sourcekeys)



def register_classes():
    """Register these classes with the `LinkFactory` """
    Gtlink_exphpsun.register_class()
    Gtlink_suntemp.register_class()
    Gtexphpsun_SG.register_class()
    Gtsuntemp_SG.register_class()
    SunMoonChain.register_class()
