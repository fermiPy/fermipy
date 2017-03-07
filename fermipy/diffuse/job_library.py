# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Module to collect configuration to run specific jobs
"""
from __future__ import absolute_import, division, print_function

import os
import sys

from fermipy.jobs.file_archive import FileFlags
from fermipy.jobs.chain import Link
from fermipy.jobs.gtlink import Gtlink
from fermipy.jobs.scatter_gather import ConfigMaker
from fermipy.jobs.lsf_impl import build_sg_from_link
from fermipy.diffuse.name_policy import NameFactory
from fermipy.diffuse.binning import Component
from fermipy.diffuse.diffuse_src_manager import make_ring_dicts,\
    make_diffuse_comp_info_dict
from fermipy.diffuse.catalog_src_manager import make_catalog_comp_dict
from fermipy.diffuse import defaults as diffuse_defaults

NAME_FACTORY = NameFactory()


def create_link_gtexpcube2(**kwargs):
    """Make a `fermipy.jobs.Gtlink` object to run gtexpcube2  """
    gtlink = Gtlink(linkname=kwargs.pop('linkname', 'gtexpcube2'),
                    appname='gtexpcube2',
                    options=dict(irfs=diffuse_defaults.gtopts['irfs'],
                                 hpx_order=diffuse_defaults.gtopts['hpx_order'],
                                 infile=(None, "Input livetime cube file", str),
                                 cmap=diffuse_defaults.gtopts['cmap'],
                                 outfile=diffuse_defaults.gtopts['outfile'],
                                 coordsys=diffuse_defaults.gtopts['coordsys']),
                    file_args=dict(infile=FileFlags.input_mask,
                                   cmap=FileFlags.input_mask,
                                   outfile=FileFlags.output_mask),
                    **kwargs)
    return gtlink


def create_link_gtscrmaps(**kwargs):
    """Make a `fermipy.jobs.Gtlink` object to run gtsrcmaps  """
    gtlink = Gtlink(linkname=kwargs.pop('linkname', 'gtsrcmaps'),
                    appname='gtsrcmaps',
                    options=dict(irfs=diffuse_defaults.gtopts['irfs'],
                                 expcube=diffuse_defaults.gtopts['expcube'],
                                 bexpmap=diffuse_defaults.gtopts['bexpmap'],
                                 cmap=diffuse_defaults.gtopts['cmap'],
                                 srcmdl=diffuse_defaults.gtopts['srcmdl'],
                                 outfile=diffuse_defaults.gtopts['outfile']),
                    file_args=dict(expcube=FileFlags.input_mask,
                                   cmap=FileFlags.input_mask,
                                   bexpmap=FileFlags.input_mask,
                                   srcmdl=FileFlags.input_mask,
                                   outfile=FileFlags.output_mask),
                    **kwargs)
    return gtlink

def create_link_fermipy_coadd(**kwargs):
    """Make a `fermipy.jobs.Link` object to run fermipy-coadd  """
    link = Link(linkname=kwargs.pop('linkname', 'fermipy-coadd'),
                appname='fermipy-coadd',
                options=dict(args=([], "List of input files", list),
                             output=(None, "Output file", str)),
                file_args=dict(args=FileFlags.input_mask,
                               output=FileFlags.output_mask),
                **kwargs)
    return link

def create_link_fermipy_vstack(**kwargs):
    """Make a `fermipy.jobs.Link` object to run fermipy-vstack  """
    link = Link(linkname=kwargs.pop('linkname', 'fermipy-vstack'),
                appname='fermipy-vstack',
                options=dict(output=(None, "Output file name", str),
                             hdu=(None, "Name of HDU to stack", str),
                             args=([], "List of input files", list),
                             gzip=(False, "Compress output", bool)),
                file_args=dict(args=FileFlags.input_mask,
                               output=FileFlags.output_mask),
                **kwargs)
    return link

def create_link_fermipy_healview(**kwargs):
    """Make a `fermipy.jobs.Link` object to run fermipy-healview  """
    link = Link(linkname=kwargs.pop('linkname', 'fermipy-healview'),
                appname='fermipy-healview',
                options=dict(input=(None, "Input file", str),
                             output=(None, "Output file name", str),
                             extension=(None, "FITS HDU with HEALPix map", str),
                             zscale=("log", "Scaling for color scale", str)),
                file_args=dict(args=FileFlags.input_mask,
                               output=FileFlags.output_mask),
                **kwargs)
    return link


class ConfigMaker_Gtexpcube2(ConfigMaker):
    """Small class to generate configurations for gtexpcube2

    This takes the following arguments:
    --comp     : binning component definition yaml file
    --data     : datset definition yaml file
    --irf_ver  : IRF verions string (e.g., 'V6')
    --coordsys : Coordinate system ['GAL' | 'CEL']
    --hpx_order: HEALPix order parameter
    """
    default_options = dict(comp=diffuse_defaults.diffuse['binning_yaml'],
                           data=diffuse_defaults.diffuse['dataset_yaml'],
                           irf_ver=diffuse_defaults.diffuse['irf_ver'],
                           hpx_order_max=diffuse_defaults.diffuse['hpx_order_expcube'],
                           coordsys=diffuse_defaults.diffuse['coordsys'])

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
                             coordsys=args['coordsys'],
                             irf_ver=args['irf_ver'],
                             fullpath=True)
            outfile = NAME_FACTORY.bexpcube(**name_keys)
            job_configs[key] = dict(cmap=NAME_FACTORY.ccube(**name_keys),
                                    infile=NAME_FACTORY.ltcube(**name_keys),
                                    outfile=outfile,
                                    irfs=NAME_FACTORY.irfs(**name_keys),
                                    hpx_order=min(comp.hpx_order, args['hpx_order_max']),
                                    evtype=comp.evtype,
                                    logfile=outfile.replace('.fits', '.log'))

        output_config = {}
        return input_config, job_configs, output_config


class ConfigMaker_SrcmapsCatalog(ConfigMaker):
    """Small class to generate configurations for gtsrcmaps for catalog sources

    This takes the following arguments:
    --comp     : binning component definition yaml file
    --data     : datset definition yaml file
    --irf_ver  : IRF verions string (e.g., 'V6')
    --sources  : Yaml file with input source model definitions
    --make_xml : Write xml files for the individual components
    """
    default_options = dict(comp=diffuse_defaults.diffuse['binning_yaml'],
                           data=diffuse_defaults.diffuse['dataset_yaml'],
                           irf_ver=diffuse_defaults.diffuse['irf_ver'],
                           sources=diffuse_defaults.diffuse['catalog_comp_yaml'],
                           make_xml=(False, 'Write xml files needed to make source maps', bool),)

    def __init__(self, link, **kwargs):
        """C'tor
        """
        ConfigMaker.__init__(self, link,
                             options=kwargs.get('options',
                                                ConfigMaker_SrcmapsCatalog.default_options.copy()))
        self.link = link

    @staticmethod
    def _make_xml_files(catalog_info_dict, comp_info_dict):
        """Make all the xml file for individual components
        """
        for val in catalog_info_dict.values():
            print ("%s : %06i" % (val.srcmdl_name, len(val.roi_model.sources)))
            val.roi_model.write_xml(val.srcmdl_name)

        for val in comp_info_dict.values():
            for val2 in val.values():
                print ("%s : %06i" % (val2.srcmdl_name, len(val2.roi_model.sources)))
                val2.roi_model.write_xml(val2.srcmdl_name)

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        input_config = {}
        job_configs = {}

        components = Component.build_from_yamlfile(args['comp'])
        NAME_FACTORY.update_base_dict(args['data'])

        ret_dict = make_catalog_comp_dict(sources=args['sources'], basedir='.')
        catalog_info_dict = ret_dict['catalog_info_dict']
        comp_info_dict = ret_dict['comp_info_dict']

        if args['make_xml']:
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
                                 irf_ver=args['irf_ver'],
                                 fullpath=True)
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

    This takes the following arguments:
    --diffuse  : Diffuse model component definition yaml file
    --outdir   : Output directory
    """
    default_options = dict(diffuse=diffuse_defaults.diffuse['diffuse_comp_yaml'],
                           outdir=(None, 'Output directory', str),)

    def __init__(self, link, **kwargs):
        """C'tor
        """
        ConfigMaker.__init__(self, link,
                             options=kwargs.get('options',
                                                ConfigMaker_SumRings.default_options.copy()))

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        input_config = {}
        job_configs = {}

        gmm = make_ring_dicts(diffuse=args['diffuse'], basedir='.')

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

    This takes the following arguments:
    --comp     : binning component definition yaml file
    --data     : datset definition yaml file
    --irf_ver  : IRF verions string (e.g., 'V6')
    --diffuse  : Diffuse model component definition yaml file'
    """
    default_options = dict(comp=diffuse_defaults.diffuse['binning_yaml'],
                           data=diffuse_defaults.diffuse['dataset_yaml'],
                           irf_ver=diffuse_defaults.diffuse['irf_ver'],
                           diffuse=diffuse_defaults.diffuse['diffuse_comp_yaml'],)

    def __init__(self, link, **kwargs):
        """C'tor
        """
        ConfigMaker.__init__(self, link,
                             options=kwargs.get('options',
                                                ConfigMaker_Vstack.default_options.copy()))

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        input_config = {}
        job_configs = {}

        components = Component.build_from_yamlfile(args['comp'])
        NAME_FACTORY.update_base_dict(args['data'])

        ret_dict = make_diffuse_comp_info_dict(components=components,
                                               diffuse=args['diffuse'],
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
                                 irf_ver=args['irf_ver'],
                                 fullpath=True)

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


class ConfigMaker_healview(ConfigMaker):
    """Small class to generate configurations for fermipy-healview
    to display source maps

    This takes the following arguments:
    --comp     : binning component definition yaml file
    --data     : datset definition yaml file
    --irf_ver  : IRF verions string (e.g., 'V6')
    --diffuse  : Diffuse model component definition yaml file'
    """
    default_options = dict(comp=diffuse_defaults.diffuse['binning_yaml'],
                           data=diffuse_defaults.diffuse['dataset_yaml'],
                           irf_ver=diffuse_defaults.diffuse['irf_ver'],
                           diffuse=diffuse_defaults.diffuse['diffuse_comp_yaml'])

    def __init__(self, link, **kwargs):
        """C'tor
        """
        ConfigMaker.__init__(self, link,
                             options=kwargs.get('options',
                                                ConfigMaker_healview.default_options.copy()))

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        input_config = {}
        job_configs = {}

        components = Component.build_from_yamlfile(args['comp'])
        NAME_FACTORY.update_base_dict(args['data'])

        ret_dict = make_diffuse_comp_info_dict(components=components,
                                               diffuse=args['diffuse'],
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

                full_key = "%s_%s" % (sub_comp_info.sourcekey, key)

                name_keys = dict(zcut=zcut,
                                 sourcekey=sub_comp_info.sourcekey,
                                 ebin=comp.ebin_name,
                                 psftype=comp.evtype_name,
                                 coordsys='GAL',
                                 irf_ver=args['irf_ver'],
                                 fullpath=True)

                infile = NAME_FACTORY.srcmaps(**name_keys)
                outfile = infile.replace('.fits', '.png')

                job_configs[full_key] = dict(input=infile,
                                             output=outfile,
                                             extension=sub_comp_info.source_name,
                                             zscale=args.get('zscale', 'log'),
                                             logfile=outfile.replace('.png', '_png.log'))
                                            

        output_config = {}
        return input_config, job_configs, output_config


def create_sg_gtexpcube2(**kwargs):
    """Build and return a ScatterGather object that can invoke gtexpcube2"""
    appname = kwargs.pop('appname', 'fermipy-gtexcube2-sg')
    link = create_link_gtexpcube2(**kwargs)
    linkname = kwargs.pop('linkname', link.linkname)

    lsf_args = {'W': 1500,
                'R': 'rhel60'}

    usage = "%s [options]"%(appname)
    description = "Run gtexpcube2 for a series of event types."

    config_maker = ConfigMaker_Gtexpcube2(link)
    lsf_sg = build_sg_from_link(link, config_maker,
                                lsf_args=lsf_args,
                                usage=usage,
                                description=description,
                                linkname=linkname,
                                appname=appname,
                                **kwargs)
    return lsf_sg


def create_sg_gtsrcmaps_catalog(**kwargs):
    """Build and return a ScatterGather object that can invoke gtsrcmaps for catalog sources"""
    appname = kwargs.pop('appname', 'fermipy-srcmaps-catalog-sg')
    link = create_link_gtscrmaps(**kwargs)
    linkname=kwargs.pop('linkname', link.linkname)

    lsf_args = {'W': 6000,
                'R': 'rhel60'}

    usage = "%s [options]"%(appname)
    description = "Run gtsrcmaps for catalog sources"

    config_maker = ConfigMaker_SrcmapsCatalog(link)
    lsf_sg = build_sg_from_link(link, config_maker,
                                lsf_args=lsf_args,
                                usage=usage,
                                description=description,
                                linkname=linkname,
                                appname=appname,
                                **kwargs)
    return lsf_sg


def create_sg_sum_ring_gasmaps(**kwargs):
    """Build and return a ScatterGather object that can invoke fermipy-coadd"""
    appname = kwargs.pop('appname', 'fermipy-sum-ring-gasmaps-sg')
    link = create_link_fermipy_coadd(**kwargs)
    linkname = kwargs.pop('linkname', link.linkname)

    lsf_args = {'W': 1500,
                'R': 'rhel60'}

    usage = "%s [options]"%(appname)
    description = "Sum gasmaps to build diffuse model components"

    config_maker = ConfigMaker_SumRings(link)
    lsf_sg = build_sg_from_link(link, config_maker,
                                lsf_args=lsf_args,
                                usage=usage,
                                description=description,
                                linkname=linkname,
                                appname=appname,
                                **kwargs)
    return lsf_sg


def create_sg_vstack_diffuse(**kwargs):
    """Build and return a ScatterGather object that can invoke fermipy-vstack"""
    appname = kwargs.pop('appname', 'fermipy-vstack-diffuse-sg')
    link = create_link_fermipy_vstack(**kwargs)
    linkname = kwargs.pop('linkname', link.linkname)

    lsf_args = {'W': 1500,
                'R': 'rhel60'}

    usage = "%s [options]"%(appname)
    description = "Sum gasmaps to build diffuse model components"

    config_maker = ConfigMaker_Vstack(link)
    lsf_sg = build_sg_from_link(link, config_maker,
                                lsf_args=lsf_args,
                                usage=usage,
                                description=description,
                                linkname=linkname,
                                appname=appname,
                                **kwargs)
    return lsf_sg


def create_sg_healview_diffuse(**kwargs):
    """Build and return a ScatterGather object that can invoke fermipy-healview"""
    appname = kwargs.pop('appname', 'fermipy-healview-sg')
    link = create_link_fermipy_healview(**kwargs)
    linkname = kwargs.pop('linkname', link.linkname)

    lsf_args = {'W': 1500,
                'R': 'rhel60'}

    usage = "%s [options]"%(appname)
    description = "Sum gasmaps to build diffuse model components"

    config_maker = ConfigMaker_healview(link)
    lsf_sg = build_sg_from_link(link, config_maker,
                                lsf_args=lsf_args,
                                usage=usage,
                                description=description,
                                linkname=linkname,
                                appname=appname,
                                **kwargs)
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

def invoke_sg_healview_diffuse():
    """Entry point for command line use for dispatching batch jobs """
    lsf_sg = create_sg_healview_diffuse()
    lsf_sg(sys.argv)
