# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Module to collect configuration to run specific jobs
"""
from __future__ import absolute_import, division, print_function

import os
import copy

from fermipy.jobs.file_archive import FileFlags
from fermipy.jobs.link import Link
from fermipy.jobs.gtlink import Gtlink
from fermipy.jobs.app_link import AppLink
from fermipy.jobs.scatter_gather import ScatterGather
from fermipy.jobs.slac_impl import make_nfs_path
from fermipy.diffuse.utils import create_inputlist
from fermipy.diffuse.name_policy import NameFactory
from fermipy.diffuse.binning import Component
from fermipy.diffuse.diffuse_src_manager import make_ring_dicts,\
    make_diffuse_comp_info_dict
from fermipy.diffuse.catalog_src_manager import make_catalog_comp_dict
from fermipy.diffuse import defaults as diffuse_defaults

NAME_FACTORY = NameFactory()


def _make_ltcube_file_list(ltsumfile, num_files):
    """Make the list of input files for a particular energy bin X psf type """
    outbasename = os.path.basename(ltsumfile)
    lt_list_file = ltsumfile.replace('fits', 'lst')
    outfile = open(lt_list_file, 'w')
    for i in range(num_files):
        split_key = "%06i" % i
        output_dir = os.path.join(NAME_FACTORY.base_dict['basedir'], 'counts_cubes', split_key)
        filepath = os.path.join(output_dir, outbasename.replace('.fits', '_%s.fits' % split_key))
        outfile.write(filepath)
        outfile.write("\n")
    outfile.close()
    return '@' + lt_list_file


class Gtlink_select(Gtlink):
    """Small wrapper to run gtselect """

    appname = 'gtselect'
    linkname_default = 'gtselect'
    usage = '%s [options]' % (appname)
    description = "Link to run %s" % (appname)

    default_options = dict(emin=diffuse_defaults.gtopts['emin'],
                           emax=diffuse_defaults.gtopts['emax'],
                           infile=diffuse_defaults.gtopts['infile'],
                           outfile=diffuse_defaults.gtopts['outfile'],
                           zmax=diffuse_defaults.gtopts['zmax'],
                           evclass=diffuse_defaults.gtopts['evclass'],
                           evtype=diffuse_defaults.gtopts['evtype'],
                           pfiles=diffuse_defaults.gtopts['pfiles'])

    default_file_args = dict(infile=FileFlags.input_mask,
                             outfile=FileFlags.output_mask)

    __doc__ += Link.construct_docstring(default_options)


class Gtlink_bin(Gtlink):
    """Small wrapper to run gtbin """

    appname = 'gtbin'
    linkname_default = 'gtbin'
    usage = '%s [options]' % (appname)
    description = "Link to run %s" % (appname)

    default_options = dict(algorithm=('HEALPIX', "Binning alogrithm", str),
                           coordsys=diffuse_defaults.gtopts['coordsys'],
                           hpx_order=diffuse_defaults.gtopts['hpx_order'],
                           evfile=diffuse_defaults.gtopts['evfile'],
                           outfile=diffuse_defaults.gtopts['outfile'],
                           emin=diffuse_defaults.gtopts['emin'],
                           emax=diffuse_defaults.gtopts['emax'],
                           enumbins=diffuse_defaults.gtopts['enumbins'],
                           pfiles=diffuse_defaults.gtopts['pfiles'])

    default_file_args = dict(evfile=FileFlags.in_stage_mask,
                             outfile=FileFlags.out_stage_mask)

    __doc__ += Link.construct_docstring(default_options)

class Gtlink_expcube2(Gtlink):
    """Small wrapper to run gtexpcube2 """

    appname = 'gtexpcube2'
    linkname_default = 'gtexpcube2'
    usage = '%s [options]' % (appname)
    description = "Link to run %s" % (appname)

    default_options = dict(irfs=diffuse_defaults.gtopts['irfs'],
                           evtype=diffuse_defaults.gtopts['evtype'],
                           hpx_order=diffuse_defaults.gtopts['hpx_order'],
                           infile=(None, "Input livetime cube file", str),
                           cmap=diffuse_defaults.gtopts['cmap'],
                           outfile=diffuse_defaults.gtopts['outfile'],
                           coordsys=('GAL', "Coordinate system", str))
    default_file_args = dict(infile=FileFlags.input_mask,
                             cmap=FileFlags.input_mask,
                             outfile=FileFlags.output_mask)

    __doc__ += Link.construct_docstring(default_options)

class Gtlink_scrmaps(Gtlink):
    """Small wrapper to run gtscrmaps """

    appname = 'gtscrmaps'
    linkname_default = 'gtscrmaps'
    usage = '%s [options]' % (appname)
    description = "Link to run %s" % (appname)

    default_options = dict(irfs=diffuse_defaults.gtopts['irfs'],
                           expcube=diffuse_defaults.gtopts['expcube'],
                           bexpmap=diffuse_defaults.gtopts['bexpmap'],
                           cmap=diffuse_defaults.gtopts['cmap'],
                           srcmdl=diffuse_defaults.gtopts['srcmdl'],
                           outfile=diffuse_defaults.gtopts['outfile'])

    default_file_args = dict(expcube=FileFlags.input_mask,
                             cmap=FileFlags.input_mask,
                             bexpmap=FileFlags.input_mask,
                             srcmdl=FileFlags.input_mask,
                             outfile=FileFlags.output_mask)

    __doc__ += Link.construct_docstring(default_options)

class Gtlink_ltsum(Gtlink):
    """Small wrapper to run gtltsum """

    appname = 'gtltsum'
    linkname_default = 'gtltsum'
    usage = '%s [options]' % (appname)
    description = "Link to run %s" % (appname)

    default_options = dict(infile1=(None, "Livetime cube 1 or list of files", str),
                           infile2=("none", "Livetime cube 2", str),
                           outfile=(None, "Output file", str))
    default_file_args = dict(infile1=FileFlags.input_mask,
                             outfile=FileFlags.output_mask)

    __doc__ += Link.construct_docstring(default_options)

class Gtlink_mktime(Gtlink):
    """Small wrapper to run gtmktime """

    appname = 'gtmktime'
    linkname_default = 'gtmktime'
    usage = '%s [options]' % (appname)
    description = "Link to run %s" % (appname)

    default_options = dict(evfile=(None, 'Input FT1 File', str),
                           outfile=(None, 'Output FT1 File', str),
                           scfile=(None, 'Input FT2 file', str),
                           roicut=(False, 'Apply ROI-based zenith angle cut', bool),
                           filter=(None, 'Filter expression', str),
                           pfiles=(None, "PFILES directory", str))

    default_file_args = dict(evfile=FileFlags.in_stage_mask,
                             scfile=FileFlags.in_stage_mask,
                             outfile=FileFlags.out_stage_mask)

    __doc__ += Link.construct_docstring(default_options)

class Gtlink_ltcube(Gtlink):
    """Small wrapper to run gtltcube """

    appname = 'gtltcube'
    linkname_default = 'gtltcube'
    usage = '%s [options]' % (appname)
    description = "Link to run %s" % (appname)

    default_options = dict(evfile=(None, 'Input FT1 File', str),
                           scfile=(None, 'Input FT2 file', str),
                           outfile=(None, 'Output Livetime cube File', str),
                           dcostheta=(0.025, 'Step size in cos(theta)', float),
                           binsz=(1., 'Pixel size (degrees)', float),
                           phibins=(0, 'Number of phi bins', int),
                           zmin=(0, 'Minimum zenith angle', float),
                           zmax=(105, 'Maximum zenith angle', float),
                           pfiles=(None, "PFILES directory", str))

    default_file_args = dict(evfile=FileFlags.in_stage_mask,
                             scfile=FileFlags.in_stage_mask,
                             outfile=FileFlags.out_stage_mask)

    __doc__ += Link.construct_docstring(default_options)

class Link_FermipyCoadd(AppLink):
    """Small wrapper to run fermipy-coadd """

    appname = 'fermipy-coadd'
    linkname_default = 'coadd'
    usage = '%s [options]' % (appname)
    description = "Link to run %s" % (appname)

    default_options = dict(args=([], "List of input files", list),
                           output=(None, "Output file", str))
    default_file_args = dict(args=FileFlags.input_mask,
                             output=FileFlags.output_mask)

    __doc__ += Link.construct_docstring(default_options)

class Link_FermipyGatherSrcmaps(AppLink):
    """Small wrapper to run fermipy-gather-srcmaps """

    appname = 'fermipy-gather-srcmaps'
    linkname_default = 'gather-srcmaps'
    usage = '%s [options]' % (appname)
    description = "Link to run %s" % (appname)

    default_options = dict(output=(None, "Output file name", str),
                           args=([], "List of input files", list),
                           gzip=(False, "Compress output", bool),
                           rm=(False, "Remove input files", bool),
                           clobber=(False, "Overwrite output", bool))
    default_file_args = dict(args=FileFlags.input_mask,
                             output=FileFlags.output_mask)

    __doc__ += Link.construct_docstring(default_options)

class Link_FermipyVstack(AppLink):
    """Small wrapper to run fermipy-vstack """

    appname = 'fermipy-vstack'
    linkname_default = 'vstack'
    usage = '%s [options]' % (appname)
    description = "Link to run %s" % (appname)

    default_options = dict(output=(None, "Output file name", str),
                           hdu=(None, "Name of HDU to stack", str),
                           args=([], "List of input files", list),
                           gzip=(False, "Compress output", bool),
                           rm=(False, "Remove input files", bool),
                           clobber=(False, "Overwrite output", bool))
    default_file_args = dict(args=FileFlags.input_mask,
                             output=FileFlags.output_mask)

    __doc__ += Link.construct_docstring(default_options)

class Link_FermipyHealview(AppLink):
    """Small wrapper to run fermipy-healview """

    appname = 'fermipy-healview'
    linkname_default = 'fermipy-healview'
    usage = '%s [options]' % (appname)
    description = "Link to run %s" % (appname)

    default_options = dict(input=(None, "Input file", str),
                           output=(None, "Output file name", str),
                           extension=(None, "FITS HDU with HEALPix map", str),
                           zscale=("log", "Scaling for color scale", str))
    default_file_args = dict(args=FileFlags.input_mask,
                             output=FileFlags.output_mask)

    __doc__ += Link.construct_docstring(default_options)

class Gtexpcube2_SG(ScatterGather):
    """Small class to generate configurations for `Gtlink_expcube2`

    """
    appname = 'fermipy-gtexcube2-sg'
    usage = "%s [options]" % (appname)
    description = "Submit gtexpube2 jobs in parallel"
    clientclass = Gtlink_expcube2

    job_time = 300

    default_options = dict(comp=diffuse_defaults.diffuse['comp'],
                           data=diffuse_defaults.diffuse['data'],
                           hpx_order_max=diffuse_defaults.diffuse['hpx_order_expcube'])

    __doc__ += Link.construct_docstring(default_options)

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        components = Component.build_from_yamlfile(args['comp'])
        datafile = args['data']
        if datafile is None or datafile == 'None':
            return job_configs
        NAME_FACTORY.update_base_dict(args['data'])

        for comp in components:
            zcut = "zmax%i" % comp.zmax

            mktimelist = copy.copy(comp.mktimefilters)
            if not mktimelist:
                mktimelist.append('none')
            evtclasslist_keys = copy.copy(comp.evtclasses)
            if not evtclasslist_keys:
                evtclasslist_vals = [NAME_FACTORY.base_dict['evclass']]
            else:
                evtclasslist_vals = copy.copy(evtclasslist_keys)

            for mktimekey in mktimelist:
                for evtclassval in evtclasslist_vals:
                    fullkey = comp.make_key(
                        '%s_%s_{ebin_name}_%s_{evtype_name}' %
                        (evtclassval, zcut, mktimekey))
                    name_keys = dict(zcut=zcut,
                                     ebin=comp.ebin_name,
                                     psftype=comp.evtype_name,
                                     coordsys=comp.coordsys,
                                     irf_ver=NAME_FACTORY.irf_ver(),
                                     mktime=mktimekey,
                                     evclass=evtclassval,
                                     fullpath=True)

                    outfile = NAME_FACTORY.bexpcube(**name_keys)
                    cmap = NAME_FACTORY.ccube(**name_keys)
                    infile = NAME_FACTORY.ltcube(**name_keys)
                    logfile = make_nfs_path(outfile.replace('.fits', '.log'))
                    job_configs[fullkey] = dict(cmap=cmap,
                                                infile=infile,
                                                outfile=outfile,
                                                irfs=NAME_FACTORY.irfs(**name_keys),
                                                hpx_order=min(
                                                    comp.hpx_order, args['hpx_order_max']),
                                                evtype=comp.evtype,
                                                logfile=logfile)

        return job_configs


class Gtltsum_SG(ScatterGather):
    """Small class to generate configurations for `Gtlink_ltsum`

    """
    appname = 'fermipy-gtltsum-sg'
    usage = "%s [options]" % (appname)
    description = "Submit gtltsum jobs in parallel"
    clientclass = Gtlink_ltsum

    job_time = 300

    default_options = dict(comp=diffuse_defaults.diffuse['comp'],
                           data=diffuse_defaults.diffuse['data'],
                           ft1file=(None, 'Input FT1 file', str))

    __doc__ += Link.construct_docstring(default_options)

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        components = Component.build_from_yamlfile(args['comp'])
        datafile = args['data']
        if datafile is None or datafile == 'None':
            return job_configs
        NAME_FACTORY.update_base_dict(args['data'])

        inputfiles = create_inputlist(args['ft1file'])
        num_files = len(inputfiles)

        for comp in components:
            zcut = "zmax%i" % comp.zmax

            mktimelist = copy.copy(comp.mktimefilters)
            if not mktimelist:
                mktimelist.append('none')
            evtclasslist_keys = copy.copy(comp.evtclasses)
            if not evtclasslist_keys:
                evtclasslist_vals = [NAME_FACTORY.base_dict['evclass']]
            else:
                evtclasslist_vals = copy.copy(evtclasslist_keys)

            for mktimekey in mktimelist:
                for evtclassval in evtclasslist_vals:
                    fullkey = comp.make_key(
                        '%s_%s_{ebin_name}_%s_{evtype_name}' %
                        (evtclassval, zcut, mktimekey))

                    name_keys = dict(zcut=zcut,
                                     ebin=comp.ebin_name,
                                     psftype=comp.evtype_name,
                                     coordsys=comp.coordsys,
                                     irf_ver=NAME_FACTORY.irf_ver(),
                                     mktime=mktimekey,
                                     evclass=evtclassval,
                                     fullpath=True)

                    outfile = os.path.join(NAME_FACTORY.base_dict['basedir'],
                                           NAME_FACTORY.ltcube(**name_keys))
                    infile1 = _make_ltcube_file_list(outfile, num_files)
                    logfile = make_nfs_path(outfile.replace('.fits', '.log'))
                    job_configs[fullkey] = dict(infile1=infile1,
                                                outfile=outfile,
                                                logfile=logfile)

        return job_configs


class SumRings_SG(ScatterGather):
    """Small class to generate configurations for `Link_FermipyCoadd`
    to sum galprop ring gasmaps
    
    """
    appname = 'fermipy-sum-rings-sg'
    usage = "%s [options]" % (appname)
    description = "Submit fermipy-coadd jobs in parallel to sum GALProp rings"
    clientclass = Link_FermipyCoadd

    job_time = 300

    default_options = dict(library=diffuse_defaults.diffuse['library'],
                           outdir=(None, 'Output directory', str),)

    __doc__ += Link.construct_docstring(default_options)

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        gmm = make_ring_dicts(library=args['library'], basedir='.')

        for galkey in gmm.galkeys():
            ring_dict = gmm.ring_dict(galkey)
            for ring_key, ring_info in ring_dict.items():
                output_file = ring_info.merged_gasmap
                file_string = ""
                for fname in ring_info.files:
                    file_string += " %s" % fname
                logfile = make_nfs_path(output_file.replace('.fits', '.log'))
                job_configs[ring_key] = dict(output=output_file,
                                             args=file_string,
                                             logfile=logfile)

        return job_configs


class Vstack_SG(ScatterGather):
    """Small class to generate configurations for `Link_FermipyVstack`
    to merge source maps

    """
    appname = 'fermipy-vstack-sg'
    usage = "%s [options]" % (appname)
    description = "Submit fermipy-vstack jobs in parallel"
    clientclass = Link_FermipyVstack

    job_time = 300

    default_options = dict(comp=diffuse_defaults.diffuse['comp'],
                           data=diffuse_defaults.diffuse['data'],
                           library=diffuse_defaults.diffuse['library'],)

    __doc__ += Link.construct_docstring(default_options)

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        components = Component.build_from_yamlfile(args['comp'])
        NAME_FACTORY.update_base_dict(args['data'])

        ret_dict = make_diffuse_comp_info_dict(components=components,
                                               library=args['library'],
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
                                 mktime='none',
                                 coordsys=comp.coordsys,
                                 irf_ver=NAME_FACTORY.irf_ver(),
                                 fullpath=True)

                outfile = NAME_FACTORY.srcmaps(**name_keys)
                outfile_tokens = os.path.splitext(outfile)
                infile_regexp = "%s_*.fits*" % outfile_tokens[0]
                full_key = "%s_%s" % (sub_comp_info.sourcekey, key)
                logfile = make_nfs_path(outfile.replace('.fits', '.log'))
                job_configs[full_key] = dict(output=outfile,
                                             args=infile_regexp,
                                             hdu=sub_comp_info.source_name,
                                             logfile=logfile)

        return job_configs


class GatherSrcmaps_SG(ScatterGather):
    """Small class to generate configurations for `Link_FermipyGatherSrcmaps`

    """
    appname = 'fermipy-gather-srcmaps-sg'
    usage = "%s [options]" % (appname)
    description = "Submit fermipy-gather-srcmaps  jobs in parallel"
    clientclass = Link_FermipyGatherSrcmaps

    job_time = 300

    default_options = dict(comp=diffuse_defaults.diffuse['comp'],
                           data=diffuse_defaults.diffuse['data'],
                           library=diffuse_defaults.diffuse['library'])

    __doc__ += Link.construct_docstring(default_options)

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        components = Component.build_from_yamlfile(args['comp'])
        NAME_FACTORY.update_base_dict(args['data'])

        ret_dict = make_catalog_comp_dict(library=args['library'],
                                          basedir=NAME_FACTORY.base_dict['basedir'])
        catalog_info_dict = ret_dict['catalog_info_dict']

        for catalog_name  in catalog_info_dict:
            for comp in components:
                zcut = "zmax%i" % comp.zmax
                key = comp.make_key('{ebin_name}_{evtype_name}')
                name_keys = dict(zcut=zcut,
                                 sourcekey=catalog_name,
                                 ebin=comp.ebin_name,
                                 psftype=comp.evtype_name,
                                 coordsys=comp.coordsys,
                                 irf_ver=NAME_FACTORY.irf_ver(),
                                 mktime='none',
                                 fullpath=True)

                outfile = NAME_FACTORY.srcmaps(**name_keys)
                outfile_tokens = os.path.splitext(outfile)
                infile_regexp = "%s_*.fits" % outfile_tokens[0]
                logfile = make_nfs_path(outfile.replace('.fits', '.log'))
                job_configs[key] = dict(output=outfile,
                                        args=infile_regexp,
                                        logfile=logfile)

        return job_configs


class Healview_SG(ScatterGather):
    """Small class to generate configurations for `Link_FermipyHealview`
    
    """
    appname = 'fermipy-healviw-sg'
    usage = "%s [options]" % (appname)
    description = "Submit fermipy-healviw jobs in parallel"
    clientclass = Link_FermipyHealview

    job_time = 60

    default_options = dict(comp=diffuse_defaults.diffuse['comp'],
                           data=diffuse_defaults.diffuse['data'],
                           library=diffuse_defaults.diffuse['library'])

    __doc__ += Link.construct_docstring(default_options)

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        components = Component.build_from_yamlfile(args['comp'])
        NAME_FACTORY.update_base_dict(args['data'])

        ret_dict = make_diffuse_comp_info_dict(components=components,
                                               library=args['library'],
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
                                 coordsys=comp.coordsys,
                                 irf_ver=NAME_FACTORY.irf_ver(),
                                 mktime='none',
                                 fullpath=True)

                infile = NAME_FACTORY.srcmaps(**name_keys)
                outfile = infile.replace('.fits', '.png')

                logfile = make_nfs_path(outfile.replace('.png', '_png.log'))
                job_configs[full_key] = dict(input=infile,
                                             output=outfile,
                                             extension=sub_comp_info.source_name,
                                             zscale=args.get('zscale', 'log'),
                                             logfile=logfile)

        return job_configs


def register_classes():
    """Register these classes with the `LinkFactory` """
    Gtlink_select.register_class()
    Gtlink_bin.register_class()
    Gtlink_expcube2.register_class()
    Gtlink_scrmaps.register_class()
    Gtlink_mktime.register_class()
    Gtlink_ltcube.register_class()
    Link_FermipyCoadd.register_class()
    Link_FermipyGatherSrcmaps.register_class()
    Link_FermipyVstack.register_class()
    Link_FermipyHealview.register_class()
    Gtexpcube2_SG.register_class()
    Gtltsum_SG.register_class()
    SumRings_SG.register_class()
    Vstack_SG.register_class()
    GatherSrcmaps_SG.register_class()
    Healview_SG.register_class()
