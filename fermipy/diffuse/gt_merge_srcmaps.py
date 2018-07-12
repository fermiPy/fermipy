# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Merge source maps to build composite sources
"""
from __future__ import absolute_import, division, print_function

import os

import BinnedAnalysis as BinnedAnalysis
import pyLikelihood as pyLike

from fermipy.jobs.file_archive import FileFlags
from fermipy.jobs.link import Link
from fermipy.jobs.scatter_gather import ScatterGather
from fermipy.jobs.slac_impl import make_nfs_path
from fermipy.diffuse.name_policy import NameFactory
from fermipy.diffuse.binning import Component
from fermipy.diffuse.catalog_src_manager import make_catalog_comp_dict
from fermipy.diffuse import defaults as diffuse_defaults

NAME_FACTORY = NameFactory()


class GtMergeSrcmaps(Link):
    """Small class to merge source maps for composite sources.

    This is useful for parallelizing source map creation.
    """
    NULL_MODEL = 'srcmdls/null.xml'

    appname = 'fermipy-merge-srcmaps'
    linkname_default = 'merge-srcmaps'
    usage = '%s [options]' % (appname)
    description = "Mrege source maps from a set of sources"

    default_options = dict(irfs=diffuse_defaults.gtopts['irfs'],
                           expcube=diffuse_defaults.gtopts['expcube'],
                           bexpmap=diffuse_defaults.gtopts['bexpmap'],
                           srcmaps=diffuse_defaults.gtopts['srcmaps'],
                           srcmdl=diffuse_defaults.gtopts['srcmdl'],
                           outfile=diffuse_defaults.gtopts['outfile'],
                           merged=(None, 'Name of merged source', str),
                           outxml=(None, 'Output source model xml file', str),
                           gzip=(False, 'Compress output file', bool))

    default_file_args = dict(expcube=FileFlags.input_mask,
                             cmap=FileFlags.input_mask,
                             bexpmap=FileFlags.input_mask,
                             srcmdl=FileFlags.input_mask,
                             outfile=FileFlags.output_mask,
                             outxml=FileFlags.output_mask)

    __doc__ += Link.construct_docstring(default_options)

    def run_analysis(self, argv):
        """Run this analysis"""
        args = self._parser.parse_args(argv)

        obs = BinnedAnalysis.BinnedObs(irfs=args.irfs,
                                       expCube=args.expcube,
                                       srcMaps=args.srcmaps,
                                       binnedExpMap=args.bexpmap)

        like = BinnedAnalysis.BinnedAnalysis(obs,
                                             optimizer='MINUIT',
                                             srcModel=GtMergeSrcmaps.NULL_MODEL,
                                             wmap=None)

        like.logLike.set_use_single_fixed_map(False)

        print("Reading xml model from %s" % args.srcmdl)
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
                # EAC, add the source directly to the model
                like.logLike.addSource(source)
                srcs_to_merge.append(source_name)
            except KeyError:
                missing_sources.append(source_name)

        comp = like.mergeSources(args.merged, source_names, 'ConstantValue')
        like.logLike.getSourceMap(comp.getName())

        print("Merged %i sources into %s" % (len(srcs_to_merge), comp.getName()))
        if missing_sources:
            print("Missed sources: ", missing_sources)

        print("Writing output source map file %s" % args.outfile)
        like.logLike.saveSourceMaps(args.outfile, False, False)
        if args.gzip:
            os.system("gzip -9 %s" % args.outfile)

        print("Writing output xml file %s" % args.outxml)
        like.writeXml(args.outxml)


class MergeSrcmaps_SG(ScatterGather):
    """Small class to generate configurations for `GtMergeSrcmaps`

    """
    appname = 'fermipy-merge-srcmaps-sg'
    usage = "%s [options]" % (appname)
    description = "Merge diffuse maps for all-sky analysis"
    clientclass = GtMergeSrcmaps

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
        ret_dict = make_catalog_comp_dict(sources=args['library'], basedir='.')
        comp_info_dict = ret_dict['comp_info_dict']

        for split_ver, split_dict in comp_info_dict.items():
            for source_key, source_dict in split_dict.items():
                full_key = "%s_%s" % (split_ver, source_key)
                merged_name = "%s_%s" % (source_dict.catalog_info.catalog_name, source_key)
                if source_dict.model_type != 'CompositeSource':
                    continue

                for comp in components:
                    zcut = "zmax%i" % comp.zmax
                    key = "%s_%s" % (full_key, comp.make_key('{ebin_name}_{evtype_name}'))
                    name_keys = dict(zcut=zcut,
                                     sourcekey=full_key,
                                     ebin=comp.ebin_name,
                                     psftype=comp.evtype_name,
                                     coordsys=comp.coordsys,
                                     mktime='none',
                                     irf_ver=NAME_FACTORY.irf_ver())
                    nested_name_keys = dict(zcut=zcut,
                                            sourcekey=source_dict.catalog_info.catalog_name,
                                            ebin=comp.ebin_name,
                                            psftype=comp.evtype_name,
                                            coordsys=comp.coordsys,
                                            mktime='none',
                                            irf_ver=NAME_FACTORY.irf_ver())
                    outfile = NAME_FACTORY.srcmaps(**name_keys)
                    logfile = make_nfs_path(outfile.replace('.fits', '.log'))
                    job_configs[key] = dict(srcmaps=NAME_FACTORY.srcmaps(**nested_name_keys),
                                            expcube=NAME_FACTORY.ltcube(**name_keys),
                                            irfs=NAME_FACTORY.irfs(**name_keys),
                                            bexpmap=NAME_FACTORY.bexpcube(**name_keys),
                                            srcmdl=NAME_FACTORY.srcmdl_xml(**name_keys),
                                            merged=merged_name,
                                            outfile=outfile,
                                            outxml=NAME_FACTORY.nested_srcmdl_xml(**name_keys),
                                            logfile=logfile)

        return job_configs


def register_merge_srcmaps():
    """Register these classes with the `LinkFactory` """
    GtMergeSrcmaps.register_class()
    MergeSrcmaps_SG.register_class()
