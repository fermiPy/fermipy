# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Run gtsrcmaps for a single energy plane for a single source

This is useful to parallize the production of the source maps
"""
from __future__ import absolute_import, division, print_function

import os

import xml.etree.cElementTree as ElementTree

import BinnedAnalysis as BinnedAnalysis
import pyLikelihood as pyLike

from fermipy import utils
from fermipy.jobs.file_archive import FileFlags
from fermipy.jobs.link import Link
from fermipy.jobs.scatter_gather import ScatterGather
from fermipy.jobs.slac_impl import make_nfs_path

from fermipy.diffuse.name_policy import NameFactory
from fermipy.diffuse.binning import Component
from fermipy.diffuse.diffuse_src_manager import make_diffuse_comp_info_dict
from fermipy.diffuse.source_factory import make_sources
from fermipy.diffuse import defaults as diffuse_defaults


NAME_FACTORY = NameFactory()
HPX_ORDER_TO_KSTEP = {5: -1, 6: -1, 7: -1, 8: 2, 9: 1}


class GtSrcmapsDiffuse(Link):
    """Small class to create srcmaps for only once source in a model,
    and optionally for only some of the energy layers.

    This is useful for parallelizing source map creation.
    """
    NULL_MODEL = 'srcmdls/null.xml'

    appname = 'fermipy-srcmaps-diffuse'
    linkname_default = 'srcmaps-diffuse'
    usage = '%s [options]' % (appname)
    description = "Run gtsrcmaps for one or more energy planes for a single source"

    default_options = dict(irfs=diffuse_defaults.gtopts['irfs'],
                           expcube=diffuse_defaults.gtopts['expcube'],
                           bexpmap=diffuse_defaults.gtopts['bexpmap'],
                           cmap=diffuse_defaults.gtopts['cmap'],
                           srcmdl=diffuse_defaults.gtopts['srcmdl'],
                           outfile=diffuse_defaults.gtopts['outfile'],
                           source=(None, 'Input source', str),
                           kmin=(0, 'Minimum Energy Bin', int),
                           kmax=(-1, 'Maximum Energy Bin', int),
                           no_psf=(False, "Do not apply PSF smearing", bool),
                           gzip=(False, 'Compress output file', bool))

    default_file_args = dict(expcube=FileFlags.input_mask,
                             cmap=FileFlags.input_mask,
                             bexpmap=FileFlags.input_mask,
                             srcmdl=FileFlags.input_mask,
                             outfile=FileFlags.output_mask)

    __doc__ += Link.construct_docstring(default_options)

    def run_analysis(self, argv):
        """Run this analysis"""
        args = self._parser.parse_args(argv)
        obs = BinnedAnalysis.BinnedObs(irfs=args.irfs,
                                       expCube=args.expcube,
                                       srcMaps=args.cmap,
                                       binnedExpMap=args.bexpmap)

        if args.no_psf:
            performConvolution = False
        else:
            performConvolution = True

        config = BinnedAnalysis.BinnedConfig(performConvolution=performConvolution)
        like = BinnedAnalysis.BinnedAnalysis(obs,
                                             optimizer='MINUIT',
                                             srcModel=GtSrcmapsDiffuse.NULL_MODEL,
                                             wmap=None,
                                             config=config)

        source_factory = pyLike.SourceFactory(obs.observation)
        source_factory.readXml(args.srcmdl, BinnedAnalysis._funcFactory,
                               False, True, True)

        source = source_factory.releaseSource(args.source)

        try:
            diffuse_source = pyLike.DiffuseSource.cast(source)
        except TypeError:
            diffuse_source = None

        if diffuse_source is not None:
            try:
                diffuse_source.mapBaseObject().projmap().setExtrapolation(False)
            except RuntimeError:
                pass

        like.logLike.saveSourceMap_partial(args.outfile, source, args.kmin, args.kmax)

        if args.gzip:
            os.system("gzip -9 %s" % args.outfile)



class SrcmapsDiffuse_SG(ScatterGather):
    """Small class to generate configurations for `GtSrcmapsDiffuse`

    """
    appname = 'fermipy-srcmaps-diffuse-sg'
    usage = "%s [options]" % (appname)
    description = "Run gtsrcmaps for diffuse sources"
    clientclass = GtSrcmapsDiffuse

    job_time = 1500

    default_options = dict(comp=diffuse_defaults.diffuse['comp'],
                           data=diffuse_defaults.diffuse['data'],
                           library=diffuse_defaults.diffuse['library'],
                           make_xml=(True, 'Write xml files needed to make source maps', bool))

    __doc__ += Link.construct_docstring(default_options)

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
            print("Writing xml for %s to %s: %s %s" % (fullkey,
                                                       comp_dict.srcmdl_name,
                                                       comp_dict.model_type,
                                                       comp_dict.Spectral_Filename))
        elif comp_dict.model_type == 'MapCubeSource':
            print("Writing xml for %s to %s: %s %s" % (fullkey,
                                                       comp_dict.srcmdl_name,
                                                       comp_dict.model_type,
                                                       comp_dict.Spatial_Filename))
        SrcmapsDiffuse_SG._write_xml(comp_dict.srcmdl_name, srcdict.values())

    @staticmethod
    def _make_xml_files(diffuse_comp_info_dict):
        """Make all the xml file for individual components
        """
        try:
            os.makedirs('srcmdls')
        except OSError:
            pass

        for sourcekey in sorted(diffuse_comp_info_dict.keys()):
            comp_info = diffuse_comp_info_dict[sourcekey]
            if comp_info.components is None:
                SrcmapsDiffuse_SG._handle_component(sourcekey, comp_info)
            else:
                for sub_comp_info in comp_info.components.values():
                    SrcmapsDiffuse_SG._handle_component(sourcekey, sub_comp_info)

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        components = Component.build_from_yamlfile(args['comp'])
        NAME_FACTORY.update_base_dict(args['data'])

        ret_dict = make_diffuse_comp_info_dict(components=components,
                                               library=args['library'],
                                               basedir='.')
        diffuse_comp_info_dict = ret_dict['comp_info_dict']
        if args['make_xml']:
            SrcmapsDiffuse_SG._make_xml_files(diffuse_comp_info_dict)

        for diffuse_comp_info_key in sorted(diffuse_comp_info_dict.keys()):
            diffuse_comp_info_value = diffuse_comp_info_dict[diffuse_comp_info_key]
            no_psf = diffuse_comp_info_value.no_psf
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
                                 no_psf=no_psf,
                                 evtype=comp.evtype)

                if kstep < 0:
                    kstep = kmax
                else:
                    pass

                for k in range(kmin, kmax, kstep):
                    full_key = "%s_%s_%02i" % (diffuse_comp_info_key, key, k)
                    khi = min(kmax, k + kstep)

                    full_dict = base_dict.copy()
                    outfile = outfile_base.replace('.fits', '_%02i.fits' % k)
                    logfile = make_nfs_path(outfile_base.replace('.fits', '_%02i.log' % k))
                    full_dict.update(dict(outfile=outfile,
                                          kmin=k, kmax=khi,
                                          logfile=logfile))
                    job_configs[full_key] = full_dict

        return job_configs


def register_srcmaps_diffuse():
    """Register these classes with the `LinkFactory` """
    GtSrcmapsDiffuse.register_class()
    SrcmapsDiffuse_SG.register_class()
