# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Run gtsrcmaps for a single energy plane for a single source

This is useful to parallize the production of the source maps
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
import math

import xml.etree.cElementTree as ElementTree

import BinnedAnalysis as BinnedAnalysis
import pyLikelihood as pyLike

from fermipy import utils
from fermipy.jobs.file_archive import FileFlags
from fermipy.jobs.chain import add_argument, Link
from fermipy.jobs.scatter_gather import ConfigMaker, build_sg_from_link
from fermipy.jobs.lsf_impl import make_nfs_path, get_lsf_default_args, LSF_Interface
from fermipy.diffuse.name_policy import NameFactory
from fermipy.diffuse.binning import Component
from fermipy.diffuse.catalog_src_manager import make_catalog_comp_dict
from fermipy.diffuse.source_factory import make_sources
from fermipy.diffuse import defaults as diffuse_defaults


NAME_FACTORY = NameFactory()

class GtSrcmapsCatalog(Link):
    """Small class to create and write srcmaps for all the catalog sources, 
    once source at a time.

    This is useful for creating source maps for all the sources in a catalog
    """
    NULL_MODEL = 'srcmdls/null.xml'

    default_options = dict(irfs=diffuse_defaults.gtopts['irfs'],
                           expcube=diffuse_defaults.gtopts['expcube'],
                           bexpmap=diffuse_defaults.gtopts['bexpmap'],
                           cmap=diffuse_defaults.gtopts['cmap'],
                           srcmdl=diffuse_defaults.gtopts['srcmdl'],
                           outfile=diffuse_defaults.gtopts['outfile'],
                           srcmin=(0, 'Index of first source', int),
                           srcmax=(-1, 'Index of last source', int),
                           gzip=(False, 'Compress output file', bool))

    def __init__(self, **kwargs):
        """C'tor
        """
        parser = argparse.ArgumentParser(usage="fermipy-srcmaps-catalog [options]",
                                         description="Run gtsrcmaps for all the sources in a catalog")
        
        Link.__init__(self, kwargs.pop('linkname', 'srcmaps-diffuse'),
                      parser=parser,
                      appname='fermipy-srcmaps-catalog',
                      options=GtSrcmapsCatalog.default_options.copy(),
                      file_args=dict(expcube=FileFlags.input_mask,
                                     cmap=FileFlags.input_mask,
                                     bexpmap=FileFlags.input_mask,
                                     srcmdl=FileFlags.input_mask,
                                     outfile=FileFlags.output_mask))
  

 
    def run_analysis(self, argv):
        """Run this analysis"""
        args = self._parser.parse_args(argv)
        obs = BinnedAnalysis.BinnedObs(irfs=args.irfs,
                                       expCube=args.expcube,
                                       srcMaps=args.cmap,
                                       binnedExpMap=args.bexpmap)

        like = BinnedAnalysis.BinnedAnalysis(obs,
                                             optimizer='MINUIT',
                                             srcModel=GtSrcmapsCatalog.NULL_MODEL,
                                             wmap=None)

        source_factory = pyLike.SourceFactory(obs.observation)
        source_factory.readXml(args.srcmdl, BinnedAnalysis._funcFactory,
                               False, True, True)

        srcNames = pyLike.StringVector()
        source_factory.fetchSrcNames(srcNames)

        min_idx = args.srcmin
        max_idx = args.srcmax
        if max_idx < 0:
            max_idx = srcNames.size();

        for i in xrange(min_idx, max_idx):
            if i == min_idx:
                like.logLike.saveSourceMaps(args.outfile)
                pyLike.CountsMapBase.copyAndUpdateDssKeywords(args.cmap,
                                                              args.outfile,
                                                              None,
                                                              args.irfs)

            srcName = srcNames[i]
            source = source_factory.releaseSource(srcName)
            like.logLike.addSource(source, False)
            like.logLike.saveSourceMap_partial(args.outfile, source)
            like.logLike.deleteSource(srcName)

        if args.gzip:
            os.system("gzip -9 %s" % args.outfile)


class ConfigMaker_SrcmapsCatalog(ConfigMaker):
    """Small class to generate configurations for gtsrcmaps for catalog sources

    This takes the following arguments:
    --comp     : binning component definition yaml file
    --data     : datset definition yaml file
    --irf_ver  : IRF verions string (e.g., 'V6')
    --sources  : Yaml file with input source model definitions
    --make_xml : Write xml files for the individual components
    --nsrc     : Number of sources per job
    """
    default_options = dict(comp=diffuse_defaults.diffuse['comp'],
                           data=diffuse_defaults.diffuse['data'],
                           irf_ver=diffuse_defaults.diffuse['irf_ver'],
                           sources=diffuse_defaults.diffuse['sources'],
                           nsrc=(500, 'Number of sources per job', int),
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
            print("%s : %06i" % (val.srcmdl_name, len(val.roi_model.sources)))
            val.roi_model.write_xml(val.srcmdl_name)

        for val in comp_info_dict.values():
            for val2 in val.values():
                print("%s : %06i" % (val2.srcmdl_name, len(val2.roi_model.sources)))
                val2.roi_model.write_xml(val2.srcmdl_name)

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        components = Component.build_from_yamlfile(args['comp'])
        NAME_FACTORY.update_base_dict(args['data'])

        ret_dict = make_catalog_comp_dict(sources=args['sources'], 
                                          basedir=NAME_FACTORY.base_dict['basedir'])
        catalog_info_dict = ret_dict['catalog_info_dict']
        comp_info_dict = ret_dict['comp_info_dict']

        n_src_per_job = args['nsrc']

        if args['make_xml']:
            ConfigMaker_SrcmapsCatalog._make_xml_files(catalog_info_dict, comp_info_dict)

        for catalog_name, catalog_info in catalog_info_dict.items():

            n_cat_src = len(catalog_info.catalog.table)
            n_job = int(math.ceil(float(n_cat_src)/n_src_per_job))

            for comp in components:
                zcut = "zmax%i" % comp.zmax
                key = comp.make_key('{ebin_name}_{evtype_name}')
                name_keys = dict(zcut=zcut,
                                 sourcekey=catalog_name,
                                 ebin=comp.ebin_name,
                                 psftype=comp.evtype_name,
                                 coordsys='GAL',
                                 irf_ver=args['irf_ver'],
                                 mktime='none',
                                 fullpath=True)

                for i_job in range(n_job):
                    full_key = "%s_%02i"%(key, i_job)
                    srcmin = i_job*n_src_per_job
                    srcmax = min(srcmin+n_src_per_job, n_cat_src)
                    outfile = NAME_FACTORY.srcmaps(**name_keys).replace('.fits', "_%02i.fits"%(i_job))
                    logfile = make_nfs_path(outfile.replace('.fits', '.log'))
                    job_configs[full_key] = dict(cmap=NAME_FACTORY.ccube(**name_keys),
                                                 expcube=NAME_FACTORY.ltcube(**name_keys),
                                                 irfs=NAME_FACTORY.irfs(**name_keys),
                                                 bexpmap=NAME_FACTORY.bexpcube(**name_keys),
                                                 outfile=outfile,
                                                 logfile=logfile,
                                                 srcmdl=catalog_info.srcmdl_name,
                                                 evtype=comp.evtype,
                                                 srcmin=srcmin,
                                                 srcmax=srcmax)

        return job_configs


def create_link_gtsrcmaps_catalog(**kwargs):
    """Build and return a `Link` object that can invoke GtAssembleModel"""
    gtsrcmap_partial = GtSrcmapsCatalog(**kwargs)
    return gtsrcmap_partial


def create_sg_gtsrcmaps_catalog(**kwargs):
    """Build and return a ScatterGather object that can invoke gtsrcmaps for catalog sources"""
    appname = kwargs.pop('appname', 'fermipy-srcmaps-catalog-sg')
    link = create_link_gtsrcmaps_catalog(**kwargs)
    linkname=kwargs.pop('linkname', link.linkname)

    batch_args = get_lsf_default_args()    
    batch_args['lsf_args']['W'] = 6000
    batch_interface = LSF_Interface(**batch_args)

    usage = "%s [options]"%(appname)
    description = "Run gtsrcmaps for catalog sources"

    config_maker = ConfigMaker_SrcmapsCatalog(link)
    lsf_sg = build_sg_from_link(link, config_maker,
                                interface=batch_interface,
                                usage=usage,
                                description=description,
                                linkname=linkname,
                                appname=appname,
                                **kwargs)
    return lsf_sg



def main_single():
    """Entry point for command line use for single job """
    gtsmc = GtSrcmapsCatalog()
    gtsmc.run_analysis(sys.argv[1:])


def main_batch():
    """Entry point for command line use  for dispatching batch jobs """
    lsf_sg = create_sg_gtsrcmaps_catalog()
    lsf_sg(sys.argv)


if __name__ == '__main__':
    main_single()
