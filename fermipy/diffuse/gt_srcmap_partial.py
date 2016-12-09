# Licensed under a 3-clause BSD style license - see LICENSE.rst
""" 
Run gtsrcmaps for a single energy plane for a single source

This is useful to parallize the production of the source maps
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import argparse

import BinnedAnalysis as BinnedAnalysis
import SummedLikelihood as SummedLikelihood
import pyLikelihood as pyLike

from fermipy.jobs.chain import Link
from fermipy.jobs.scatter_gather import ConfigMaker
from fermipy.jobs.lsf_impl import build_sg_from_link
from fermipy.diffuse.name_policy import NameFactory
from fermipy.diffuse.binning import Component
from fermipy.diffuse.diffuse_src_manager import make_diffuse_comp_info_dict


NAME_FACTORY = NameFactory()
HPX_ORDER_TO_KSTEP = {5:-1, 6:-1, 7:-1, 8:2, 9:1}


class GtSrcmapPartial(object):
    """Small class to create srcmaps for only once source in a model, 
    and optionally for only some of the energy layers.

    This is useful for parallelizing source map creation.
    """
    NULL_MODEL = 'srcmdls/null.xml'

    def __init__(self):
        """C'tor
        """
        self.parser = GtSrcmapPartial._make_parser()
        self.link = GtSrcmapPartial._make_link(self.parser)

    @staticmethod
    def _make_parser():
        """Make an argument parser for this class """
        usage = "gt_srcmap_partial.py [options]" 
        description = "Run gtsrcmaps for one or more energy planes for a single source"
        
        parser = argparse.ArgumentParser(usage=usage, description=description)
        parser.add_argument('--irfs', type=str, default='CALDB',
                            help='Instrument response functions')
        parser.add_argument('--expcube', type=str, default=None,
                            help='Input Livetime cube file')
        parser.add_argument('--cmap', type=str, default=None, 
                            help='Input counts map file')
        parser.add_argument('--bexpmap', type=str, default=None, 
                            help='Input binned exposure map file')
        parser.add_argument('--srcmdl', type=str, default=None, 
                            help='Input source model xml file')
        parser.add_argument('--source', type=str, default=None, 
                            help='Input source')
        parser.add_argument('--outfile', type=str, default=None, 
                            help='Output file')
        parser.add_argument('--kmin', type=int, default=0, 
                            help='Minimum Energy Bin')
        parser.add_argument('--kmax', type=int, default=-1, 
                            help='Maximum Energy Bin (-1 for all bins)')
        parser.add_argument('--gzip', action='store_true',
                            help='Compress output file')

    @staticmethod
    def _make_link(parser):        
        link = Link('gtsrcmap_partial',
                    appname=os.path.abspath(__file__).replace('.pyc','.py'),
                    options=dict(irfs=None, expcube=None, cmap=None,
                                 bexpmap=None, srcmdl=None, source=None,
                                 outfile=None, kmin=0, kmax=-1, gzip=True),
                    flags=['gzip'],
                    input_file_args=['expcube', 'cmap', 'bexpmap', 'srcmdl'],
                    output_file_args=['outfile'])
        return link

    def run(self, argv):
        args = self.parser.parse_args(argv)
        obs = BinnedAnalysis.BinnedObs(irfs=args.irfs,
                                       expCube=args.expcube,
                                       srcMaps=args.cmap,
                                       binnedExpMap=args.bexpmap)
    
        like = BinnedAnalysis.BinnedAnalysis(obs, 
                                             optimizer='MINUIT',
                                             srcModel=GtSrcmapPartial.NULL_MODEL,
                                             wmap=None)
        
        source_factory = pyLike.SourceFactory(obs.observation)
        source_factory.readXml(args.srcmdl, BinnedAnalysis._funcFactory,
                               False, True, True)
        
        source = source_factory.releaseSource(args.source)

        try:
            diffuse_source = pyLike.DiffuseSource.cast(source)
            diffuse_source.mapBaseObject().projmap().setExtrapolation(False)
        except:
            pass
        
        like.logLike.saveSourceMap_partial(args.outfile,source,args.kmin,args.kmax)
        
        if args.gzip:
            os.system("gzip -9 %s"%args.outfile)



class ConfigMaker_SrcmapPartial(ConfigMaker):
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
                                               basedir='.')
        diffuse_comp_info_dict = ret_dict['comp_info_dict']
        diffuse_comp_info_keys = diffuse_comp_info_dict.keys()
        diffuse_comp_info_keys.sort()

        for diffuse_comp_info_key in diffuse_comp_info_keys:
            diffuse_comp_info_value = diffuse_comp_info_dict[diffuse_comp_info_key]
            for comp in components:                
                zcut = "zmax%i"%comp.zmax
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
                                 evtype=comp.evtype)

                if kstep < 0:
                    kstep = kmax
                    for k in range(kmin, kmax, kstep):
                        full_key = "%s_%s_%02i"%(diffuse_comp_info_key, key, k)
                        khi = min(kmax, k+kstep)
                        
                        full_dict = base_dict.copy()
                        full_dict.update(dict(outfile=outfile_base.replace('.fits','_%02i.fits'%k),
                                              kmin=k, kmax=khi,
                                              logfile=outfile_base.replace('.fits','_%02i.log'%k)))
                        job_configs[full_key] = full_dict
            
        output_config = {}        
        return input_config, job_configs, output_config



def build_scatter_gather():
    """Build and return a ScatterGather object that can invoke this script"""
    gtsmp = GtSrcmapPartial()
    link = gtsmp.link

    lsf_args = {'W':1500,
                'R':'rhel60'}

    usage = "gt_split_and_bin.py [options] input"
    description = "Prepare data for diffuse all-sky analysis"
    
    config_maker = ConfigMaker_SrcmapPartial(link)    
    lsf_sg = build_sg_from_link(link, config_maker,
                                scatter_lsf_args=lsf_args,
                                usage=usage, 
                                description=description)
    return lsf_sg


def main_single():
    """Entry point for command line use for single job """
    gtsmp = GtSrcmapPartial()
    gtsmp.run(sys.argv[1:])

def main_batch():
    """Entry point for command line use  for dispatching batch jobs """
    lsf_sg = build_scatter_gather()
    lsf_sg(sys.argv)


if __name__ == '__main__':
    main_single()




                               



