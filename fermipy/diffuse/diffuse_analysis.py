# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Scripts to run the all-sky diffuse analysis
"""
from __future__ import absolute_import, division, print_function

import sys
import os
import argparse

from collections import OrderedDict

from fermipy.utils import load_yaml
from fermipy.jobs.job_archive import JobArchive
from fermipy.jobs.link import Link 
from fermipy.jobs.chain import Chain, insert_app_config, purge_dict
from fermipy.jobs.slac_impl import check_log

from fermipy.diffuse import defaults as diffuse_defaults

from fermipy.diffuse.name_policy import NameFactory

NAME_FACTORY = NameFactory()


class DiffuseAnalysisChain(Chain):
    """Small class to define diffuse analysis chain"""
    appname = 'fermipy-diffuse-analysis'
    linkname_default = 'diffuse'
    usage = '%s [options]' %(appname)
    description='Run diffuse analysis chain'   

    default_options = dict(config=diffuse_defaults.diffuse['config'],
                           dry_run=diffuse_defaults.diffuse['dry_run'])

    def __init__(self, **kwargs):
        """C'tor
        """
        linkname, init_dict = self._init_dict(**kwargs)
        super(DiffuseAnalysisChain, self).__init__(linkname, **init_dict)

    def _register_link_classes(self):    
        from fermipy.diffuse.gt_split_and_bin import SplitAndBinChain
        from fermipy.diffuse.diffuse_src_manager import DiffuseCompChain
        from fermipy.diffuse.catalog_src_manager import CatalogCompChain
        from fermipy.diffuse.gt_assemble_model import AssembleModelChain
        SplitAndBinChain.register_class()
        DiffuseCompChain.register_class()
        CatalogCompChain.register_class()
        AssembleModelChain.register_class()
    
    def _map_arguments(self, input_dict):
        """Map from the top-level arguments to the arguments provided to
        the indiviudal links """
        config_yaml = input_dict['config']
        o_dict = OrderedDict()
        config_dict = load_yaml(config_yaml)
        
        dry_run = input_dict.get('dry_run', False)
        
        data = config_dict.get('data')
        comp = config_dict.get('comp')
        library = config_dict.get('library')
        models = config_dict.get('models')
        scratch = config_dict.get('scratch')

        insert_app_config(o_dict, 'prepare',
                          'fermipy-split-and-bin-chain',
                          comp=comp, data=data,
                          ft1file=config_dict.get('ft1file'),
                          hpx_order_ccube=config_dict.get('hpx_order_ccube'),
                          hpx_order_expcube=config_dict.get('hpx_order_expcube'),
                          scratch=scratch,
                          dry_run=dry_run)


        insert_app_config(o_dict, 'diffuse-comp',
                          'fermipy-diffuse-comp-chain',
                          comp=comp, data=data,
                          library=library, 
                          make_xml=config_dict.get('make_diffuse_comp_xml', False),
                          outdir=config_dict.get('merged_gasmap_dir', 'merged_gasmap'),
                          dry_run=dry_run)

        insert_app_config(o_dict, 'catalog-comp',
                          'fermipy-catalog-comp-chain',
                          comp=comp, data=data,
                          library=library, 
                          make_xml=config_dict.get('make_catalog_comp_xml', False),
                          nsrc=config_dict.get('catalog_nsrc', 500),
                          dry_run=dry_run)
        
        insert_app_config(o_dict, 'assemble-model',
                          'fermipy-assemble-model-chain',
                          comp=comp, data=data,
                          library=library, 
                          models=models,
                          hpx_order=config_dict.get('hpx_order_fitting'),
                          dry_run=dry_run)
        
        return o_dict


def main_chain():
    """Energy point for running the entire Cosmic-ray analysis """
    job_archive = JobArchive.build_archive(job_archive_table='job_archive_diffuse.fits',
                                           file_archive_table='file_archive_diffuse.fits',
                                           base_path=os.path.abspath('.') + '/')
                                           
    the_chain = DiffuseAnalysisChain('Diffuse', job_archive=job_archive)
    args = the_chain.run_argparser(sys.argv[1:])
    logfile = "log_%s_top.log" % the_chain.linkname
    the_chain.archive_self(logfile)
    if args.dry_run:
        outstr = sys.stdout
    else:
        outstr = open(logfile, 'append')

    the_chain.run_chain(sys.stdout, args.dry_run, sub_logs=True)
    if not args.dry_run:
        outstr.close()
    the_chain.finalize(args.dry_run)
    job_archive.update_job_status(check_log)
    job_archive.write_table_file()



def register_classes():
    DiffuseAnalysisChain.register_class()
