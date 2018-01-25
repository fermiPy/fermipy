# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Scripts to run the all-sky diffuse analysis
"""
from __future__ import absolute_import, division, print_function

import sys
import os
import argparse

import yaml

from fermipy.jobs.job_archive import JobArchive
from fermipy.jobs.chain import Link, Chain
from fermipy.jobs.lsf_impl import check_log

from fermipy.diffuse import defaults as diffuse_defaults

from fermipy.diffuse.name_policy import NameFactory
from fermipy.diffuse.job_library import create_sg_gtexpcube2,\
    create_sg_fermipy_coadd, create_sg_sum_ring_gasmaps,\
    create_sg_vstack_diffuse
from fermipy.diffuse.gt_srcmaps_catalog import create_sg_gtsrcmaps_catalog
from fermipy.diffuse.gt_srcmap_partial import create_sg_srcmap_partial
from fermipy.diffuse.gt_assemble_model import create_sg_assemble_model,\
    create_link_init_model
from fermipy.diffuse.gt_split_and_bin import create_sg_split_and_bin
from fermipy.diffuse.gt_merge_srcmaps import create_sg_merge_srcmaps



NAME_FACTORY = NameFactory()


class DiffuseAnalysisChain(Chain):
    """Small class to define diffuse analysis chain"""
    default_options = diffuse_defaults.diffuse.copy()

    def __init__(self, linkname, **kwargs):
        """C'tor
        """
        link_split_and_bin = create_sg_split_and_bin(linkname="%s.split"%linkname,
                                                     mapping={'hpx_order_max':'hpx_order_ccube',
                                                              'action': 'action_split'})
        link_coadd_split = create_sg_fermipy_coadd(linkname="%s.coadd"%linkname,
                                                   mapping={'action': 'action_coadd'})
        link_expcube = create_sg_gtexpcube2(linkname="%s.expcube"%linkname,
                                            mapping={'hpx_order_max':'hpx_order_expcube',
                                                     'action': 'action_expcube'})
        link_gasmaps = create_sg_sum_ring_gasmaps(linkname="%s.gasmaps"%linkname,
                                                  mapping={'action':'action_gasmaps'})
        link_srcmaps_diffuse = create_sg_srcmap_partial(linkname="%s.srcmaps_diffuse"%linkname,
                                                        mapping={'action':'action_srcmaps_diffuse'})
        link_vstack_diffuse = create_sg_vstack_diffuse(linkname="%s.vstack_diffuse"%linkname,
                                                       mapping={'action':'action_vstack_diffuse'})
        link_srcmaps_catalogs = create_sg_gtsrcmaps_catalog(linkname="%s.srcmaps_catalog"%linkname,
                                                            mapping={'action':'action_srcmaps_catalog'})
        link_gather_catalogs = create_sg_gather_srcmaps(linkname="%s.gather_catalog"%linkname,
                                                        mapping={'action':'action_gather_catalog'})
        link_merge_catalogs = create_sg_merge_srcmaps(linkname="%s.merge_catalog"%linkname,
                                                         mapping={'action':'action_merge_catalog'})
        link_init_model = create_link_init_model(linkname="%s.init_model"%linkname,
                                                 mapping={'hpx_order':'hpx_order_fitting',
                                                          'action': 'action_init'})
        link_assemble_model = create_sg_assemble_model(linkname="%s.assemble"%linkname,
                                                       mapping={'hpx_order':'hpx_order_fitting',
                                                                'action': 'action_assemble'})

        parser = argparse.ArgumentParser(usage='fermipy-diffuse-analysis',
                                         description="Run diffuse analysis setup")
        
        Chain.__init__(self, linkname,
                       appname='fermipy-diffuse-analysis',
                       links=[link_split_and_bin, link_coadd_split, link_expcube,
                              link_gasmaps, link_srcmaps_diffuse, link_vstack_diffuse,
                              link_srcmaps_catalogs, link_gather_catalogs, link_merge_catalogs,
                              link_init_model, link_assemble_model],
                       options=DiffuseAnalysisChain.default_options.copy(),
                       argmapper=self._map_arguments,
                       parser=parser,
                       **kwargs)
    
    def _map_arguments(self, input_dict):
        """Map from the top-level arguments to the arguments provided to
        the indiviudal links """
        output_dict = input_dict.copy()
      
        if input_dict.get('dry_run', False):
            action = 'run'
        else:
            action = 'run'
       
        output_dict['action_split'] = action
        output_dict['action_coadd'] = action
        output_dict['action_expcube'] = action
        output_dict['action_gasmaps'] = action
        output_dict['action_srcmaps_diffuse'] = action
        output_dict['action_vstack_diffuse'] = action
        output_dict['action_srcmaps_catalog'] = action
        output_dict['action_gather_catalog'] = action
        output_dict['action_merge_catalog'] = action
        output_dict['action_init'] = action
        output_dict['action_assemble'] = action

        output_dict.pop('link', None)
        return output_dict



def create_chain_diffuse_analysis(**kwargs):
    """Build and return a `DiffuseAnalysisChain` object """
    ret_chain = DiffuseAnalysisChain(linkname=kwargs.pop('linkname', 'Diffuse'),
                                     **kwargs)
    return ret_chain

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

if __name__ == '__main__':
    main_chain()


