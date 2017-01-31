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

from fermipy.diffuse.job_library import create_link_gtexpcube2, create_link_gtscrmaps,\
    create_link_fermipy_coadd, create_link_fermipy_vstack,\
    create_sg_gtexpcube2, create_sg_gtsrcmaps_catalog,\
    create_sg_sum_ring_gasmaps, create_sg_vstack_diffuse
from fermipy.diffuse.gt_assemble_model import create_link_assemble_model, create_sg_assemble_model
from fermipy.diffuse.gt_coadd_split import create_chain_coadd_split
from fermipy.diffuse.gt_merge_srcmaps import create_link_merge_srcmaps, create_sg_merge_srcmaps
from fermipy.diffuse.gt_split_and_bin import create_chain_split_and_bin, create_sg_split_and_bin
from fermipy.diffuse.gt_srcmap_partial import create_link_srcmap_partial, create_sg_srcmap_partial
from fermipy.diffuse.residual_cr import create_link_residual_cr, create_chain_residual_cr,\
    create_sg_residual_cr
from fermipy.diffuse.diffuse_src_manager import create_chain_diffuse_comps
from fermipy.diffuse.catalog_src_manager import create_chain_catalog_comps


def build_analysis_link(linktype, **kwargs):
    builder_name = 'create_%s'%linktype
    try:
        builder_func = eval(builder_name)
    except NameError:
        raise NameError("Could not build an analysis link using a creator function %s"%builder_name)
    return builder_func(**kwargs)


if __name__ == '__main__':

    from fermipy.jobs.job_archive import JobArchive

    job_archive = JobArchive.build_archive(job_archive_table='job_archive_temp.fits',
                                           file_archive_table='file_archive_temp.fits',
                                           base_path=os.path.abspath('.')+'/')
    
    usage = "diffuse_analysis.py [options] analyses"
    description = "Run a high level analysis"
    
    parser = argparse.ArgumentParser(usage=usage, description=description)
    parser.add_argument('--config', type=str, default=None, help="Yaml configuration file")
    parser.add_argument('analyses', nargs='+', type=str, help="Analysis steps to run")

    args = parser.parse_args()
    
    config = yaml.load(open(args.config))

    for analysis in args.analyses:
        analysis_config = config[analysis]
        chain = build_analysis_link(analysis)
        chain.update_args(analysis_config)
        chain.run(sys.stdout, True)
        job_archive.register_jobs(chain.get_jobs())

    #job_dict = chain.get_jobs()
    #for i, job_details in enumerate(job_dict.values()):
    #    if i % 10 == 0:
    #        print ("Working on job %i"%i)
    #    job_archive.register_job(job_details)


