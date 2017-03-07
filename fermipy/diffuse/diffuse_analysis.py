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


BUILDER_DICT = {'create_link_gtexpcube2':create_link_gtexpcube2,
                'create_link_gtscrmaps':create_link_gtscrmaps,
                'create_link_fermipy_coadd':create_link_fermipy_coadd,
                'create_link_fermipy_vstack':create_link_fermipy_vstack,
                'create_sg_gtexpcube2':create_sg_gtexpcube2,
                'create_sg_gtsrcmaps_catalog':create_sg_gtsrcmaps_catalog,
                'create_sg_sum_ring_gasmaps':create_sg_sum_ring_gasmaps,
                'create_sg_vstack_diffuse':create_sg_vstack_diffuse,
                'create_link_assemble_model':create_link_assemble_model,
                'create_sg_assemble_model':create_sg_assemble_model,
                'create_chain_coadd_split':create_chain_coadd_split,
                'create_link_merge_srcmaps':create_link_merge_srcmaps,
                'create_sg_merge_srcmaps':create_sg_merge_srcmaps,
                'create_chain_split_and_bin':create_chain_split_and_bin,
                'create_sg_split_and_bin':create_sg_split_and_bin,
                'create_link_srcmap_partial':create_link_srcmap_partial,
                'create_sg_srcmap_partial':create_sg_srcmap_partial,
                'create_link_residual_cr':create_link_residual_cr,
                'create_chain_residual_cr':create_chain_residual_cr,
                'create_sg_residual_cr':create_sg_residual_cr,
                'create_chain_diffuse_comps':create_chain_diffuse_comps,
                'create_chain_catalog_comps':create_chain_catalog_comps}


def build_analysis_link(linktype, **kwargs):
    """Build and return a `fermipy.jobs.Link` object to run a
    part of the analysis"""

    builder_name = 'create_%s'%linktype
    try:
        builder_func = BUILDER_DICT[builder_name]
    except KeyError:
        raise KeyError("Could not build an analysis link using a creator function %s"%builder_name)
    return builder_func(**kwargs)



if __name__ == '__main__':

    JOB_ARCHIVE = JobArchive.build_archive(job_archive_table='job_archive_temp2.fits',
                                           file_archive_table='file_archive_temp2.fits',
                                           base_path=os.path.abspath('.')+'/')

    PARSER = argparse.ArgumentParser(usage="diffuse_analysis.py [options] analyses",
                                     description="Run a high level analysis")
    PARSER.add_argument('--config', type=str, default=None, help="Yaml configuration file")
    PARSER.add_argument('--dry_run', action='store_true', help="Dry run only")
    PARSER.add_argument('analyses', nargs='+', type=str, help="Analysis steps to run")

    ARGS = PARSER.parse_args()

    CONFIG = yaml.load(open(ARGS.config))

    for ANALYSIS in ARGS.analyses:
        ANALYSIS_CONFIG = CONFIG[ANALYSIS]
        LINK = build_analysis_link(ANALYSIS)
        LINK.update_args(ANALYSIS_CONFIG)
        JOB_ARCHIVE.register_jobs(LINK.get_jobs())
        LINK.run(sys.stdout, ARGS.dry_run)
 
    JOB_ARCHIVE.file_archive.update_file_status()
    JOB_ARCHIVE.write_table_file()
