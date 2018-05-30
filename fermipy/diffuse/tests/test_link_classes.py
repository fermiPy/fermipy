# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

from fermipy.tests.utils import requires_dependency, requires_st_version

try:
    import GtApp
except ImportError:
    pass

# Skip tests in this file if Fermi ST aren't available
pytestmark = requires_dependency('Fermi ST')

def test_gtlink_classes():
    """ Test that we can create `Gtlink` classes """
    from fermipy.diffuse.job_library import Gtlink_select, Gtlink_bin,\
        Gtlink_expcube2, Gtlink_scrmaps, Gtlink_ltsum, Gtlink_mktime,\
        Gtlink_ltcube
    from fermipy.diffuse.solar import Gtlink_expcube2_wcs, Gtlink_exphpsun,\
        Gtlink_suntemp
    
    Gtlink_select.create()
    Gtlink_bin.create()
    Gtlink_expcube2.create()
    Gtlink_scrmaps.create()
    Gtlink_ltsum.create()
    Gtlink_mktime.create()
    Gtlink_ltcube.create()
    Gtlink_expcube2_wcs.create()
    Gtlink_exphpsun.create()
    Gtlink_suntemp.create()


def test_applink_classes():
    """ Test that we can create `Applink` classes """
    from fermipy.diffuse.job_library import Link_FermipyCoadd,\
        Link_FermipyGatherSrcmaps, Link_FermipyVstack, Link_FermipyHealview

    Link_FermipyCoadd.create()
    Link_FermipyGatherSrcmaps.create()
    Link_FermipyVstack.create()
    Link_FermipyHealview.create()


def test_link_classes():
    """ Test that we can create `Link` classes """
    from fermipy.diffuse.gt_assemble_model import InitModel, AssembleModel
    from fermipy.diffuse.gt_merge_srcmaps import GtMergeSrcmaps
    from fermipy.diffuse.gt_srcmap_partial import GtSrcmapsDiffuse
    from fermipy.diffuse.gt_srcmaps_catalog import GtSrcmapsCatalog
    from fermipy.diffuse.residual_cr import ResidualCR

    InitModel.create()
    AssembleModel.create()
    GtMergeSrcmaps.create()
    GtSrcmapsDiffuse.create()
    GtSrcmapsCatalog.create()
    ResidualCR.create()


def test_sg_classes():
    """ Test that we can create `ScatterGather` classes """
    from fermipy.diffuse.job_library import Gtexpcube2_SG, Gtltsum_SG,\
        SumRings_SG, Vstack_SG, GatherSrcmaps_SG, Healview_SG
    from fermipy.diffuse.gt_assemble_model import AssembleModel_SG
    from fermipy.diffuse.gt_coadd_split import CoaddSplit_SG
    from fermipy.diffuse.gt_merge_srcmaps import MergeSrcmaps_SG
    from fermipy.diffuse.gt_split_and_bin import SplitAndBin_SG
    from fermipy.diffuse.gt_split_and_mktime import SplitAndMktime_SG
    from fermipy.diffuse.gt_srcmap_partial import SrcmapsDiffuse_SG
    from fermipy.diffuse.gt_srcmaps_catalog import SrcmapsCatalog_SG
    from fermipy.diffuse.residual_cr import ResidualCR_SG
    from fermipy.diffuse.solar import Gtexpcube2wcs_SG, Gtexphpsun_SG,\
        Gtsuntemp_SG

    Gtexpcube2_SG.create()
    Gtltsum_SG.create()
    SumRings_SG.create()
    Vstack_SG.create()
    GatherSrcmaps_SG.create()
    Healview_SG.create()
    ResidualCR_SG.create()
    AssembleModel_SG.create()
    CoaddSplit_SG.create()
    MergeSrcmaps_SG.create()
    SplitAndBin_SG.create()
    SplitAndMktime_SG.create()
    SrcmapsDiffuse_SG.create()
    SrcmapsCatalog_SG.create()
    ResidualCR_SG.create()
    Gtexpcube2wcs_SG.create()
    Gtexphpsun_SG.create()
    Gtsuntemp_SG.create()


def test_chain_classes():
    """ Test that we can create `ScatterGather` classes """
    from fermipy.diffuse.diffuse_analysis import DiffuseCompChain,\
        CatalogCompChain, DiffuseAnalysisChain
    from fermipy.diffuse.gt_assemble_model import AssembleModelChain
    from fermipy.diffuse.gt_coadd_split import CoaddSplit
    from fermipy.diffuse.gt_split_and_bin import SplitAndBin, SplitAndBinChain
    from fermipy.diffuse.gt_split_and_mktime import SplitAndMktime, SplitAndMktimeChain
    from fermipy.diffuse.residual_cr import ResidualCRChain
    from fermipy.diffuse.solar import SunMoonChain
    DiffuseCompChain.create()
    CatalogCompChain.create()
    DiffuseAnalysisChain.create()
    AssembleModelChain.create()
    CoaddSplit.create()
    SplitAndBin.create()
    SplitAndBinChain.create()
    SplitAndMktime.create()
    SplitAndMktimeChain.create()
    ResidualCRChain.create()
    SunMoonChain.create()


if __name__ == '__main__':
    test_gtlink_classes()
    test_applink_classes()
    test_link_classes()
    test_sg_classes()
    test_chain_classes()
