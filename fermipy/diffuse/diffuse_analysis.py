# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Scripts to run the all-sky diffuse analysis
"""
from __future__ import absolute_import, division, print_function


from collections import OrderedDict

from fermipy.utils import load_yaml
from fermipy.jobs.chain import Chain, insert_app_config

from fermipy.diffuse import defaults as diffuse_defaults
from fermipy.diffuse.name_policy import NameFactory

NAME_FACTORY = NameFactory()

class DiffuseCompChain(Chain):
    """Small class to build srcmaps for diffuse components
    """
    appname = 'fermipy-diffuse-comp-chain'
    linkname_default = 'diffuse-comp'
    usage = '%s [options]' % (appname)
    description = 'Run diffuse component analysis'

    default_options = dict(comp=diffuse_defaults.diffuse['comp'],
                           data=diffuse_defaults.diffuse['data'],
                           library=diffuse_defaults.diffuse['library'],
                           make_xml=diffuse_defaults.diffuse['make_xml'],
                           outdir=(None, 'Output directory', str),
                           dry_run=diffuse_defaults.diffuse['dry_run'])

    def __init__(self, **kwargs):
        """C'tor
        """
        linkname, init_dict = self._init_dict(**kwargs)
        super(DiffuseCompChain, self).__init__(linkname, **init_dict)
        self.comp_dict = None

    def _register_link_classes(self):
        from fermipy.diffuse.job_library import SumRings_SG, Vstack_SG
        from fermipy.diffuse.gt_srcmap_partial import SrcmapsDiffuse_SG
        SumRings_SG.register_class()
        SrcmapsDiffuse_SG.register_class()
        Vstack_SG.register_class()

    def _map_arguments(self, input_dict):
        """Map from the top-level arguments to the arguments provided to
        the indiviudal links """
        o_dict = OrderedDict()

        data = input_dict.get('data')
        comp = input_dict.get('comp')
        library = input_dict.get('library')
        dry_run = input_dict.get('dry_run', False)

        insert_app_config(o_dict, 'sum-rings',
                          'fermipy-sum-rings-sg',
                          library=library,
                          outdir=input_dict['outdir'],
                          dry_run=dry_run)

        insert_app_config(o_dict, 'srcmaps-diffuse',
                          'fermipy-srcmaps-diffuse-sg',
                          comp=comp, data=data,
                          library=library,
                          make_xml=input_dict['make_xml'],
                          dry_run=dry_run)

        insert_app_config(o_dict, 'vstack-diffuse',
                          'fermipy-vstack-sg',
                          comp=comp, data=data,
                          library=library,
                          dry_run=dry_run)

        return o_dict


class CatalogCompChain(Chain):
    """Small class to build srcmaps for diffuse components
    """
    appname = 'fermipy-catalog-comp-chain'
    linkname_default = 'catalog-comp'
    usage = '%s [options]' % (appname)
    description = 'Run catalog component analysis'

    default_options = dict(comp=diffuse_defaults.diffuse['comp'],
                           data=diffuse_defaults.diffuse['data'],
                           library=diffuse_defaults.diffuse['library'],
                           nsrc=(500, 'Number of sources per job', int),
                           make_xml=(False, "Make XML files for diffuse components", bool),
                           dry_run=diffuse_defaults.diffuse['dry_run'])

    def __init__(self, **kwargs):
        """C'tor
        """
        linkname, init_dict = self._init_dict(**kwargs)
        super(CatalogCompChain, self).__init__(linkname, **init_dict)
        self.comp_dict = None

    def _register_link_classes(self):
        from fermipy.diffuse.job_library import GatherSrcmaps_SG
        from fermipy.diffuse.gt_merge_srcmaps import MergeSrcmaps_SG
        from fermipy.diffuse.gt_srcmaps_catalog import SrcmapsCatalog_SG
        GatherSrcmaps_SG.register_class()
        MergeSrcmaps_SG.register_class()
        SrcmapsCatalog_SG.register_class()

    def _map_arguments(self, input_dict):
        """Map from the top-level arguments to the arguments provided to
        the indiviudal links """
        o_dict = OrderedDict()

        data = input_dict.get('data')
        comp = input_dict.get('comp')
        library = input_dict.get('library')
        dry_run = input_dict.get('dry_run', False)

        insert_app_config(o_dict, 'srcmaps-catalog',
                          'fermipy-srcmaps-catalog-sg',
                          comp=comp, data=data,
                          library=library,
                          nsrc=input_dict.get('nsrc', 500),
                          dry_run=dry_run)

        insert_app_config(o_dict, 'gather-srcmaps',
                          'fermipy-gather-srcmaps-sg',
                          comp=comp, data=data,
                          library=library,
                          dry_run=dry_run)

        insert_app_config(o_dict, 'merge-srcmaps',
                          'fermipy-merge-srcmaps-sg',
                          comp=comp, data=data,
                          library=library,
                          dry_run=dry_run)

        return o_dict


class DiffuseAnalysisChain(Chain):
    """Small class to define diffuse analysis chain"""
    appname = 'fermipy-diffuse-analysis'
    linkname_default = 'diffuse'
    usage = '%s [options]' % (appname)
    description = 'Run diffuse analysis chain'

    default_options = dict(config=diffuse_defaults.diffuse['config'],
                           dry_run=diffuse_defaults.diffuse['dry_run'])

    def __init__(self, **kwargs):
        """C'tor
        """
        linkname, init_dict = self._init_dict(**kwargs)
        super(DiffuseAnalysisChain, self).__init__(linkname, **init_dict)

    def _register_link_classes(self):
        from fermipy.diffuse.gt_split_and_bin import SplitAndBinChain
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




def register_classes():
    """Register these classes with the `LinkFactory` """
    DiffuseCompChain.register_class()
    CatalogCompChain.register_class()
    DiffuseAnalysisChain.register_class()
