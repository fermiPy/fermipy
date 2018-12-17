# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Scripts to run the all-sky diffuse analysis
"""
from __future__ import absolute_import, division, print_function

from fermipy.utils import load_yaml
from fermipy.jobs.link import Link
from fermipy.jobs.chain import Chain

from fermipy.diffuse import defaults as diffuse_defaults
from fermipy.diffuse.name_policy import NameFactory

from fermipy.diffuse.job_library import SumRings_SG, Vstack_SG, GatherSrcmaps_SG
from fermipy.diffuse.gt_srcmap_partial import SrcmapsDiffuse_SG
from fermipy.diffuse.gt_merge_srcmaps import MergeSrcmaps_SG
from fermipy.diffuse.gt_srcmaps_catalog import SrcmapsCatalog_SG
from fermipy.diffuse.gt_split_and_bin import SplitAndBinChain
from fermipy.diffuse.gt_assemble_model import AssembleModelChain


NAME_FACTORY = NameFactory()

class DiffuseCompChain(Chain):
    """Chain to build srcmaps for diffuse components

    This chain consists of:

    sum-rings : SumRings_SG
        Merge GALProp gas maps by type and ring

    srcmaps-diffuse : SrcmapsDiffuse_SG
        Compute diffuse component source maps in parallel

    vstack-diffuse : Vstack_SG
        Combine diffuse component source maps

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

    __doc__ += Link.construct_docstring(default_options)

    def __init__(self, **kwargs):
        """C'tor
        """
        super(DiffuseCompChain, self).__init__(**kwargs)
        self.comp_dict = None

    def _map_arguments(self, args):
        """Map from the top-level arguments to the arguments provided to
        the indiviudal links """
        data = args.get('data')
        comp = args.get('comp')
        library = args.get('library')
        dry_run = args.get('dry_run', False)

        self._set_link('sum-rings', SumRings_SG,
                       library=library,
                       outdir=args['outdir'],
                       dry_run=dry_run)

        self._set_link('srcmaps-diffuse', SrcmapsDiffuse_SG,
                       comp=comp, data=data,
                       library=library,
                       make_xml=args['make_xml'],
                       dry_run=dry_run)

        self._set_link('vstack-diffuse', Vstack_SG,
                       comp=comp, data=data,
                       library=library,
                       dry_run=dry_run)


class CatalogCompChain(Chain):
    """Small class to build srcmaps for catalog components

    This chain consists of:

    srcmaps-catalog  : SrcmapsCatalog_SG
        Build source maps for all catalog sources in parallel

    gather-srcmaps : GatherSrcmaps_SG
        Gather source maps into 

    merge-srcmaps : MergeSrcmaps_SG
        Compute source maps for merged sources

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

    __doc__ += Link.construct_docstring(default_options)

    def __init__(self, **kwargs):
        """C'tor
        """
        super(CatalogCompChain, self).__init__(**kwargs)
        self.comp_dict = None


    def _map_arguments(self, args):
        """Map from the top-level arguments to the arguments provided to
        the indiviudal links """
        data = args.get('data')
        comp = args.get('comp')
        library = args.get('library')
        dry_run = args.get('dry_run', False)

        self._set_link('srcmaps-catalog', SrcmapsCatalog_SG,
                       comp=comp, data=data,
                       library=library,
                       nsrc=args.get('nsrc', 500),
                       dry_run=dry_run)

        self._set_link('gather-srcmaps', GatherSrcmaps_SG,
                       comp=comp, data=data,
                       library=library,
                       dry_run=dry_run)
        
        self._set_link('merge-srcmaps', MergeSrcmaps_SG,
                       comp=comp, data=data,
                       library=library,
                       dry_run=dry_run)


class DiffuseAnalysisChain(Chain):
    """Chain to define diffuse all-sky analysis

    This chain consists of:

    prepare : `SplitAndBinChain`
        Bin the data and make the exposure maps

    diffuse-comp : `DiffuseCompChain`
        Make source maps for diffuse components

    catalog-comp : `CatalogCompChain`
        Make source maps for catalog components

    assemble-model : `AssembleModelChain`
        Assemble the models for fitting

    """
    appname = 'fermipy-diffuse-analysis'
    linkname_default = 'diffuse'
    usage = '%s [options]' % (appname)
    description = 'Run diffuse analysis chain'

    default_options = dict(config=diffuse_defaults.diffuse['config'],
                           dry_run=diffuse_defaults.diffuse['dry_run'])

    __doc__ += Link.construct_docstring(default_options)

    def _map_arguments(self, args):
        """Map from the top-level arguments to the arguments provided to
        the indiviudal links """
        config_yaml = args['config']
        config_dict = load_yaml(config_yaml)

        dry_run = args.get('dry_run', False)

        data = config_dict.get('data')
        comp = config_dict.get('comp')
        library = config_dict.get('library')
        models = config_dict.get('models')
        scratch = config_dict.get('scratch')

        self._set_link('prepare', SplitAndBinChain,
                       comp=comp, data=data,
                       ft1file=config_dict.get('ft1file'),
                       hpx_order_ccube=config_dict.get('hpx_order_ccube'),
                       hpx_order_expcube=config_dict.get('hpx_order_expcube'),
                       scratch=scratch,
                       dry_run=dry_run)

        self._set_link('diffuse-comp', DiffuseCompChain,
                       comp=comp, data=data,
                       library=library,
                       make_xml=config_dict.get('make_diffuse_comp_xml', False),
                       outdir=config_dict.get('merged_gasmap_dir', 'merged_gasmap'),
                       dry_run=dry_run)

        self._set_link('catalog-comp', CatalogCompChain,
                       comp=comp, data=data,
                       library=library,
                       make_xml=config_dict.get('make_catalog_comp_xml', False),
                       nsrc=config_dict.get('catalog_nsrc', 500),
                       dry_run=dry_run)
        
        self._set_link('assemble-model', AssembleModelChain,
                       comp=comp, data=data,
                       library=library,
                       models=models,
                       hpx_order=config_dict.get('hpx_order_fitting'),
                       dry_run=dry_run)


def register_classes():
    """Register these classes with the `LinkFactory` """
    DiffuseCompChain.register_class()
    CatalogCompChain.register_class()
    DiffuseAnalysisChain.register_class()
