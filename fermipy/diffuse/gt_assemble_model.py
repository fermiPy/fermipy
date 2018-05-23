# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Merge source maps to build composite sources
"""
from __future__ import absolute_import, division, print_function

import os
import sys

import yaml

from astropy.io import fits
from fermipy.skymap import HpxMap

from fermipy.utils import load_yaml

from fermipy.jobs.scatter_gather import ScatterGather
from fermipy.jobs.slac_impl import make_nfs_path
from fermipy.jobs.link import Link
from fermipy.jobs.chain import Chain

from fermipy.diffuse.binning import Component
from fermipy.diffuse.name_policy import NameFactory
from fermipy.diffuse import defaults as diffuse_defaults
from fermipy.diffuse.model_manager import make_library

NAME_FACTORY = NameFactory()


class InitModel(Link):
    """Small class to preprate files fermipy analysis.

    Specifically this create the srcmap_manifest and fermipy_config_yaml files
    """
    appname = 'fermipy-init-model'
    linkname_default = 'init-model'
    usage = '%s [options]' % (appname)
    description = "Initialize model fitting directory"

    default_options = dict(comp=diffuse_defaults.diffuse['comp'],
                           data=diffuse_defaults.diffuse['data'],
                           library=diffuse_defaults.diffuse['library'],
                           models=diffuse_defaults.diffuse['models'],
                           hpx_order=diffuse_defaults.diffuse['hpx_order_fitting'])

    def run_analysis(self, argv):
        """ Build the manifest for all the models
        """
        args = self._parser.parse_args(argv)
        components = Component.build_from_yamlfile(args.comp)
        NAME_FACTORY.update_base_dict(args.data)
        model_dict = make_library(**args.__dict__)
        model_manager = model_dict['ModelManager']
        models = load_yaml(args.models)
        data = args.data
        hpx_order = args.hpx_order
        for modelkey in models:
            model_manager.make_srcmap_manifest(modelkey, components, data)
            model_manager.make_fermipy_config_yaml(modelkey, components, data,
                                                   hpx_order=hpx_order,
                                                   irf_ver=NAME_FACTORY.irf_ver())


class AssembleModel(Link):
    """Small class to assemple source map files for fermipy analysis.

    This is useful for re-merging after parallelizing source map creation.
    """
    appname = 'fermipy-assemble-model'
    linkname_default = 'assemble-model'
    usage = '%s [options]' % (appname)
    description = "Assemble sourcemaps for model fitting"

    default_options = dict(input=(None, 'Input yaml file', str),
                           compname=(None, 'Component name.', str),
                           hpx_order=diffuse_defaults.diffuse['hpx_order_fitting'])

    @staticmethod
    def copy_ccube(ccube, outsrcmap, hpx_order):
        """Copy a counts cube into outsrcmap file
        reducing the HEALPix order to hpx_order if needed.
        """
        sys.stdout.write("  Copying counts cube from %s to %s\n" % (ccube, outsrcmap))
        try:
            hdulist_in = fits.open(ccube)
        except IOError:
            hdulist_in = fits.open("%s.gz" % ccube)

        hpx_order_in = hdulist_in[1].header['ORDER']

        if hpx_order_in > hpx_order:
            hpxmap = HpxMap.create_from_hdulist(hdulist_in)
            hpxmap_out = hpxmap.ud_grade(hpx_order, preserve_counts=True)
            hpxlist_out = hdulist_in
            #hpxlist_out['SKYMAP'] = hpxmap_out.create_image_hdu()
            hpxlist_out[1] = hpxmap_out.create_image_hdu()
            hpxlist_out[1].name = 'SKYMAP'
            hpxlist_out.writeto(outsrcmap)
            return hpx_order
        else:
            os.system('cp %s %s' % (ccube, outsrcmap))
            #os.system('cp %s.gz %s.gz' % (ccube, outsrcmap))
            #os.system('gunzip -f %s.gz' % (outsrcmap))
        return None

    @staticmethod
    def open_outsrcmap(outsrcmap):
        """Open and return the outsrcmap file in append mode """
        outhdulist = fits.open(outsrcmap, 'append')
        return outhdulist

    @staticmethod
    def append_hdus(hdulist, srcmap_file, source_names, hpx_order):
        """Append HEALPix maps to a list

        Parameters
        ----------

        hdulist : list
            The list being appended to
        srcmap_file : str
            Path to the file containing the HDUs
        source_names : list of str
            Names of the sources to extract from srcmap_file
        hpx_order : int
            Maximum order for maps
        """
        sys.stdout.write("  Extracting %i sources from %s" % (len(source_names), srcmap_file))
        try:
            hdulist_in = fits.open(srcmap_file)
        except IOError:
            try:
                hdulist_in = fits.open('%s.gz' % srcmap_file)
            except IOError:
                sys.stdout.write("  Missing file %s\n" % srcmap_file)
                return

        for source_name in source_names:
            sys.stdout.write('.')
            sys.stdout.flush()
            if hpx_order is None:
                hdulist.append(hdulist_in[source_name])
            else:
                try:
                    hpxmap = HpxMap.create_from_hdulist(hdulist_in, hdu=source_name)
                except IndexError:
                    print("  Index error on source %s in file %s" % (source_name, srcmap_file))
                    continue
                except KeyError:
                    print("  Key error on source %s in file %s" % (source_name, srcmap_file))
                    continue
                hpxmap_out = hpxmap.ud_grade(hpx_order, preserve_counts=True)
                hdulist.append(hpxmap_out.create_image_hdu(name=source_name))
        sys.stdout.write("\n")
        hdulist.flush()
        hdulist_in.close()

    @staticmethod
    def assemble_component(compname, compinfo, hpx_order):
        """Assemble the source map file for one binning component

        Parameters
        ----------

        compname : str
            The key for this component (e.g., E0_PSF3)
        compinfo : dict
            Information about this component
        hpx_order : int
            Maximum order for maps

        """
        sys.stdout.write("Working on component %s\n" % compname)
        ccube = compinfo['ccube']
        outsrcmap = compinfo['outsrcmap']
        source_dict = compinfo['source_dict']

        hpx_order = AssembleModel.copy_ccube(ccube, outsrcmap, hpx_order)
        hdulist = AssembleModel.open_outsrcmap(outsrcmap)

        for comp_name in sorted(source_dict.keys()):
            source_info = source_dict[comp_name]
            source_names = source_info['source_names']
            srcmap_file = source_info['srcmap_file']
            AssembleModel.append_hdus(hdulist, srcmap_file,
                                      source_names, hpx_order)
        sys.stdout.write("Done!\n")

    def run_analysis(self, argv):
        """Assemble the source map file for one binning component
        FIXME
        """
        args = self._parser.parse_args(argv)
        manifest = yaml.safe_load(open(args.input))

        compname = args.compname
        value = manifest[compname]
        self.assemble_component(compname, value, args.hpx_order)


class AssembleModel_SG(ScatterGather):
    """Small class to generate configurations for this script

    Parameters
    ----------

    --compname  : binning component definition yaml file
    --data      : datset definition yaml file
    --models    : model definitino yaml file
    args        : Names of models to assemble source maps for
    """
    appname = 'fermipy-assemble-model-sg'
    usage = "%s [options]" % (appname)
    description = "Copy source maps from the library to a analysis directory"
    clientclass = AssembleModel

    job_time = 300

    default_options = dict(comp=diffuse_defaults.diffuse['comp'],
                           data=diffuse_defaults.diffuse['data'],
                           models=diffuse_defaults.diffuse['models'])

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        components = Component.build_from_yamlfile(args['comp'])
        NAME_FACTORY.update_base_dict(args['data'])

        models = load_yaml(args['models'])

        for modelkey in models:
            manifest = os.path.join('analysis', 'model_%s' % modelkey,
                                    'srcmap_manifest_%s.yaml' % modelkey)
            for comp in components:
                key = comp.make_key('{ebin_name}_{evtype_name}')
                fullkey = "%s_%s" % (modelkey, key)
                outfile = NAME_FACTORY.merged_srcmaps(modelkey=modelkey,
                                                      component=key,
                                                      coordsys=comp.coordsys,
                                                      mktime='none',
                                                      irf_ver=NAME_FACTORY.irf_ver())
                logfile = make_nfs_path(outfile.replace('.fits', '.log'))
                job_configs[fullkey] = dict(input=manifest,
                                            compname=key,
                                            logfile=logfile)
        return job_configs


class AssembleModelChain(Chain):
    """Small class to split, apply mktime and bin data according to some user-provided specification
    """
    appname = 'fermipy-assemble-model-chain'
    linkname_default = 'assemble-model-chain'
    usage = '%s [options]' % (appname)
    description = 'Run init-model and assemble-model'

    default_options = dict(data=diffuse_defaults.diffuse['data'],
                           comp=diffuse_defaults.diffuse['comp'],
                           library=diffuse_defaults.diffuse['library'],
                           models=diffuse_defaults.diffuse['models'],
                           hpx_order=diffuse_defaults.diffuse['hpx_order_fitting'],
                           dry_run=diffuse_defaults.diffuse['dry_run'])

    def __init__(self, **kwargs):
        """C'tor
        """
        super(AssembleModelChain, self).__init__(**kwargs)
        self.comp_dict = None

    def _register_link_classes(self):
        InitModel.register_class()
        AssembleModel_SG.register_class()

    def _map_arguments(self, input_dict):
        """Map from the top-level arguments to the arguments provided to
        the indiviudal links """
        data = input_dict.get('data')
        comp = input_dict.get('comp')
        library = input_dict.get('library')
        models = input_dict.get('models')
        hpx_order = input_dict.get('hpx_order_fitting')
        dry_run = input_dict.get('dry_run', False)

        self._set_link('init-model', InitModel,
                       comp=comp, data=data,
                       library=library,
                       models=models,
                       hpx_order=hpx_order,
                       dry_run=dry_run)

        self._set_link('assemble-model', AssembleModel_SG,
                       comp=comp, data=data,
                       models=models)


def register_classes():
    """Register these classes with the `LinkFactory` """
    InitModel.register_class()
    AssembleModel.register_class()
    AssembleModel_SG.register_class()
    AssembleModelChain.register_class()
