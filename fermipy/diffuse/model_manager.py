# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Classes and utilities that manage fitting models for diffuse analyses
"""
from __future__ import absolute_import, division, print_function

import os
from collections import OrderedDict

import yaml

from fermipy import utils
from fermipy.diffuse.name_policy import NameFactory
from fermipy.diffuse.binning import Component
from fermipy.diffuse.diffuse_src_manager import GalpropMapManager,\
    DiffuseModelManager, make_diffuse_comp_info_dict
from fermipy.diffuse.catalog_src_manager import CatalogSourceManager, make_catalog_comp_dict
from fermipy.diffuse.source_factory import SourceFactory
from fermipy.diffuse.spectral import SpectralLibrary


class ModelComponent(object):
    """ Small helper class to tie a ModelComponentInfo to a spectrum """

    def __init__(self, **kwargs):
        """
        """
        self.info = kwargs.get('info')
        self.spectrum = kwargs.get('spectrum')


class ModelInfo(object):
    """ Small helper class to keep track of a single fitting model """

    def __init__(self, **kwargs):
        """
        """
        self.model_name = kwargs.get('model_name')
        self.model_components = kwargs.get('model_components', OrderedDict())

    @property
    def component_names(self):
        """ Return the list of name of the components """
        return sorted(self.model_components.keys())

    def __getitem__(self, key):
        """ Return a single model info by name """
        return self.model_components[key]

    def items(self):
        """ Return the key, value pairs of model components """
        return self.model_components.items()

    def make_srcmap_manifest(self, components, name_factory):
        """  Build a yaml file that specfies how to make the srcmap files for a particular model

        Parameters
        ----------

        components : list
            The binning components used in this analysis
        name_factory : `NameFactory`
            Object that handles naming conventions

        Returns a dictionary that contains information about where to find the
        source maps for each component of the model
        """
        ret_dict = {}

        for comp in components:
            compkey = comp.make_key('{ebin_name}_{evtype_name}')
            zcut = "zmax%i" % comp.zmax
            name_keys = dict(modelkey=self.model_name,
                             zcut=zcut,
                             ebin=comp.ebin_name,
                             psftype=comp.evtype_name)
            outsrcmap = name_factory.merged_srcmaps(**name_keys)
            ccube = name_factory.ccube(**name_keys)
            src_dict = {}
            for comp_name, model_comp in self.model_components.items():
                comp_info = model_comp.info
                model_type = comp_info.model_type
                name_keys['sourcekey'] = comp_name
                if model_type in ['CatalogSources']:
                    #sourcekey = comp_info.comp_key
                    sources = comp_info.source_names
                    name_keys['sourcekey'] = comp_info.catalog_info.catalog_name
                elif model_type in ['CompositeSource']:
                    #sourcekey = comp_info.sourcekey
                    name_keys['sourcekey'] = comp_info.sourcekey
                    sources = [comp_info.source_name]
                else:
                    #sourcekey = comp_name
                    sources = [comp_info.source_name]
                src_dict[comp_name] = dict(sourcekey=comp_name,
                                           srcmap_file=name_factory.srcmaps(
                                               **name_keys),
                                           source_names=sources)
            comp_dict = dict(outsrcmap=outsrcmap,
                             ccube=ccube,
                             source_dict=src_dict)
            ret_dict[compkey] = comp_dict

        return ret_dict

    def make_model_rois(self, components, name_factory):
        """ Make the fermipy roi_model objects for each of a set of binning components """
        ret_dict = {}

        # Figure out which sources need to be split by components
        master_roi_source_info = {}
        sub_comp_sources = {}
        for comp_name, model_comp in self.model_components.items():
            comp_info = model_comp.info
            if comp_info.components is None:
                master_roi_source_info[comp_name] = model_comp
            else:
                sub_comp_sources[comp_name] = model_comp

        # Build the xml for the master
        master_roi = SourceFactory.make_roi(master_roi_source_info)
        master_xml_mdl = name_factory.master_srcmdl_xml(
            modelkey=self.model_name)
        print ("Writing master ROI model to %s" % master_xml_mdl)
        master_roi.write_xml(master_xml_mdl)
        ret_dict['master'] = master_roi

        # Now deal with the components
        for comp in components:
            zcut = "zmax%i" % comp.zmax
            compkey = "%s_%s" % (zcut, comp.make_key(
                '{ebin_name}_{evtype_name}'))
            #name_keys = dict(zcut=zcut,
            #                 modelkey=self.model_name,
            #                 component=compkey)
            comp_roi_source_info = {}
            for comp_name, model_comp in sub_comp_sources.items():
                comp_info = model_comp.info
                #comps = comp_info.components
                if comp_info.selection_dependent:
                    key = comp.make_key('{ebin_name}_{evtype_name}')
                elif comp_info.moving:
                    key = zcut
                comp_roi_source_info[comp_name] =\
                    ModelComponent(info=comp_info.clone_and_merge_sub(key),
                                   spectrum=model_comp.spectrum)

            # Build the xml for the component
            comp_roi = SourceFactory.make_roi(comp_roi_source_info)
            comp_xml_mdl = name_factory.comp_srcmdl_xml(modelkey=self.model_name,
                                                        component=compkey)
            print ("Writing component ROI model to %s" % comp_xml_mdl)
            comp_roi.write_xml(comp_xml_mdl)
            ret_dict[compkey] = comp_roi

        return ret_dict


class ModelManager(object):
    """ Small helper class to create fitting models and manager XML files for fermipy

    This class contains a 'library', which is a dictionary of all the source components:

    specifically it maps:

    sourcekey : `model_component.ModelComponentInfo`

    """

    def __init__(self, **kwargs):
        """
        Keyword arguments
        -------------------
        basedir : str
            Top level directory for finding files
        """
        self._name_factory = NameFactory(**kwargs)
        self._dmm = kwargs.get('DiffuseModelManager',
                               DiffuseModelManager(**kwargs))
        self._gmm = kwargs.get('GalpropMapManager',
                               GalpropMapManager(**kwargs))
        self._csm = kwargs.get('CatalogSourceManager',
                               CatalogSourceManager(**kwargs))
        self._library = {}
        self._models = {}
        self._spec_lib = SpectralLibrary({})

    def read_model_yaml(self, modelkey):
        """ Read the yaml file for the diffuse components
        """
        model_yaml = self._name_factory.model_yaml(modelkey=modelkey,
                                                   fullpath=True)
        model = yaml.safe_load(open(model_yaml))
        return model

    @property
    def dmm(self):
        """ Return the DiffuseModelManager """
        return self._dmm

    @property
    def gmm(self):
        """ Return the GalpropMapManager """
        return self._gmm

    @property
    def csm(self):
        """ Return the CatalogSourceManager """
        return self._csm

    def make_library(self, diffuse_yaml, catalog_yaml, binning_yaml):
        """ Build up the library of all the components

        Parameters
        ----------

        diffuse_yaml : str
            Name of the yaml file with the library of diffuse component definitions
        catalog_yaml : str
            Name of the yaml file with the library of catalog split definitions
        binning_yaml : str
            Name of the yaml file with the binning definitions
        """
        ret_dict = {}
        #catalog_dict = yaml.safe_load(open(catalog_yaml))
        components_dict = Component.build_from_yamlfile(binning_yaml)
        diffuse_ret_dict = make_diffuse_comp_info_dict(GalpropMapManager=self._gmm,
                                                       DiffuseModelManager=self._dmm,
                                                       diffuse=diffuse_yaml,
                                                       components=components_dict)
        catalog_ret_dict = make_catalog_comp_dict(sources=catalog_yaml,
                                                  CatalogSourceManager=self._csm)
        ret_dict.update(diffuse_ret_dict['comp_info_dict'])
        ret_dict.update(catalog_ret_dict['comp_info_dict'])
        self._library.update(ret_dict)
        return ret_dict

    def make_model_info(self, modelkey):
        """ Build a dictionary with the information for a particular model.

        Parameters
        ----------

        modelkey : str
            Key used to identify this particular model

        Return `ModelInfo`
        """
        model = self.read_model_yaml(modelkey)
        sources = model['sources']
        components = OrderedDict()
        spec_model_yaml = self._name_factory.fullpath(localpath=model['spectral_models'])
        self._spec_lib.update(yaml.safe_load(open(spec_model_yaml)))
        for source, source_info in sources.items():
            model_type = source_info.get('model_type', None)
            version = source_info['version']
            spec_type = source_info['SpectrumType']
            sourcekey = "%s_%s" % (source, version)
            if model_type == 'galprop_rings':
                comp_info_dict = self.gmm.diffuse_comp_info_dicts(version)
                def_spec_type = spec_type['default']
                for comp_key, comp_info in comp_info_dict.items():
                    model_comp = ModelComponent(info=comp_info,
                                                spectrum=\
                                                    self._spec_lib[spec_type.get(comp_key,
                                                                                 def_spec_type)])
                    components[comp_key] = model_comp
            elif model_type == 'Catalog':
                comp_info_dict = self.csm.split_comp_info_dict(source, version)
                def_spec_type = spec_type['default']
                for comp_key, comp_info in comp_info_dict.items():
                    model_comp = ModelComponent(info=comp_info,
                                                spectrum=\
                                                    self._spec_lib[spec_type.get(comp_key,
                                                                                 def_spec_type)])
                    components[comp_key] = model_comp
            else:
                comp_info = self.dmm.diffuse_comp_info(sourcekey)
                model_comp = ModelComponent(info=comp_info,
                                            spectrum=self._spec_lib[spec_type])
                components[sourcekey] = model_comp
        ret_val = ModelInfo(model_name=modelkey,
                            model_components=components)
        self._models[modelkey] = ret_val
        return ret_val

    def make_srcmap_manifest(self, modelkey, components, data):
        """Build a yaml file that specfies how to make the srcmap files for a particular model

        Parameters
        ----------

        modelkey : str
            Key used to identify this particular model
        components : list
            The binning components used in this analysis
        data : str
            Path to file containing dataset definition
        """
        try:
            model_info = self._models[modelkey]
        except KeyError:
            model_info = self.make_model_info(modelkey)
        self._name_factory.update_base_dict(data)
        outfile = os.path.join('analysis', 'model_%s' %
                               modelkey, 'srcmap_manifest_%s.yaml' % modelkey)
        manifest = model_info.make_srcmap_manifest(
            components, self._name_factory)

        outdir = os.path.dirname(outfile)
        try:
            os.makedirs(outdir)
        except OSError:
            pass
        utils.write_yaml(manifest, outfile)

    def make_fermipy_config_yaml(self, modelkey, components, data, **kwargs):
        """Build a fermipy top-level yaml configuration file

        Parameters
        ----------

        modelkey : str
            Key used to identify this particular model
        components : list
            The binning components used in this analysis
        data : str
            Path to file containing dataset definition
        """
        model_dir = os.path.join('analysis', 'model_%s' % modelkey)
        hpx_order = kwargs.get('hpx_order', 9)
        self._name_factory.update_base_dict(data)
        try:
            model_info = self._models[modelkey]
        except KeyError:
            model_info = self.make_model_info(modelkey)

        model_info.make_model_rois(components, self._name_factory)

        #source_names = model_info.component_names

        master_xml_mdl = os.path.basename(
            self._name_factory.master_srcmdl_xml(modelkey=modelkey))

        master_data = dict(scfile=self._name_factory.ft2file(fullpath=True),
                           cacheft1=False)
        master_binning = dict(projtype='HPX',
                              coordsys=kwargs.get('coordsys', 'GAL'),
                              roiwidth=180.,
                              binsperdec=8,
                              hpx_ordering_scheme="RING",
                              hpx_order=hpx_order,
                              hpx_ebin=True)
        master_fileio = dict(outdir=model_dir,
                             logfile=os.path.join(model_dir, 'fermipy.log'))
        master_gtlike = dict(irfs=self._name_factory.irfs(**kwargs),
                             edisp_disable=['isodiff', 'diffuse', 'limb'])
        master_selection = dict(glat=0., glon=0., radius=180.)
        master_model = dict(catalogs=[master_xml_mdl])

        master = dict(data=master_data,
                      binning=master_binning,
                      fileio=master_fileio,
                      selection=master_selection,
                      gtlike=master_gtlike,
                      model=master_model,
                      components=[])

        fermipy_dict = master

        #comp_rois = {}

        for comp in components:
            zcut = "zmax%i" % comp.zmax
            compkey = "%s_%s" % (zcut, comp.make_key(
                '{ebin_name}_{evtype_name}'))
            name_keys = dict(zcut=zcut,
                             modelkey=modelkey,
                             component=compkey,
                             fullpath=True)
            comp_data = dict(ltcube=self._name_factory.ltcube(**name_keys))
            comp_selection = dict(logemin=comp.log_emin,
                                  logemax=comp.log_emax,
                                  zmax=comp.zmax,
                                  evtype=comp.evtype)
            comp_binning = dict(enumbins=comp.enumbins,
                                hpx_order=min(comp.hpx_order, hpx_order))
            comp_gtlike = dict(srcmap=self._name_factory.merged_srcmaps(**name_keys),
                               bexpmap=self._name_factory.bexpcube(**name_keys))
            #comp_roi_source_info = {}

            comp_xml_mdl = os.path.basename(self._name_factory.comp_srcmdl_xml(modelkey=modelkey,
                                                                               component=compkey))
            comp_model = dict(catalogs=[master_xml_mdl, comp_xml_mdl])
            sub_dict = dict(data=comp_data,
                            binning=comp_binning,
                            selection=comp_selection,
                            gtlike=comp_gtlike,
                            model=comp_model)
            fermipy_dict['components'].append(sub_dict)

        outfile = os.path.join(model_dir, 'config.yaml')
        print ("Writing fermipy config file %s"%outfile)
        utils.write_yaml(fermipy_dict, outfile)
        return fermipy_dict

    @staticmethod
    def get_sub_comp_info(source_info, comp):
        """Build and return information about a sub-component for a particular selection
        """
        sub_comps = source_info.get('components', None)
        if sub_comps is None:
            return source_info.copy()
        moving = source_info.get('moving', False)
        selection_dependent = source_info.get('selection_dependent', False)
        if selection_dependent:
            key = comp.make_key('{ebin_name}_{evtype_name}')
        elif moving:
            key = "zmax%i" % comp.zmax
        ret_dict = source_info.copy()
        ret_dict.update(sub_comps[key])
        return ret_dict


def make_library(**kwargs):
    """Build and return a ModelManager object and fill the associated model library
    """
    diffuse_yaml = kwargs.pop('diffuse', 'config/diffuse_components.yaml')
    catalog_yaml = kwargs.pop('sources', 'config/catalog_components.yaml')
    comp_yaml = kwargs.pop('comp', 'config/binning.yaml')
    basedir = kwargs.pop('basedir', 
                         '/nfs/slac/kipac/fs1/u/dmcat/data/flight/diffuse_fitting')

    model_man = kwargs.get('ModelManager', ModelManager(basedir=basedir))
    model_comp_dict = model_man.make_library(diffuse_yaml, catalog_yaml, comp_yaml)

    return dict(model_comp_dict=model_comp_dict,
                ModelManager=model_man)
