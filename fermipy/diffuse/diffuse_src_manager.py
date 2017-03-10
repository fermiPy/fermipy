# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Classes and utilities that manage the diffuse emission background models
"""
from __future__ import absolute_import, division, print_function

import sys
import argparse

import yaml

from fermipy.jobs.chain import Link, Chain

from fermipy.diffuse.name_policy import NameFactory
from fermipy.diffuse.binning import Component
from fermipy.diffuse.model_component import GalpropMergedRingInfo,\
    IsoComponentInfo, MapCubeComponentInfo
from fermipy.diffuse import defaults as diffuse_defaults


class GalpropMapManager(object):
    """ Small helper class to keep track of Galprop gasmaps

    This keeps track of two types of dictionaries.
    Both are keyed by: key = {source_name}_{ring}_{galkey}

    Where:
    {source_name} is something like 'merged_C0'
    {ring} is the ring index
    {galkey} is a key specifying which version of galprop rings to use.

    The two dictionaries are:
    ring_dict[key] = `model_component.GalpropMergedRingInfo`
    diffuse_comp_info_dict[key] ] `model_component.ModelComponentInfo`

    The dictionaries are defined in files called.
    models/galprop_rings_{galkey}.yaml
    """

    def __init__(self, **kwargs):
        """ C'tor

        Keyword arguments
        -----------------

        maptype : str [newGasMaps_ST]
            Used to define path to gasmap files
        projtype : str [healpix]
            Used to define path to gasmap files
        basedir : str
            Top level directory for finding files
        """
        self.maptype = kwargs.get('maptype', 'newGasMaps_ST')
        self.projtype = kwargs.get('projtype', 'healpix')
        self._name_factory = NameFactory(basedir=kwargs.get('basedir'))
        self._ring_dicts = {}
        self._diffuse_comp_info_dicts = {}

    def read_galprop_rings_yaml(self, galkey):
        """ Read the yaml file for a partiuclar galprop key
        """
        galprop_rings_yaml = self._name_factory.galprop_rings_yaml(galkey=galkey,
                                                                   fullpath=True)
        galprop_rings = yaml.safe_load(open(galprop_rings_yaml))
        return galprop_rings

    def galkeys(self):
        """ Return the list of galprop keys used """
        return sorted(self._ring_dicts.keys())

    def ring_dict(self, galkey):
        """ Return the ring dictionary for a particular galprop key
        """
        return self._ring_dicts[galkey]

    def diffuse_comp_info_dicts(self, galkey):
        """ Return the components info dictionary for a particular galprop key
        """
        return self._diffuse_comp_info_dicts[galkey]

    def merged_components(self, galkey):
        """ Return the set of merged components for a particular galprop key
        """
        return sorted(self._diffuse_comp_info_dicts[galkey].keys())

    def make_ring_filename(self, source_name, ring, galprop_run):
        """ Make the name of a gasmap file for a single ring

        Parameters
        ----------

        source_name : str
            The galprop component, used to define path to gasmap files
        ring : int
            The ring index
        galprop_run : str
            String identifying the galprop parameters
        """
        format_dict = self.__dict__.copy()
        format_dict['sourcekey'] = self._name_factory.galprop_ringkey(source_name=source_name,
                                                                      ringkey="ring_%i" % ring)
        format_dict['galprop_run'] = galprop_run
        return self._name_factory.galprop_gasmap(**format_dict)

    def make_merged_name(self, source_name, galkey, fullpath):
        """ Make the name of a gasmap file for a set of merged rings

        Parameters
        ----------

        source_name : str
            The galprop component, used to define path to gasmap files
        galkey : str
            A short key identifying the galprop parameters
        fullpath : bool
            Return the full path name
        """
        format_dict = self.__dict__.copy()
        format_dict['sourcekey'] = self._name_factory.galprop_sourcekey(source_name=source_name,
                                                                        galpropkey=galkey)
        format_dict['fullpath'] = fullpath
        return self._name_factory.merged_gasmap(**format_dict)

    def make_xml_name(self, source_name, galkey, fullpath):
        """ Make the name of an xml file for a model definition for a set of merged rings

        Parameters
        ----------

        source_name : str
            The galprop component, used to define path to gasmap files
        galkey : str
            A short key identifying the galprop parameters
        fullpath : bool
            Return the full path name
        """
        format_dict = self.__dict__.copy()
        format_dict['sourcekey'] = self._name_factory.galprop_sourcekey(source_name=source_name,
                                                                        galpropkey=galkey)
        format_dict['fullpath'] = fullpath
        return self._name_factory.srcmdl_xml(**format_dict)

    def make_ring_filelist(self, sourcekeys, rings, galprop_run):
        """ Make a list of all the template files for a merged component

        Parameters
        ----------

        sourcekeys : list-like of str
            The names of the componenents to merge
        rings : list-like of int
            The indices of the rings to merge
        galprop_run : str
            String identifying the galprop parameters
        """
        flist = []
        for sourcekey in sourcekeys:
            for ring in rings:
                flist += [self.make_ring_filename(sourcekey,
                                                  ring, galprop_run)]
        return flist

    def make_ring_dict(self, galkey):
        """ Make a dictionary mapping the merged component names to list of template files

        Parameters
        ----------

        galkey : str
            Unique key for this ring dictionary

        Returns `model_component.GalpropMergedRingInfo`
        """
        galprop_rings = self.read_galprop_rings_yaml(galkey)
        galprop_run = galprop_rings['galprop_run']
        ring_limits = galprop_rings['ring_limits']
        comp_dict = galprop_rings['diffuse_comp_dict']
        ring_dict = {}
        nring = len(ring_limits) - 1
        for source_name, source_value in comp_dict.items():
            base_dict = dict(source_name=source_name,
                             galkey=galkey,
                             galprop_run=galprop_run)
            for iring in range(nring):
                sourcekey = "%s_%i" % (source_name, iring)
                full_key = "%s_%s" % (sourcekey, galkey)
                rings = range(ring_limits[iring], ring_limits[iring + 1])
                base_dict.update(dict(ring=iring,
                                      sourcekey=sourcekey,
                                      files=self.make_ring_filelist(source_value,
                                                                    rings, galprop_run),
                                      merged_gasmap=self.make_merged_name(sourcekey,
                                                                          galkey, False)))
                ring_dict[full_key] = GalpropMergedRingInfo(**base_dict)
        self._ring_dicts[galkey] = ring_dict
        return ring_dict

    def make_diffuse_comp_info(self, merged_name, galkey):
        """ Make the information about a single merged component

        Parameters
        ----------

        merged_name : str
            The name of the merged component
        galkey : str
            A short key identifying the galprop parameters

        Returns `odel_component.ModelComponentInfo`
        """
        kwargs = dict(source_name=merged_name,
                      source_ver=galkey,
                      model_type='MapCubeSource',
                      Spatial_Filename=self.make_merged_name(
                          merged_name, galkey, fullpath=True),
                      srcmdl_name=self.make_xml_name(merged_name, galkey, fullpath=True))
        return MapCubeComponentInfo(**kwargs)

    def make_diffuse_comp_info_dict(self, galkey):
        """ Make a dictionary maping from merged component to information about that component

        Parameters
        ----------

        galkey : str
            A short key identifying the galprop parameters
        """
        galprop_rings = self.read_galprop_rings_yaml(galkey)
        ring_limits = galprop_rings.get('ring_limits')
        comp_dict = galprop_rings.get('diffuse_comp_dict')
        diffuse_comp_info_dict = {}
        nring = len(ring_limits) - 1
        for source_key in sorted(comp_dict.keys()):
            for iring in range(nring):
                source_name = "%s_%i" % (source_key, iring)
                full_key = "%s_%s" % (source_name, galkey)
                diffuse_comp_info_dict[full_key] =\
                    self.make_diffuse_comp_info(source_name, galkey)
        self._diffuse_comp_info_dicts[galkey] = diffuse_comp_info_dict
        return diffuse_comp_info_dict


class DiffuseModelManager(object):
    """ Small helper class to keep track of diffuse component templates

    This keeps track of the 'diffuse component infomation' dictionary

    This keyed by: key = {source_name}_{source_ver}
    Where:
    {source_name} is something like 'loopI'
    {source_ver} is somthinng like v00

    The dictioary is
    diffuse_comp_info_dict[key] - > `model_component.ModelComponentInfo`

    Note that some components ( those that represent moving sources or are selection depedent )
    will have a sub-dictionary of diffuse_comp_info_dict object for each sub-component

    The compoents are defined in a file called
    config/diffuse_components.yaml
    """

    def __init__(self, **kwargs):
        """ C'tor

        Keyword arguments
        -----------------

        name_policy : str
            Name of yaml file contain file naming policy definitions
        basedir : str
            Top level directory for finding files
        """
        self._name_factory = NameFactory(basedir=kwargs.get('basedir'))
        self._diffuse_comp_info_dict = {}

    @staticmethod
    def read_diffuse_component_yaml(yamlfile):
        """ Read the yaml file for the diffuse components
        """
        diffuse_components = yaml.safe_load(open(yamlfile))
        return diffuse_components

    def sourcekeys(self):
        """Return the list of source keys"""
        return sorted(self._diffuse_comp_info_dict.keys())

    def diffuse_comp_info(self, sourcekey):
        """Return the Component info associated to a particular key
        """
        return self._diffuse_comp_info_dict[sourcekey]

    def make_template_name(self, model_type, sourcekey):
        """ Make the name of a template file for particular component

        Parameters
        ----------

        model_type : str
            Type of model to use for this component
        sourcekey : str
            Key to identify this component

        Returns filename or None if component does not require a template file
        """
        format_dict = self.__dict__.copy()
        format_dict['sourcekey'] = sourcekey
        if model_type == 'IsoSource':
            return self._name_factory.spectral_template(**format_dict)
        elif model_type == 'MapCubeSource':
            return self._name_factory.diffuse_template(**format_dict)
        else:
            raise ValueError("Unexpected model_type %s" % model_type)

    def make_xml_name(self, sourcekey):
        """ Make the name of an xml file for a model definition of a single component

        Parameters
        ----------

        sourcekey : str
            Key to identify this component
        """
        format_dict = self.__dict__.copy()
        format_dict['sourcekey'] = sourcekey
        return self._name_factory.srcmdl_xml(**format_dict)

    def make_diffuse_comp_info(self, source_name, source_ver, diffuse_dict,
                               components=None, comp_key=None):
        """ Make a dictionary mapping the merged component names to list of template files

        Parameters
        ----------

        source_name : str
           Name of the source
        source_ver : str
           Key identifying the version of the source
        diffuse_dict : dict
           Information about this component
        comp_key : str
           Used when we need to keep track of sub-components, i.e.,
           for moving and selection dependent sources.

        Returns `model_component.ModelComponentInfo` or
        `model_component.IsoComponentInfo`
        """
        model_type = diffuse_dict['model_type']
        sourcekey = '%s_%s' % (source_name, source_ver)
        if comp_key is None:
            template_name = self.make_template_name(model_type, sourcekey)
            srcmdl_name = self.make_xml_name(sourcekey)
        else:
            template_name = self.make_template_name(
                model_type, "%s_%s" % (sourcekey, comp_key))
            srcmdl_name = self.make_xml_name("%s_%s" % (sourcekey, comp_key))

        template_name = self._name_factory.fullpath(localpath=template_name)
        srcmdl_name = self._name_factory.fullpath(localpath=srcmdl_name)

        kwargs = dict(source_name=source_name,
                      source_ver=source_ver,
                      model_type=model_type,
                      srcmdl_name=srcmdl_name,
                      components=components,
                      comp_key=comp_key)
        kwargs.update(diffuse_dict)
        if model_type == 'IsoSource':
            kwargs['Spectral_Filename'] = template_name
            return IsoComponentInfo(**kwargs)
        elif model_type == 'MapCubeSource':
            kwargs['Spatial_Filename'] = template_name
            return MapCubeComponentInfo(**kwargs)
        else:
            raise ValueError("Unexpected model type %s" % model_type)

    def make_diffuse_comp_info_dict(self, diffuse_sources, components):
        """ Make a dictionary maping from diffuse component to information about that component

        Parameters
        ----------

        diffuse_sources : dict
            Dictionary with diffuse source defintions
        components : dict
            Dictionary with event selection defintions,
            needed for selection depenedent diffuse components

        Returns
        -------

        ret_dict : dict
            Dictionary mapping sourcekey to `model_component.ModelComponentInfo`
        """
        ret_dict = {}
        for key, value in diffuse_sources.items():
            model_type = value.get('model_type', 'mapcube')
            if model_type == 'galprop_rings':
                continue
            selection_dependent = value.get('selection_dependent', False)
            moving = value.get('moving', False)
            versions = value.get('versions', [])
            for version in versions:
                #sourcekey = self._name_factory.sourcekey(source_name=key,
                #                                         source_ver=version)
                comp_dict = None
                if selection_dependent:
                    # For selection dependent diffuse sources we need to split
                    # by binning component
                    comp_dict = {}
                    for comp in components:
                        comp_key = comp.make_key('{ebin_name}_{evtype_name}')
                        comp_dict[comp_key] = self.make_diffuse_comp_info(
                            key, version, value, None, comp_key)
                elif moving:
                    # For moving diffuse sources we need to split by zmax cut
                    comp_dict = {}
                    zmax_dict = {}
                    for comp in components:
                        zmax_dict[int(comp.zmax)] = True
                    zmax_list = sorted(zmax_dict.keys())
                    for zmax in zmax_list:
                        comp_key = "zmax%i" % (zmax)
                        comp_dict[comp_key] = self.make_diffuse_comp_info(
                            key, version, value, None, comp_key)

                comp_info = self.make_diffuse_comp_info(
                    key, version, value, comp_dict)
                ret_dict[comp_info.sourcekey] = comp_info

        self._diffuse_comp_info_dict.update(ret_dict)
        return ret_dict


def make_ring_dicts(**kwargs):
    """Build and return the information about the Galprop rings
    """
    diffuse_yamlfile = kwargs.pop('diffuse', 'config/diffuse_components.yaml')
    gmm = kwargs.get('GalpropMapManager', GalpropMapManager(**kwargs))
    diffuse_comps = DiffuseModelManager.read_diffuse_component_yaml(
        diffuse_yamlfile)
    for diffuse_value in diffuse_comps.values():
        if diffuse_value['model_type'] != 'galprop_rings':
            continue
        versions = diffuse_value['versions']
        for version in versions:
            gmm.make_ring_dict(version)
    return gmm


def make_diffuse_comp_info_dict(**kwargs):
    """Build and return the information about the diffuse components
    """
    diffuse_yamlfile = kwargs.pop('diffuse', 'config/diffuse_components.yaml')
    components = kwargs.pop('components', None)
    if components is None:
        comp_yamlfile = kwargs.pop('comp', 'config/binning.yaml')
        components = Component.build_from_yamlfile(comp_yamlfile)
    gmm = kwargs.get('GalpropMapManager', GalpropMapManager(**kwargs))
    dmm = kwargs.get('DiffuseModelManager', DiffuseModelManager(**kwargs))
    diffuse_comps = DiffuseModelManager.read_diffuse_component_yaml(
        diffuse_yamlfile)
    diffuse_comp_info_dict = dmm.make_diffuse_comp_info_dict(
        diffuse_comps, components)
    for diffuse_value in diffuse_comps.values():
        if diffuse_value['model_type'] != 'galprop_rings':
            continue
        versions = diffuse_value['versions']
        for version in versions:
            galprop_dict = gmm.make_diffuse_comp_info_dict(version)
            diffuse_comp_info_dict.update(galprop_dict)
    
    return dict(comp_info_dict=diffuse_comp_info_dict,
                GalpropMapManager=gmm,
                DiffuseModelManager=dmm)


class DiffuseComponentChain(Chain):
    """Small class to build srcmaps for diffuse components
    """
    default_options = dict(comp=diffuse_defaults.diffuse['binning_yaml'],
                           data=diffuse_defaults.diffuse['dataset_yaml'],
                           diffuse=diffuse_defaults.diffuse['diffuse_comp_yaml'],
                           make_xml=(False, "Make XML files for diffuse components", bool),
                           dry_run=diffuse_defaults.diffuse['dry_run'])

    def __init__(self, linkname):
        """C'tor
        """
        from fermipy.diffuse.job_library import create_sg_sum_ring_gasmaps, create_sg_vstack_diffuse
        from fermipy.diffuse.gt_srcmap_partial import create_sg_srcmap_partial

        link_gasmaps = create_sg_sum_ring_gasmaps(linkname="%s.merge_galprop"%linkname)

        link_srcmaps = create_sg_srcmap_partial(linkname="%s.srcmaps"%linkname)

        link_vstack_srcmaps = create_sg_vstack_diffuse(linkname="%s.vstack"%linkname)

        Chain.__init__(self, linkname,
                       appname='fermipy-diffuse-chain',
#                       links=[link_gasmaps, link_srcmaps,
#                              link_vstack_srcmaps],
                       links=[link_srcmaps,link_vstack_srcmaps],
                       options=DiffuseComponentChain.default_options.copy(),
                       parser=DiffuseComponentChain._make_parser())

    @staticmethod
    def _make_parser():
        """Make an argument parser for this chain """
        usage = "FIXME [options]"
        description = "Run diffuse component analysis"

        parser = argparse.ArgumentParser(usage=usage, description=description)
        return parser

    def run_argparser(self, argv):
        """Initialize a link with a set of arguments using argparser
        """
        args = Link.run_argparser(self, argv)
        for link in self._links.values():
            link.run_link(stream=sys.stdout, dry_run=True)
        return args

def create_chain_diffuse_comps(**kwargs):
    """Create and return a `DiffuseComponentChain` object """
    chain = DiffuseComponentChain(linkname=kwargs.get('linkname', 'diffuse.diffuse_comps'))
    return chain

def main_chain():
    """Entry point for command line use for single job """
    chain = DiffuseComponentChain("diffuse.diffuse_comps")
    args = chain.run_argparser(sys.argv[1:])
    chain.run_chain(sys.stdout, args.dry_run)
    chain.finalize(args.dry_run)


if __name__ == '__main__':
    main_chain()



