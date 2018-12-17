# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Classes and utilities that manage catalog sources
"""
from __future__ import absolute_import, division, print_function

import os

import yaml
import numpy as np

from fermipy.diffuse.name_policy import NameFactory
from fermipy.diffuse.source_factory import SourceFactory
from fermipy.diffuse.model_component import CatalogInfo, CompositeSourceInfo, CatalogSourcesInfo


def mask_extended(cat_table):
    """Remove rows representing extended sources from a catalog table
    """
    return np.invert(select_extended(cat_table))


def select_extended(cat_table):
    """Select only rows representing extended sources from a catalog table
    """
    try:
        l = [len(row.strip()) > 0 for row in cat_table['Extended_Source_Name'].data]
        return np.array(l, bool)
    except KeyError:
        return cat_table['Extended']


def make_mask(cat_table, cut):
    """Mask a bit mask selecting the rows that pass a selection
    """
    cut_var = cut['cut_var']
    min_val = cut.get('min_val', None)
    max_val = cut.get('max_val', None)

    nsrc = len(cat_table)
    if min_val is None:
        min_mask = np.ones((nsrc), bool)
    else:
        min_mask = cat_table[cut_var] >= min_val
    if max_val is None:
        max_mask = np.ones((nsrc), bool)
    else:
        max_mask = cat_table[cut_var] <= max_val
    full_mask = min_mask * max_mask
    return full_mask


def select_sources(cat_table, cuts):
    """Select only rows passing a set of cuts from catalog table
    """
    nsrc = len(cat_table)
    full_mask = np.ones((nsrc), bool)
    for cut in cuts:
        if cut == 'mask_extended':
            full_mask *= mask_extended(cat_table)
        elif cut == 'select_extended':
            full_mask *= select_extended(cat_table)
        else:
            full_mask *= make_mask(cat_table, cut)

    lout = [src_name.strip() for src_name in cat_table['Source_Name'][full_mask]]
    return lout


class CatalogSourceManager(object):
    """ Small helper class to keep track of how we deal with catalog sources

    This keeps track of two dictionaries

    One of the dictionaries is keyed by catalog name, and contains information
    about complete catalogs
    catalog_comp_info_dicts[catalog_name] : `model_component.CatalogInfo`

    The other dictionary is keyed by [{catalog_name}_{split_ver}][{split_key}]
    Where:
    {catalog_name} is something like '3FGL'
    {split_ver} is somthing like 'v00' and specifes how to divide sources in the catalog
    {split_key} refers to a specific sub-selection of sources

    split_comp_info_dicts[splitkey] : `model_component.ModelComponentInfo`
    """

    def __init__(self, **kwargs):
        """ C'tor

        Keyword arguments
        -----------------

        basedir : str
            Top level directory for finding files
        """
        self._name_factory = NameFactory(**kwargs)
        self._catalog_comp_info_dicts = {}
        self._split_comp_info_dicts = {}

    def read_catalog_info_yaml(self, splitkey):
        """ Read the yaml file for a particular split key
        """
        catalog_info_yaml = self._name_factory.catalog_split_yaml(sourcekey=splitkey,
                                                                  fullpath=True)
        yaml_dict = yaml.safe_load(open(catalog_info_yaml))
        # resolve env vars
        yaml_dict['catalog_file'] = os.path.expandvars(yaml_dict['catalog_file'])
        yaml_dict['catalog_extdir'] = os.path.expandvars(yaml_dict['catalog_extdir'])
        return yaml_dict

    def build_catalog_info(self, catalog_info):
        """ Build a CatalogInfo object """
        cat = SourceFactory.build_catalog(**catalog_info)
        catalog_info['catalog'] = cat
        # catalog_info['catalog_table'] =
        #    Table.read(catalog_info['catalog_file'])
        catalog_info['catalog_table'] = cat.table
        catalog_info['roi_model'] =\
            SourceFactory.make_fermipy_roi_model_from_catalogs([cat])
        catalog_info['srcmdl_name'] =\
            self._name_factory.srcmdl_xml(sourcekey=catalog_info['catalog_name'])
        return CatalogInfo(**catalog_info)

    def catalogs(self):
        """ Return the list of full catalogs used """
        return sorted(self._catalog_comp_info_dicts.keys())

    def splitkeys(self):
        """ Return the list of catalog split keys used """
        return sorted(self._split_comp_info_dicts.keys())

    def catalog_comp_info_dict(self, catkey):
        """ Return the roi_model for an entire catalog """
        return self._catalog_comp_info_dicts[catkey]

    def split_comp_info_dict(self, catalog_name, split_ver):
        """ Return the information about a particular scheme for how to handle catalog sources """
        return self._split_comp_info_dicts["%s_%s" % (catalog_name, split_ver)]

    def catalog_components(self, catalog_name, split_ver):
        """ Return the set of merged components for a particular split key """
        return sorted(self._split_comp_info_dicts["%s_%s" % (catalog_name, split_ver)].keys())

    def split_comp_info(self, catalog_name, split_ver, split_key):
        """ Return the info for a particular split key """
        return self._split_comp_info_dicts["%s_%s" % (catalog_name, split_ver)][split_key]

    def make_catalog_comp_info(self, full_cat_info, split_key, rule_key, rule_val, sources):
        """ Make the information about a single merged component

        Parameters
        ----------

        full_cat_info : `_model_component.CatalogInfo`
            Information about the full catalog
        split_key : str
            Key identifying the version of the spliting used
        rule_key : str
            Key identifying the specific rule for this component
        rule_val : list
            List of the cuts used to define this component
        sources : list
            List of the names of the sources in this component

        Returns `CompositeSourceInfo` or `CatalogSourcesInfo`
        """
        merge = rule_val.get('merge', True)
        sourcekey = "%s_%s_%s" % (
            full_cat_info.catalog_name, split_key, rule_key)
        srcmdl_name = self._name_factory.srcmdl_xml(sourcekey=sourcekey)
        srcmdl_name = self._name_factory.fullpath(localpath=srcmdl_name)
        kwargs = dict(source_name="%s_%s" % (full_cat_info.catalog_name, rule_key),
                      source_ver=split_key,
                      sourcekey=sourcekey,
                      srcmdl_name=srcmdl_name,
                      source_names=sources,
                      catalog_info=full_cat_info,
                      roi_model=SourceFactory.copy_selected_sources(full_cat_info.roi_model,
                                                                    sources))
        if merge:
            return CompositeSourceInfo(**kwargs)
        return CatalogSourcesInfo(**kwargs)

    def make_catalog_comp_info_dict(self, catalog_sources):
        """ Make the information about the catalog components

        Parameters
        ----------

        catalog_sources : dict
            Dictionary with catalog source defintions

        Returns
        -------

        catalog_ret_dict : dict
            Dictionary mapping catalog_name to `model_component.CatalogInfo`
        split_ret_dict : dict
            Dictionary mapping sourcekey to `model_component.ModelComponentInfo`
        """
        catalog_ret_dict = {}
        split_ret_dict = {}
        for key, value in catalog_sources.items():
            if value is None:
                continue
            if value['model_type'] != 'catalog':
                continue
            versions = value['versions']
            for version in versions:
                ver_key = "%s_%s" % (key, version)
                source_dict = self.read_catalog_info_yaml(ver_key)
                try:
                    full_cat_info = catalog_ret_dict[key]
                except KeyError:
                    full_cat_info = self.build_catalog_info(source_dict)
                    catalog_ret_dict[key] = full_cat_info

                try:
                    all_sources = [x.strip() for x in full_cat_info.catalog_table[
                        'Source_Name'].astype(str).tolist()]
                except KeyError:
                    print(full_cat_info.catalog_table.colnames)
                used_sources = []
                rules_dict = source_dict['rules_dict']
                split_dict = {}
                for rule_key, rule_val in rules_dict.items():
                    # full_key =\
                    #    self._name_factory.merged_sourcekey(catalog=ver_key,
                    #                                        rulekey=rule_key)
                    sources = select_sources(
                        full_cat_info.catalog_table, rule_val['cuts'])
                    used_sources.extend(sources)
                    split_dict[rule_key] = self.make_catalog_comp_info(
                        full_cat_info, version, rule_key, rule_val, sources)

                # Now deal with the remainder
                for source in used_sources:
                    try:
                        all_sources.remove(source)
                    except ValueError:
                        continue
                rule_val = dict(cuts=[],
                                merge=source_dict['remainder'].get('merge', False))
                split_dict['remain'] = self.make_catalog_comp_info(
                    full_cat_info, version, 'remain', rule_val, all_sources)

                # Merge in the info for this version of splits
                split_ret_dict[ver_key] = split_dict

        self._catalog_comp_info_dicts.update(catalog_ret_dict)
        self._split_comp_info_dicts.update(split_ret_dict)
        return (catalog_ret_dict, split_ret_dict)


def make_catalog_comp_dict(**kwargs):
    """Build and return the information about the catalog components
    """
    library_yamlfile = kwargs.pop('library', 'models/library.yaml')
    csm = kwargs.pop('CatalogSourceManager', CatalogSourceManager(**kwargs))
    if library_yamlfile is None or library_yamlfile == 'None':
        yamldict = {}
    else:
        yamldict = yaml.safe_load(open(library_yamlfile))
    catalog_info_dict, comp_info_dict = csm.make_catalog_comp_info_dict(yamldict)
    return dict(catalog_info_dict=catalog_info_dict,
                comp_info_dict=comp_info_dict,
                CatalogSourceManager=csm)
