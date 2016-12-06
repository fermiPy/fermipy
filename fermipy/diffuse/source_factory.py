# Licensed under a 3-clause BSD style license - see LICENSE.rst
""" 
Classes and utilities that create fermipy source objects
"""
from __future__ import absolute_import, division, print_function

import sys
import os

import yaml

from astropy.coordinates import SkyCoord
from collections import OrderedDict
from fermipy import roi_model
from fermipy import catalog


def make_point_source(name, src_dict):
    """
    """
    return roi_model.Source(name, src_dict)


def make_mapcube_source(name, Spatial_Filename, spectrum):
    """
    """
    data = dict(Spatial_Filename=Spatial_Filename)
    if spectrum is not None:
        data.update(spectrum)

    return roi_model.MapCubeSource(name, data)


def make_isotropic_source(name, Spectrum_Filename, spectrum):
    """
    """
    data = dict(Spectrum_Filename=Spectrum_Filename)
    if spectrum is not None:
        data.update(spectrum)

    return roi_model.IsoSource(name, data)


def make_composite_source(name, catalog_roi_model, spectrum, nested_source_names):
    """
    """
    #nested_sources = [ catalog_roi_model[nested_source_name] for nested_source_name in nested_source_names ]
    #data = dict(nested_sources=nested_sources)
    data = {}
    if spectrum is not None:
        data.update(spectrum)

    return roi_model.CompositeSource(name, data)


def make_catalog_sources(catalog_roi_model, source_names):
    """
    """
    sources = {}
    for source_name in source_names:
        sources[source_name] = catalog_roi_model[source_name]
    return sources


def make_sources(comp_key, comp_dict):
    """
    """
    srcdict = OrderedDict()
    try:
        comp_info = comp_dict.info
    except AttributeError:
        comp_info = comp_dict
    try:
        spectrum = comp_dict.spectrum
    except AttributeError:
        spectrum = None

    model_type = comp_info.model_type
    if model_type == 'PointSource':
        srcdict[comp_key] = make_point_source(comp_info.source_name,
                                              comp_info.src_dict)
    elif model_type == 'MapCubeSource':
        srcdict[comp_key] = make_mapcube_source(comp_info.source_name,
                                                comp_info.Spatial_Filename,
                                                spectrum)
    elif model_type == 'IsoSource':
        srcdict[comp_key] = make_isotropic_source(comp_info.source_name,
                                                  comp_info.Spectral_Filename,
                                                  spectrum)
    elif model_type == 'CompositeSource':
        srcdict[comp_key] = make_composite_source(comp_info.source_name,
                                                  comp_info.roi_model,
                                                  spectrum,
                                                  comp_info.source_names)
    elif model_type == 'CatalogSources':
        srcdict.update(make_catalog_sources(comp_info.roi_model,
                                            comp_info.source_names))
    else:
        raise ValueError("Unrecognized model_type %s" % model_type)
    return srcdict


class SourceFactory(object):
    """ 
    """

    def __init__(self):
        """
        """
        self._source_info_dict = OrderedDict()
        self._sources = OrderedDict()

    @property
    def sources(self):
        return self._sources

    @property
    def source_info_dict(self):
        return self._source_info_dict

    def add_sources(self, source_info_dict):
        """
        """
        self._source_info_dict.update(source_info_dict)
        for key, value in source_info_dict.items():
            self._sources.update(make_sources(key, value))

    @staticmethod
    def build_catalog(catalog_type, catalog_file, catalog_extdir, **kwargs):
        """
        """
        print (catalog_file, catalog_extdir)
        if catalog_type == '2FHL':
            return catalog.Catalog2FHL(fitsfile=catalog_file, extdir=catalog_extdir)
        elif catalog_type == '3FGL':
            return catalog.Catalog3FGL(fitsfile=catalog_file, extdir=catalog_extdir)
        elif Catalog4FGLP == '4FGLP':
            return catalog.Catalog4FGLP(fitsfile=catalog_file, extdir=catalog_extdir)
        else:
            table = Table.read(catalog_file)
        return catalog.Catalog(table, extdir=catalog_extdir)

    @staticmethod
    def make_fermipy_roi_model_from_catalogs(cataloglist):
        """
        """
        data = dict(catalogs=cataloglist,
                    src_roiwidth=360.)
        return roi_model.ROIModel(data, skydir=SkyCoord(0.0, 0.0, unit='deg'))

    @staticmethod
    def make_roi(sources={}):
        """
        """
        sf = SourceFactory()
        sf.add_sources(sources)
        ret_model = roi_model.ROIModel(
            {}, skydir=SkyCoord(0.0, 0.0, unit='deg'))
        for source_name, source in sf.sources.items():
            ret_model.load_source(
                source, build_index=False, merge_sources=False)
        return ret_model

    @staticmethod
    def copy_selected_sources(roi, source_names):
        """
        """
        roi_new = SourceFactory.make_roi()
        for source_name in source_names:
            src_cp = roi.copy_source(source_name)
            roi_new.load_source(src_cp, build_index=False)
        return roi_new
