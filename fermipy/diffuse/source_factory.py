# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Classes and utilities that create fermipy source objects
"""
from __future__ import absolute_import, division, print_function

from collections import OrderedDict

from astropy.table import Table
from astropy.coordinates import SkyCoord
from fermipy import roi_model
from fermipy import catalog


def make_point_source(name, src_dict):
    """Construct and return a `fermipy.roi_model.Source` object
    """
    return roi_model.Source(name, src_dict)


def make_mapcube_source(name, Spatial_Filename, spectrum):
    """Construct and return a `fermipy.roi_model.MapCubeSource` object
    """
    data = dict(Spatial_Filename=Spatial_Filename)
    if spectrum is not None:
        data.update(spectrum)

    return roi_model.MapCubeSource(name, data)


def make_isotropic_source(name, Spectrum_Filename, spectrum):
    """Construct and return a `fermipy.roi_model.IsoSource` object
    """
    data = dict(Spectrum_Filename=Spectrum_Filename)
    if spectrum is not None:
        data.update(spectrum)

    return roi_model.IsoSource(name, data)


def make_composite_source(name, spectrum):
    """Construct and return a `fermipy.roi_model.CompositeSource` object
    """
    data = dict(SpatialType='CompositeSource',
                SpatialModel='CompositeSource',
                SourceType='CompositeSource')
    if spectrum is not None:
        data.update(spectrum)
    return roi_model.CompositeSource(name, data)


def make_catalog_sources(catalog_roi_model, source_names):
    """Construct and return dictionary of sources that are a subset of sources
    in catalog_roi_model.

    Parameters
    ----------

    catalog_roi_model : dict or `fermipy.roi_model.ROIModel`
        Input set of sources

    source_names : list
        Names of sourcs to extract

    Returns dict mapping source_name to `fermipy.roi_model.Source` object
    """
    sources = {}
    for source_name in source_names:
        sources[source_name] = catalog_roi_model[source_name]
    return sources


def make_sources(comp_key, comp_dict):
    """Make dictionary mapping component keys to a source
    or set of sources

    Parameters
    ----------

    comp_key : str
        Key used to access sources

    comp_dict : dict
        Information used to build sources

    return `OrderedDict` maping comp_key to `fermipy.roi_model.Source`
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
                                                  spectrum)
    elif model_type == 'CatalogSources':
        srcdict.update(make_catalog_sources(comp_info.roi_model,
                                            comp_info.source_names))
    else:
        raise ValueError("Unrecognized model_type %s" % model_type)
    return srcdict


class SourceFactory(object):
    """Small helper class to build and keep track of sources
    """

    def __init__(self):
        """C'tor
        """
        self._source_info_dict = OrderedDict()
        self._sources = OrderedDict()

    @property
    def sources(self):
        """Return the dictionary of sources"""
        return self._sources

    @property
    def source_info_dict(self):
        """Return the dictionary of source_info objects used to build sources"""
        return self._source_info_dict

    def add_sources(self, source_info_dict):
        """Add all of the sources in source_info_dict to this factory
        """
        self._source_info_dict.update(source_info_dict)
        for key, value in source_info_dict.items():
            self._sources.update(make_sources(key, value))

    @staticmethod
    def build_catalog(**kwargs):
        """Build a `fermipy.catalog.Catalog` object

        Parameters
        ----------

        catalog_type : str
            Specifies catalog type, options include 2FHL | 3FGL | 4FGLP
        catalog_file : str
            FITS file with catalog tables
        catalog_extdir : str
            Path to directory with extended source templates
        """
        catalog_type = kwargs.get('catalog_type')
        catalog_file = kwargs.get('catalog_file')
        catalog_extdir = kwargs.get('catalog_extdir')
        if catalog_type == '2FHL':
            return catalog.Catalog2FHL(fitsfile=catalog_file, extdir=catalog_extdir)
        elif catalog_type == '3FGL':
            return catalog.Catalog3FGL(fitsfile=catalog_file, extdir=catalog_extdir)
        elif catalog_type == '4FGLP':
            return catalog.Catalog4FGLP(fitsfile=catalog_file, extdir=catalog_extdir)
        elif catalog_type == 'FL8Y':
            return catalog.CatalogFL8Y(fitsfile=catalog_file, extdir=catalog_extdir)
        else:
            table = Table.read(catalog_file)
        return catalog.Catalog(table, extdir=catalog_extdir)

    @staticmethod
    def make_fermipy_roi_model_from_catalogs(cataloglist):
        """Build and return a `fermipy.roi_model.ROIModel object from
        a list of fermipy.catalog.Catalog` objects
        """
        data = dict(catalogs=cataloglist,
                    src_roiwidth=360.)
        return roi_model.ROIModel(data, skydir=SkyCoord(0.0, 0.0, unit='deg'))

    @classmethod
    def make_roi(cls, sources=None):
        """Build and return a `fermipy.roi_model.ROIModel` object from
        a dict with information about the sources
        """
        if sources is None:
            sources = {}
        src_fact = cls()
        src_fact.add_sources(sources)
        ret_model = roi_model.ROIModel(
            {}, skydir=SkyCoord(0.0, 0.0, unit='deg'))
        for source in src_fact.sources.values():
            ret_model.load_source(source,
                                  build_index=False, merge_sources=False)
        return ret_model

    @classmethod
    def copy_selected_sources(cls, roi, source_names):
        """Build and return a `fermipy.roi_model.ROIModel` object
        by copying selected sources from another such object
        """
        roi_new = cls.make_roi()
        for source_name in source_names:
            try:
                src_cp = roi.copy_source(source_name)
            except Exception:
                continue
            roi_new.load_source(src_cp, build_index=False)
        return roi_new
