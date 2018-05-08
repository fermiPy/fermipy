# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Small helper classes that wrap information about model components
"""
from __future__ import absolute_import, division, print_function

import copy


class CatalogInfo(object):
    """Information about a source catalog

    Parameters
    ----------

    catalog_name : str
        The name given to the merged component, e.g., merged_CO or merged_HI
    catalog_file : str
        Fits file with catalog data
    catalog_extdir : str
        Directory with extended source templates
    catalog_type : str
        Identifies the format of the catalog fits file: e.g., '3FGL' or '4FGLP'
    catalog : `fermipy.catalog.Catalog`
        Catalog object
    roi_model : `fermipy.roi_model.ROIModel`
        Fermipy object describing all the catalog sources
    srcmdl_name : str
        Name of xml file with the catalog source model
    """

    def __init__(self, **kwargs):
        """C'tor: copies keyword arguments to data members
        """
        self.catalog_name = kwargs.get('catalog_name')
        self.catalog_file = kwargs.get('catalog_file')
        self.catalog_extdir = kwargs.get('catalog_extdir')
        self.catalog_type = kwargs.get('catalog_type')
        self.catalog_table = kwargs.get('catalog_table', None)
        self.catalog = kwargs.get('catalog', None)
        self.roi_model = kwargs.get('roi_model', None)
        self.srcmdl_name = kwargs.get('srcmdl_name', None)
        self.spectrum = kwargs.get('spectrum', None)

    def update(self, **kwargs):
        """Update data members from keyword arguments
        """
        self.__dict__.update(**kwargs)


class GalpropMergedRingInfo(object):
    """Information about a set of Merged Galprop Rings

    Parameters
    ----------

    source_name : str
        The name given to the merged component, e.g., merged_CO or merged_HI
    ring : int
        The index of the merged ring
    sourcekey : str
        Key that identifies this component, e.g., merged_CO_1, or merged_HI_3
    galkey : str
        Key that identifies how to merge the galprop rings, e.g., 'ref'
    galprop_run : str
        Key that idenfifies the galprop run used to make the input rings
    files : str
        List of files of the input gasmap files
    merged_gasmap : str
        Filename for the merged gasmap
    """

    def __init__(self, **kwargs):
        """C'tor: copies keyword arguments to data members
        """
        self.source_name = kwargs.get('source_name')
        self.ring = kwargs.get('ring')
        self.sourcekey = kwargs.get(
            'sourcekey', "%s_%i" % (self.source_name, self.ring))
        self.galkey = kwargs.get('galkey', None)
        self.galprop_run = kwargs.get('galprop_run', None)
        self.files = kwargs.get('files', [])
        self.merged_gasmap = kwargs.get('merged_gasmap', None)

    def update(self, **kwargs):
        """Update data members from keyword arguments
        """
        self.__dict__.update(**kwargs)


class ModelComponentInfo(object):
    """Information about a model component

    Parameters
    ----------

    source_name : str
        The name given to the component, e.g., loop_I or moon
    source_ver : str
        Key to indentify the model version of the source, e.g., v00
    sourcekey : str
        Key that identifies this component, e.g., loop_I_v00 or moon_v00
    model_type : str
        Type of model, 'MapCubeSource' | 'IsoSource' | 'CompositeSource' | 'Catalog' | 'PointSource'
    srcmdl_name : str
        Name of the xml file with the xml just for this component
    moving : bool
        Flag for moving sources (i.e., the sun and moon)
    selection_dependent : bool
        Flag for selection dependent sources (i.e., the residual cosmic ray model)
    no_psf : bool
        Flag to indicate that we do not smear this component with the PSF
    components : dict
        Sub-dictionary of `ModelComponentInfo` objects for moving and selection_dependent sources
    comp_key : str
        Component key for this component of moving and selection_dependent sources
    """

    def __init__(self, **kwargs):
        """C'tor: copies keyword arguments to data members
        """
        self.source_name = kwargs.get('source_name')
        self.source_ver = kwargs.get('source_ver')
        self.sourcekey = kwargs.get('sourcekey', '%s_%s' % (
            self.source_name, self.source_ver))
        self.srcmdl_name = kwargs.get('srcmdl_name')
        self.moving = kwargs.get('moving', False)
        self.selection_dependent = kwargs.get('selection_dependent', False)
        self.no_psf = kwargs.get('no_psf', False)
        self.components = kwargs.get('components', None)
        self.comp_key = kwargs.get('comp_key', None)

    def update(self, **kwargs):
        """Update data members from keyword arguments
        """
        self.__dict__.update(**kwargs)

    def get_component_info(self, comp):
        """Return the information about sub-component specific to a particular data selection

        Parameters
        ----------

        comp : `binning.Component` object
            Specifies the sub-component

        Returns `ModelComponentInfo` object
        """
        if self.components is None:
            raise ValueError(
                'Model component %s does not have sub-components' % self.sourcekey)
        if self.moving:
            comp_key = "zmax%i" % (comp.zmax)
        elif self.selection_dependent:
            comp_key = comp.make_key('{ebin_name}_{evtype_name}')
        else:
            raise ValueError(
                'Model component %s is not moving or selection dependent' % self.sourcekey)
        return self.components[comp_key]

    def add_component_info(self, compinfo):
        """Add sub-component specific information to a particular data selection

        Parameters
        ----------

        compinfo : `ModelComponentInfo` object
            Sub-component being added
        """
        if self.components is None:
            self.components = {}
        self.components[compinfo.comp_key] = compinfo

    def clone_and_merge_sub(self, key):
        """Clones self and merges clone with sub-component specific information

        Parameters
        ----------

        key : str
            Key specifying which sub-component

        Returns `ModelComponentInfo` object
        """
        new_comp = copy.deepcopy(self)
        #sub_com = self.components[key]
        new_comp.components = None
        new_comp.comp_key = key
        return new_comp


class MapCubeComponentInfo(ModelComponentInfo):
    """ Information about a model component represented by a MapCubeSource

    Parameters
    ----------

    Spatial_Filename : str
        Name of the template file for the spatial model
    """

    def __init__(self, **kwargs):
        """C'tor: copies keyword arguments to data members
        """
        super(MapCubeComponentInfo, self).__init__(**kwargs)
        self.model_type = 'MapCubeSource'
        self.Spatial_Filename = kwargs.get('Spatial_Filename', None)


class IsoComponentInfo(ModelComponentInfo):
    """ Information about a model component represented by a IsoSource

    Parameters
    ----------

    Spectral_Filename : str
        Name of the template file for the spatial model
    """

    def __init__(self, **kwargs):
        """C'tor: copies keyword arguments to data members
        """
        super(IsoComponentInfo, self).__init__(**kwargs)
        self.model_type = 'IsoSource'
        self.Spectral_Filename = kwargs.get('Spectral_Filename', None)


class PointSourceInfo(ModelComponentInfo):
    """ Information about a model component represented by a PointSource\
    """

    def __init__(self, **kwargs):
        """C'tor: copies keyword arguments to data members
        """
        super(PointSourceInfo, self).__init__(**kwargs)
        self.model_type = 'PointSource'


class CompositeSourceInfo(ModelComponentInfo):
    """ Information about a model component represented by a CompositeSource

    Parameters
    ----------

    source_names : list
        The names of the nested sources
    catalog_info : `model_component.CatalogInfo` or None
        Information about the catalog containing the nested sources
    roi_model : `fermipy.roi_model.ROIModel`
        Fermipy object describing the nested sources
    """

    def __init__(self, **kwargs):
        """C'tor: copies keyword arguments to data members
        """
        super(CompositeSourceInfo, self).__init__(**kwargs)
        self.model_type = 'CompositeSource'
        self.source_names = kwargs.get('source_names', [])
        self.catalog_info = kwargs.get('catalog_info', None)
        self.roi_model = kwargs.get('roi_model', None)


class CatalogSourcesInfo(ModelComponentInfo):
    """ Information about a model component consisting of sources from a catalog

    Parameters
    ----------

    source_names : list
        The names of the nested sources
    catalog_info : `model_component.CatalogInfo` or None
        Information about the catalog containing the nested sources
    roi_model : `fermipy.roi_model.ROIModel`
        Fermipy object describing the nested sources
    """

    def __init__(self, **kwargs):
        """C'tor: copies keyword arguments to data members
        """
        super(CatalogSourcesInfo, self).__init__(**kwargs)
        self.model_type = 'CatalogSources'
        self.source_names = kwargs.get('source_names', [])
        self.catalog_info = kwargs.get('catalog_info', None)
        self.roi_model = kwargs.get('roi_model', None)
