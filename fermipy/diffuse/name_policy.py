# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Handle the naming conventions for composite likelihood analysis
"""
from __future__ import absolute_import, division, print_function

import yaml

# Map reprocessing 'key' to IRF name
DATASET_DICTIONARY = dict(P8_P302='P8R2', P8_P304='P8R3', P8_P305='P8R3')

# Map event class 'key' to IRF name
EVCLASS_NAME_DICTIONARY = dict(source='SOURCE',
                               clean='CLEAN',
                               ultraclean='ULTRACLEAN',
                               ultracleanveto='ULTRACLEANVETO')

# Map event class 'key' to evtype bit mask value
EVCLASS_MASK_DICTIONARY = dict(source=128,
                               clean=256,
                               ultraclean=512,
                               ultracleanveto=1024)


class NameFactory(object):
    """ Helper class to define file names and keys consistently. """

    # dataset: specifies the data selection,
    # e.g., P8_P302_8years_source or P8_P302_8years_ultracleanveto
    dataset_format = '{data_pass}_{data_ver}_{data_time}_{evclass}'

    # Binning component : specifies the sub-selection,
    # e.g., zmax105_E3_PSF3 or zmax100_E1_ALL
    component_format = '{zcut}_{ebin}_{psftype}'

    # sourcekeys, these are how we specify sources

    # sourcekey for diffuse templates : specifies the source and version of the source
    # e.g., loopI_v00 or
    sourcekey_format = '{source_name}_{source_ver}'
    # sourcekey for galprop input maps : specifies the component and ring.
    # e.g., pi0_decay_HIR_ring_11
    galprop_ringkey_format = '{source_name}_{ringkey}'
    # sourcekey for merged galprop maps : specifies the merged component and merging scheme
    # e.g., merged_CO_0_ref
    galprop_sourcekey_format = '{source_name}_{galpropkey}'
    # sourcekey for merged sets of point sources : specifies the catalog and merging rule
    # e.g., 3FLG_v00_faint
    merged_sourcekey_format = '{catalog}_{rulekey}'

    # File formats

    # Galprop inputs
    # Galprop input gasmaps
    galprop_gasmap_format = 'gasmap/{sourcekey}_{projtype}_{galprop_run}.gz'
    # Galprop merged gasmaps
    merged_gasmap_format = 'merged_gasmaps/{sourcekey}_{projtype}.fits'

    # Other diffuse map templates
    diffuse_template_format = 'templates/template_{sourcekey}.fits'
    # Spectral templates
    spectral_template_format = 'templates/spectral_{sourcekey}.txt'

    # Source model xml files (input to gtrsrcmaps and gtlike)
    srcmdl_xml_format = 'srcmdls/{sourcekey}.xml'
    nested_srcmdl_xml_format = 'srcmdls/{sourcekey}_sources.xml'

    # ScienceTools output files
    # The input ft1 file list
    ft1file_format = '{dataset}_{zcut}.lst'
    # The input ft2 file list
    ft2file_format = 'ft2_files/ft2_{data_time}.lst'

    # Livetime cubes (output of gtltcube)
    ltcube_format = 'lt_cubes/ltcube_{data_time}_{mktime}_{zcut}.fits'
    # Selected events (output of gtselect)
    select_format = 'counts_cubes/select_{dataset}_{component}.fits'
    # MKtime select events (output of gtmktime)
    mktime_format = 'counts_cubes/mktime_{dataset}_{mktime}_{component}.fits'

    # Counts cubes (output of gtbin)
    ccube_format = 'counts_cubes/ccube_{dataset}_{mktime}_{component}_{coordsys}.fits'
    # Binned exposure cubes (output of gtexpcube2)
    bexpcube_format = 'bexp_cubes/bexcube_{dataset}_{mktime}_{component}_{coordsys}_{irf_ver}.fits'
    # Sources maps (output of gtsrcmaps)
    srcmaps_format = 'srcmaps/srcmaps_{sourcekey}_{dataset}_{mktime}_{component}_{coordsys}_{irf_ver}.fits'
    # Model cubes (output of gtmodel outtype=CCUBE)
    mcube_format = 'model_cubes/mcube_{sourcekey}_{dataset}_{mktime}_{component}_{coordsys}_{irf_ver}.fits'

    # SolarTools output files
    # gtltcubesun output (for sun)
    ltcubesun_format = 'sunmoon/ltcube_{data_time}_{mktime}_{zcut}_sun.fits'
    # gtltcubesun output (for moon)
    ltcubemoon_format = 'sunmoon/ltcube_{data_time}_{mktime}_{zcut}_moon.fits'
    # Binned exposure cubes (output of gtexphpsun, for sun)
    bexpcubesun_format = 'bexp_cubes/bexcube_{dataset}_{mktime}_{component}_{irf_ver}_sun.fits'
    # Binned exposure cubes (output of gtexphpsun, for moon)
    bexpcubemoon_format = 'bexp_cubes/bexcube_{dataset}_{mktime}_{component}_{irf_ver}_moon.fits'
    # Angular spectrum profile
    angprofile_format = 'templates/profile_{sourcekey}.fits'

    # Binned exposure cubes (output of gtexphpsun, for sun)
    templatesunmoon_format = 'templates/template_{sourcekey}_{zcut}.fits'

    # residual CR output files
    residual_cr_format = 'residual_cr/residual_cr_{dataset}_{mktime}_{component}_{coordsys}_{irf_ver}.fits'

    # Model specific stuff

    # galprop rings merging yaml file
    galprop_rings_yaml_format = 'models/galprop_rings_{galkey}.yaml'
    # catalog split yaml file
    catalog_split_yaml_format = 'models/catalog_{sourcekey}.yaml'
    # model yaml file
    model_yaml_format = 'models/model_{modelkey}.yaml'

    # Merged source map file for one binning component
    merged_srcmaps_format =\
        'analysis/model_{modelkey}/srcmaps_{dataset}_{mktime}_{component}_{coordsys}_{irf_ver}.fits'
    # Master XML model file
    master_srcmdl_xml_format = 'analysis/model_{modelkey}/srcmdl_{modelkey}_master.xml'
    # Component XML model file
    comp_srcmdl_xml_format = 'analysis/model_{modelkey}/srcmdl_{modelkey}_{component}.xml'

    # Stamp files from scatter gather jobs
    stamp_format = 'stamps/{linkname}.stamp'

    # Full filepath
    fullpath_format = '{basedir}/{localpath}'

    def __init__(self, **kwargs):
        """ C'tor.  Set baseline dictionary used to resolve names
        """
        self.base_dict = kwargs.copy()

    def _replace_none(self, aDict):
        """ Replace all None values in a dict with 'none' """
        for k, v in aDict.items():
            if v is None:
                aDict[k] = 'none'

    def update_base_dict(self, yamlfile):
        """ Update the values in baseline dictionary used to resolve names
        """
        self.base_dict.update(**yaml.safe_load(open(yamlfile)))

    def irf_ver(self, **kwargs):
        """ Get the name of the IRF version
        """
        return kwargs.get('irf_ver', self.base_dict.get('irf_ver', None))

    def irfs(self, **kwargs):
        """ Get the name of IFRs associted with a particular dataset
        """
        dsval = kwargs.get('dataset', self.dataset(**kwargs))
        tokens = dsval.split('_')
        irf_name = "%s_%s_%s" % (DATASET_DICTIONARY['%s_%s' % (tokens[0], tokens[1])],
                                 EVCLASS_NAME_DICTIONARY[tokens[3]],
                                 kwargs.get('irf_ver'))
        return irf_name

    def evclassmask(self, evclass_str):
        """ Get the bitmask for a particular event class
        """
        return EVCLASS_MASK_DICTIONARY[evclass_str]

    def dataset(self, **kwargs):
        """ Return a key that specifies the data selection
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        self._replace_none(kwargs_copy)
        try:
            return NameFactory.dataset_format.format(**kwargs_copy)
        except KeyError:
            return None

    def component(self, **kwargs):
        """ Return a key that specifies data the sub-selection
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        self._replace_none(kwargs_copy)        
        try:
            return NameFactory.component_format.format(**kwargs_copy)
        except KeyError:
            return None

    def sourcekey(self, **kwargs):
        """ Return a key that specifies the name and version of a source or component
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        self._replace_none(kwargs_copy)        
        try:
            return NameFactory.sourcekey_format.format(**kwargs_copy)
        except KeyError:
            return None

    def galprop_ringkey(self, **kwargs):
        """ return the sourcekey for galprop input maps : specifies the component and ring
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        self._replace_none(kwargs_copy)        
        try:
            return NameFactory.galprop_ringkey_format.format(**kwargs_copy)
        except KeyError:
            return None

    def galprop_sourcekey(self, **kwargs):
        """ return the sourcekey for merged galprop maps :
        specifies the merged component and merging scheme
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        self._replace_none(kwargs_copy)        
        try:
            return NameFactory.galprop_sourcekey_format.format(**kwargs_copy)
        except KeyError:
            return None

    def merged_sourcekey(self, **kwargs):
        """ return the sourcekey for merged sets of point sources :
        specifies the catalog and merging rule
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        self._replace_none(kwargs_copy)        
        try:
            return NameFactory.merged_sourcekey_format.format(**kwargs_copy)
        except KeyError:
            return None

    def galprop_gasmap(self, **kwargs):
        """ return the file name for Galprop input gasmaps
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        self._replace_none(kwargs_copy)        
        localpath = NameFactory.galprop_gasmap_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def merged_gasmap(self, **kwargs):
        """ return the file name for Galprop merged gasmaps
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        self._replace_none(kwargs_copy)        
        localpath = NameFactory.merged_gasmap_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def diffuse_template(self, **kwargs):
        """ return the file name for other diffuse map templates
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        self._replace_none(kwargs_copy)        
        localpath = NameFactory.diffuse_template_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def spectral_template(self, **kwargs):
        """ return the file name for spectral templates
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        localpath = NameFactory.spectral_template_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def srcmdl_xml(self, **kwargs):
        """ return the file name for source model xml files
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        localpath = NameFactory.srcmdl_xml_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def nested_srcmdl_xml(self, **kwargs):
        """ return the file name for source model xml files of nested sources
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        self._replace_none(kwargs_copy)        
        localpath = NameFactory.nested_srcmdl_xml_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def ft1file(self, **kwargs):
        """ return the name of the input ft1 file list
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        kwargs_copy['dataset'] = kwargs.get('dataset', self.dataset(**kwargs))
        self._replace_none(kwargs_copy)        
        localpath = NameFactory.ft1file_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def ft2file(self, **kwargs):
        """ return the name of the input ft2 file list
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        kwargs_copy['data_time'] = kwargs.get(
            'data_time', self.dataset(**kwargs))
        self._replace_none(kwargs_copy)        
        localpath = NameFactory.ft2file_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def ltcube(self, **kwargs):
        """ return the name of a livetime cube file
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        kwargs_copy['dataset'] = kwargs.get('dataset', self.dataset(**kwargs))
        localpath = NameFactory.ltcube_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def select(self, **kwargs):
        """ return the name of a selected events ft1file
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        kwargs_copy['dataset'] = kwargs.get('dataset', self.dataset(**kwargs))
        kwargs_copy['component'] = kwargs.get(
            'component', self.component(**kwargs))
        self._replace_none(kwargs_copy)        
        localpath = NameFactory.select_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def mktime(self, **kwargs):
        """ return the name of a selected events ft1file
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        kwargs_copy['dataset'] = kwargs.get('dataset', self.dataset(**kwargs))
        kwargs_copy['component'] = kwargs.get(
            'component', self.component(**kwargs))
        self._replace_none(kwargs_copy)        
        localpath = NameFactory.mktime_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def ccube(self, **kwargs):
        """ return the name of a counts cube file
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        kwargs_copy['dataset'] = kwargs.get('dataset', self.dataset(**kwargs))
        kwargs_copy['component'] = kwargs.get(
            'component', self.component(**kwargs))
        localpath = NameFactory.ccube_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def bexpcube(self, **kwargs):
        """ return the name of a binned exposure cube file
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        kwargs_copy['dataset'] = kwargs.get('dataset', self.dataset(**kwargs))
        kwargs_copy['component'] = kwargs.get(
            'component', self.component(**kwargs))
        self._replace_none(kwargs_copy)        
        localpath = NameFactory.bexpcube_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def srcmaps(self, **kwargs):
        """ return the name of a source map file
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        kwargs_copy['dataset'] = kwargs.get('dataset', self.dataset(**kwargs))
        kwargs_copy['component'] = kwargs.get(
            'component', self.component(**kwargs))
        self._replace_none(kwargs_copy)        
        localpath = NameFactory.srcmaps_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def mcube(self, **kwargs):
        """ return the name of a model cube file
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        kwargs_copy['dataset'] = kwargs.get('dataset', self.dataset(**kwargs))
        kwargs_copy['component'] = kwargs.get(
            'component', self.component(**kwargs))
        self._replace_none(kwargs_copy)        
        localpath = NameFactory.mcube_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def ltcube_sun(self, **kwargs):
        """ return the name of a livetime cube file
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        kwargs_copy['dataset'] = kwargs.get('dataset', self.dataset(**kwargs))
        self._replace_none(kwargs_copy)        
        localpath = NameFactory.ltcubesun_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def ltcube_moon(self, **kwargs):
        """ return the name of a livetime cube file
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        kwargs_copy['dataset'] = kwargs.get('dataset', self.dataset(**kwargs))
        self._replace_none(kwargs_copy)        
        localpath = NameFactory.ltcubemoon_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def bexpcube_sun(self, **kwargs):
        """ return the name of a binned exposure cube file
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        kwargs_copy['dataset'] = kwargs.get('dataset', self.dataset(**kwargs))
        kwargs_copy['component'] = kwargs.get(
            'component', self.component(**kwargs))
        self._replace_none(kwargs_copy)        
        localpath = NameFactory.bexpcubesun_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def bexpcube_moon(self, **kwargs):
        """ return the name of a binned exposure cube file
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        kwargs_copy['dataset'] = kwargs.get('dataset', self.dataset(**kwargs))
        kwargs_copy['component'] = kwargs.get(
            'component', self.component(**kwargs))
        self._replace_none(kwargs_copy)        
        localpath = NameFactory.bexpcubemoon_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def angprofile(self, **kwargs):
        """ return the file name for sun or moon angular profiles
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        self._replace_none(kwargs_copy)        
        localpath = NameFactory.angprofile_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def template_sunmoon(self, **kwargs):
        """ return the file name for sun or moon template files
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        kwargs_copy['dataset'] = kwargs.get('dataset', self.dataset(**kwargs))
        kwargs_copy['component'] = kwargs.get(
            'component', self.component(**kwargs))
        self._replace_none(kwargs_copy)        
        localpath = NameFactory.templatesunmoon_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def residual_cr(self, **kwargs):
        """Return the name of the residual CR analysis output files"""
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        kwargs_copy['dataset'] = kwargs.get('dataset', self.dataset(**kwargs))
        kwargs_copy['component'] = kwargs.get(
            'component', self.component(**kwargs))
        localpath = NameFactory.residual_cr_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def galprop_rings_yaml(self, **kwargs):
        """ return the name of a galprop rings merging yaml file
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        self._replace_none(kwargs_copy)        
        localpath = NameFactory.galprop_rings_yaml_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def catalog_split_yaml(self, **kwargs):
        """ return the name of a catalog split yaml file
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        self._replace_none(kwargs_copy)        
        localpath = NameFactory.catalog_split_yaml_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def model_yaml(self, **kwargs):
        """ return the name of a model yaml file
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        self._replace_none(kwargs_copy)        
        localpath = NameFactory.model_yaml_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def merged_srcmaps(self, **kwargs):
        """ return the name of a source map file
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        kwargs_copy['dataset'] = kwargs.get('dataset', self.dataset(**kwargs))
        kwargs_copy['component'] = kwargs.get(
            'component', self.component(**kwargs))
        self._replace_none(kwargs_copy)        
        localpath = NameFactory.merged_srcmaps_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def master_srcmdl_xml(self, **kwargs):
        """ return the name of a source model file
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        self._replace_none(kwargs_copy)        
        localpath = NameFactory.master_srcmdl_xml_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def comp_srcmdl_xml(self, **kwargs):
        """ return the name of a source model file
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        kwargs_copy['dataset'] = kwargs.get('dataset', self.dataset(**kwargs))
        kwargs_copy['component'] = kwargs.get(
            'component', self.component(**kwargs))
        self._replace_none(kwargs_copy)        
        localpath = NameFactory.comp_srcmdl_xml_format.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def stamp(self, **kwargs):
        """Return the path for a stamp file for a scatter gather job"""
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        self._replace_none(kwargs_copy)        
        return NameFactory.stamp_format.format(**kwargs_copy)

    def fullpath(self, **kwargs):
        """Return a full path name for a given file
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        self._replace_none(kwargs_copy)        
        return NameFactory.fullpath_format.format(**kwargs_copy)

    def generic(self, input_string, **kwargs):
        """ return a generic filename for a given dataset and component
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        kwargs_copy['dataset'] = kwargs.get('dataset', self.dataset(**kwargs))
        kwargs_copy['component'] = kwargs.get(
            'component', self.component(**kwargs))
        self._replace_none(kwargs_copy)        
        return input_string.format(**kwargs_copy)

    def make_filenames(self, **kwargs):
        """ Make a dictionary of filenames for various types
        """
        out_dict = dict(ft1file=self.ft1file(**kwargs),
                        ltcube=self.ltcube(**kwargs),
                        ccube=self.ccube(**kwargs),
                        bexpcube=self.bexpcube(**kwargs),
                        srcmaps=self.srcmaps(**kwargs),
                        mcube=self.mcube(**kwargs))
        return out_dict
