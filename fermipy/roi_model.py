from __future__ import absolute_import, division, print_function, \
    unicode_literals

import os
import copy
import re
import collections
import numpy as np
import xml.etree.cElementTree as ElementTree

import pyLikelihood as pyLike

from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.io.fits as pyfits
from astropy.table import Table, Column

import fermipy
import fermipy.config
import fermipy.utils as utils
import fermipy.wcs_utils as wcs_utils
import fermipy.gtutils as gtutils
import fermipy.catalog as catalog
import fermipy.defaults as defaults
from fermipy.logger import Logger
from fermipy.logger import logLevel as ll


def resolve_file_path(path, **kwargs):
    dirs = kwargs.get('search_dirs', [])

    if os.path.isabs(os.path.expandvars(path)) and \
            os.path.isfile(os.path.expandvars(path)):
        return path

    for d in dirs:
        if not os.path.isdir(os.path.expandvars(d)):
            continue
        p = os.path.join(d, path)
        if os.path.isfile(os.path.expandvars(p)):
            return p

    raise Exception('Failed to resolve file path: %s' % path)


def get_skydir_distance_mask(src_skydir, skydir, dist, min_dist=None,
                             square=False, coordsys='CEL'):
    """Retrieve sources within a certain angular distance of an
    (ra,dec) coordinate.  This function supports two types of
    geometric selections: circular (square=False) and square
    (square=True).  The circular selection finds all sources with a given
    angular distance of the target position.  The square selection
    finds sources within an ROI-like region of size R x R where R
    = 2 x dist.

    Parameters
    ----------
    
    src_skydir : `~astropy.coordinates.SkyCoord` 
      Array of sky directions.

    skydir : `~astropy.coordinates.SkyCoord` 
      Sky direction with respect to which the selection will be applied.

    dist : float
      Maximum distance in degrees from the sky coordinate.

    square : bool
      Choose whether to apply a circular or square selection.

    coordsys : str
      Coordinate system to use when applying a selection with square=True.
    
    """

    if dist is None: dist = 180.

    if not square:
        dtheta = src_skydir.separation(skydir).rad
    elif coordsys == 'CEL':
        dtheta = get_linear_dist(skydir,
                                 src_skydir.ra.rad,
                                 src_skydir.dec.rad,
                                 coordsys=coordsys)
    elif coordsys == 'GAL':
        dtheta = get_linear_dist(skydir,
                                 src_skydir.galactic.l.rad,
                                 src_skydir.galactic.b.rad,
                                 coordsys=coordsys)
    else:
        raise Exception('Unrecognized coordinate system: %s' % coordsys)

    msk = (dtheta < np.radians(dist))
    if min_dist is not None: msk &= (dtheta > np.radians(min_dist))
    return msk


def get_linear_dist(skydir, lon, lat, coordsys='CEL'):
    xy = wcs_utils.sky_to_offset(skydir, np.degrees(lon), np.degrees(lat),
                             coordsys=coordsys)

    x = np.radians(xy[:, 0])
    y = np.radians(xy[:, 1])
    delta = np.array([np.abs(x), np.abs(y)])
    dtheta = np.max(delta, axis=0)
    return dtheta


def get_dist_to_edge(skydir, lon, lat, width, coordsys='CEL'):
    xy = wcs_utils.sky_to_offset(skydir, np.degrees(lon), np.degrees(lat),
                             coordsys=coordsys)

    x = np.radians(xy[:, 0])
    y = np.radians(xy[:, 1])

    delta_edge = np.array([np.abs(x) - width, np.abs(y) - width])
    dtheta = np.max(delta_edge, axis=0)
    return dtheta


def get_params_dict(pars_dict):

    params = {}
    for k, p in pars_dict.items():
        val = p['value']*p['scale']
        err = np.nan
        if 'error' in p:
            err = p['error']*np.abs(p['scale'])
        params[k]=np.array([val,err])

    return params

class Model(object):
    """Base class for source objects.  This class is a container for both
    spectral and spatial parameters as well as other source properties
    such as TS, Npred, and location within the ROI.
    """

    def __init__(self, name, data=None):

        self._data = defaults.make_default_dict(defaults.source_output)
        self._data.setdefault('spectral_pars', {})
        self._data.setdefault('spatial_pars', {})
        self._data.setdefault('catalog', {})
        self._data['assoc'] = {}
        self._data['name'] = name
        
        if data is not None:
            self._data.update(data)
            
        if not self.spectral_pars:
            pdict = gtutils.get_function_pars_dict(self['SpectrumType'])
            self._data['spectral_pars'] = pdict
            for k, v in self.spectral_pars.items():
                self._data['spectral_pars'][k] = gtutils.make_parameter_dict(v)
            
        self._names = [name]
        catalog = self._data['catalog']

        for k in ROIModel.src_name_cols:

            if k not in catalog: 
                continue
            name = catalog[k].strip()
            if name != '' and name not in self._names:
                self._names.append(name)

            self._data['assoc'][k] = name

        if self.params:
            self._sync_spectral_pars()
        else:
            self._sync_params()
            
    def __contains__(self, key):
        return key in self._data

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):

        data = copy.deepcopy(self.data)
        data['names'] = self.names

        output = []
        output += ['{:15s}:'.format('Name') + ' {name:s}']
        output += ['{:15s}:'.format('TS') + ' {ts:.2f}']
        output += ['{:15s}:'.format('Npred') + ' {npred:.2f}']
        output += ['{:15s}:'.format('SpatialModel') + ' {SpatialModel:s}']
        output += ['{:15s}:'.format('SpectrumType') + ' {SpectrumType:s}']
        output += ['Spectral Parameters']

        for k, v in self['params'].items():
            if isinstance(v, np.ndarray):
                output += [
                    '{:15s}: {:10.4g} +/- {:10.4g}'.format(k, v[0], v[1])]

        return '\n'.join(output).format(**data)

    def items(self):
        return self._data.items()

    @property
    def data(self):
        return self._data

    @property
    def params(self):
        return self._data['params']

    @property
    def spectral_pars(self):
        return self._data['spectral_pars']

    @property
    def spatial_pars(self):
        return self._data['spatial_pars']

    @property
    def name(self):
        return self._data['name']

    @property
    def names(self):
        return self._names

    @property
    def assoc(self):
        return self._data['assoc']

    @staticmethod
    def create_from_dict(src_dict, roi_skydir=None):
        
        src_dict.setdefault('SpatialModel','PointSource')
        src_dict.setdefault('SpatialType',
                            gtutils.get_spatial_type(src_dict['SpatialModel']))

        # Need this to handle old convention for
        # MapCubeFunction/ConstantValue sources
        if src_dict['SpatialModel'] == 'DiffuseSource':            
            src_dict['SpatialModel'] = src_dict['SpatialType']

        if 'spectral_pars' in src_dict:
            src_dict['spectral_pars'] = gtutils.cast_pars_dict(src_dict['spectral_pars'])

        if 'spatial_pars' in src_dict:
            src_dict['spatial_pars'] = gtutils.cast_pars_dict(src_dict['spatial_pars'])
        
        if src_dict['SpatialModel'] == 'ConstantValue':
            return IsoSource(src_dict['name'],src_dict)
        elif src_dict['SpatialModel'] == 'MapCubeFunction':
            return MapCubeSource(src_dict['name'],src_dict)
        else:
            return Source.create_from_dict(src_dict,roi_skydir)

    def _sync_spectral_pars(self):
        """Update spectral parameters dictionary."""
        
        sp = self['spectral_pars']
        for k, p in sp.items():
            sp[k]['value'] = self['params'][k][0]/sp[k]['scale']
            if np.isfinite(self['params'][k][1]):
                sp[k]['error'] = self['params'][k][1]/np.abs(sp[k]['scale'])
            sp[k] = gtutils.make_parameter_dict(sp[k])

    def _sync_params(self):        
        self._data['params'] = get_params_dict(self['spectral_pars'])

    def get_norm(self):

        par_name = gtutils.get_function_norm_par_name(self['SpectrumType'])
        val = self.spectral_pars[par_name]['value']
        scale = self.spectral_pars[par_name]['scale']
        return float(val)*float(scale)

    def get_catalog_dict(self):

        o = {'Spectral_Index' : np.nan,
             'Flux_Density' : np.nan,
             'Pivot_Energy' : np.nan,
             'beta' : np.nan,
             'Exp_Index' : np.nan,
             'Cutoff' : np.nan}
        
        if self['SpectrumType'] == 'PowerLaw':
            o['Spectral_Index'] = -1.0*self.params['Index'][0]
            o['Flux_Density'] = self.params['Prefactor'][0]
            o['Pivot_Energy'] = self.params['Scale'][0]
        elif self['SpectrumType'] == 'LogParabola':
            o['Spectral_Index'] = self.params['alpha'][0]
            o['Flux_Density'] = self.params['norm'][0]
            o['Pivot_Energy'] = self.params['Eb'][0]
            o['beta'] = self.params['beta'][0]
        elif self['SpectrumType'] == 'PLSuperExpCutoff':
            o['Spectral_Index'] = -self.params['Index1'][0]
            o['Exp_Index'] = self.params['Index2'][0]
            o['Flux_Density'] = self.params['Prefactor'][0]
            o['Pivot_Energy'] = self.params['Scale'][0]
            o['Cutoff'] = self.params['Cutoff'][0]
            
        return o
    
    def check_cuts(self, cuts):

        if cuts is None:
            return True

        if isinstance(cuts, tuple):
            cuts = {cuts[0]: (cuts[1], cuts[2])}
        elif isinstance(cuts, list):
            cuts = {c[0]: (c[1], c[2]) for c in cuts}

        for k, v in cuts.items():

            # if not isinstance(c,tuple) or len(c) != 3:
            #    raise Exception('Wrong format for cuts tuple.')

            if k in self._data:
                if not utils.apply_minmax_selection(self[k], v):
                    return False
            elif 'catalog' in self._data and k in self._data['catalog']:
                if not utils.apply_minmax_selection(self['catalog'][k], v):
                    return False
            else:
                return False

        return True

    def set_spectral_pars(self,spectral_pars):

        self._data['spectral_pars'] = copy.deepcopy(spectral_pars)
        self._sync_params()

    def update_spectral_pars(self,spectral_pars):

        self._data['spectral_pars'] = utils.merge_dict(self.spectral_pars,spectral_pars)
        self._sync_params()
        
    def set_name(self, name, names=None):
        self._data['name'] = name
        if names is None:
            self._names = [name]
        else:
            self._names = names

    def add_name(self, name):
        if name not in self._names:
            self._names.append(name)
            
    def update_data(self, d):
        self._data = utils.merge_dict(self._data, d, add_new_keys=True)
        #if self.params:
        #    self._sync_spectral_pars()

    def update_from_source(self, src):

        self._data['spectral_pars'] = {}
        self._data['spatial_pars'] = {}

        self._data = utils.merge_dict(self.data, src.data, add_new_keys=True)
        self._name = src.name
        self._names = list(set(self._names + src.names))


class IsoSource(Model):
    def __init__(self, name, data):

        data['SpectrumType'] = 'FileFunction'
        data['SpatialType'] = 'ConstantValue'
        data['SpatialModel'] = 'ConstantValue'
        data['SourceType'] = 'DiffuseSource'

        if not 'spectral_pars' in data:        
            data['spectral_pars'] = {
                'Normalization': {'name': 'Normalization', 'scale': 1.0,
                                  'value': 1.0,
                                  'min': 0.001, 'max': 1000.0,
                                  'free': False }}
        
        super(IsoSource, self).__init__(name, data)

        self._init_spatial_pars()
        
    @property
    def filefunction(self):
        return self._data['filefunction']

    @property
    def diffuse(self):
        return True

    def _init_spatial_pars(self):

        self['spatial_pars'] = {
            'Value': {'name': 'Value', 'scale': '1',
                      'value': '1', 'min': '0', 'max': '10',
                      'free': '0'}}
    
    def write_xml(self, root):

        source_element = utils.create_xml_element(root, 'source',
                                                  dict(name=self.name,
                                                       type='DiffuseSource'))

        filename = re.sub(r'\$([a-zA-Z\_]+)', r'$(\1)', self.filefunction)
        spec_el = utils.create_xml_element(source_element, 'spectrum',
                                           dict(file=filename,
                                                type='FileFunction',
                                                ctype='-1'))

        spat_el = utils.create_xml_element(source_element, 'spatialModel',
                                           dict(type='ConstantValue'))

        for k, v in self.spectral_pars.items():
            utils.create_xml_element(spec_el, 'parameter', v)

        for k, v in self.spatial_pars.items():
            utils.create_xml_element(spat_el, 'parameter', v)


class MapCubeSource(Model):
    def __init__(self, name, data):

        data['SpectrumType'] = 'PowerLaw'
        data['SpatialType'] = 'MapCubeFunction'
        data['SpatialModel'] = 'MapCubeFunction'
        data['SourceType'] = 'DiffuseSource'

        if not 'spectral_pars' in data:
            data['spectral_pars'] = {
                'Prefactor': {'name': 'Prefactor', 'scale': 1.0,
                              'value': 1.0, 'min': 0.1, 'max': '10.0',
                              'free': False},
                'Index': {'name': 'Index', 'scale': -1.0,
                          'value': 0.0, 'min': -1.0, 'max': 1.0,
                          'free': False},
                'Scale': {'name': 'Scale', 'scale': 1.0,
                          'value': 1000.0,
                          'min': 1000.0, 'max': 1000.0,
                          'free': False},
            }
        
        super(MapCubeSource, self).__init__(name, data)

        self._init_spatial_pars()

    @property
    def mapcube(self):
        return self._data['mapcube']

    @property
    def diffuse(self):
        return True

    def _init_spatial_pars(self):

        self['spatial_pars'] = {
            'Normalization':
                {'name': 'Normalization', 'scale': '1',
                 'value': '1', 'min': '0', 'max': '10',
                 'free': '0'}}
    
    def write_xml(self, root):

        source_element = utils.create_xml_element(root, 'source',
                                                  dict(name=self.name,
                                                       type='DiffuseSource'))

        spec_el = utils.create_xml_element(source_element, 'spectrum',
                                           dict(type='PowerLaw'))

        filename = re.sub(r'\$([a-zA-Z\_]+)', r'$(\1)', self.mapcube)
        spat_el = utils.create_xml_element(source_element, 'spatialModel',
                                           dict(type='MapCubeFunction',
                                                file=filename))

        for k, v in self.spectral_pars.items():
            utils.create_xml_element(spec_el, 'parameter', v)

        for k, v in self.spatial_pars.items():
            utils.create_xml_element(spat_el, 'parameter', v)


class Source(Model):
    """Class representation of a source (non-diffuse) model component.
    A source object serves as a container for the properties of that
    source (position, spatial/spectral parameters, TS, etc.) as
    derived in the current analysis.  Most properties of a source
    object can be accessed with the bracket operator:

    # Return the TS of this source
    >>> print src['ts']

    # Get a skycoord representation of the source position
    >>> print src.skydir
    """

    def __init__(self, name, data, radec=None):

        data.setdefault('SpatialModel','PointSource')
        data.setdefault('SpectrumType','PowerLaw')
        data.setdefault('SpatialType',gtutils.get_spatial_type(data['SpatialModel']))
        data.setdefault('SourceType',gtutils.get_source_type(data['SpatialType']))
        
        super(Source, self).__init__(name, data)

        catalog = self.data.get('catalog', {})

        if radec is not None:
            self._set_radec(radec)
        elif 'ra' in self.data and 'dec' in self.data:
            self._set_radec([self.data['ra'], self.data['dec']])
        elif 'RAJ2000' in catalog and 'DEJ2000' in catalog:
            self._set_radec([catalog['RAJ2000'], catalog['DEJ2000']])
        else:
            raise Exception('Failed to infer RADEC for source: %s' % name)

        self._init_spatial_pars()
        
    def __str__(self):

        data = copy.deepcopy(self.data)
        data['names'] = self.names

        try:
            data['flux'], data['flux_err'] = data['flux'][0], data['flux'][1]
        except:
            data['flux'], data['flux_err'] = 0., 0.

        try:
            data['eflux'], data['eflux_err'] = data['eflux'][0], data['eflux'][1]
        except:           
            data['eflux'], data['eflux_err'] = 0., 0.
        
        output = []
        output += ['{:15s}:'.format('Name') + ' {name:s}']
        output += ['{:15s}:'.format('Associations') + ' {names:s}']
        output += ['{:15s}:'.format('RA/DEC') + ' {ra:10.3f}/{dec:10.3f}']
        output += ['{:15s}:'.format('GLON/GLAT') + ' {glon:10.3f}/{glat:10.3f}']
        output += ['{:15s}:'.format('TS') + ' {ts:.2f}']
        output += ['{:15s}:'.format('Npred') + ' {npred:.2f}']
        output += ['{:15s}:'.format('Flux') + ' {flux:9.4g} +/- {flux_err:8.3g}']
        output += ['{:15s}:'.format('EnergyFlux') + ' {eflux:9.4g} +/- {eflux_err:8.3g}']
        output += ['{:15s}:'.format('SpatialModel') + ' {SpatialModel:s}']
        output += ['{:15s}:'.format('SpectrumType') + ' {SpectrumType:s}']
        output += ['Spectral Parameters']

        for k, v in self['params'].items():
            if isinstance(v, np.ndarray):
                output += [
                    '{:15s}: {:10.4g} +/- {:10.4g}'.format(k, v[0], v[1])]

        return '\n'.join(output).format(**data)

    def _set_radec(self,radec):

        self['radec'] = np.array(radec,ndmin=1)
        self['RAJ2000'] = radec[0]
        self['DEJ2000'] = radec[1]
        self['ra'] = radec[0]
        self['dec'] = radec[1]        
        glonlat = utils.eq2gal(radec[0], radec[1])
        self['glon'], self['glat'] = glonlat[0][0], glonlat[1][0]
        if 'RA' in self.spatial_pars:
            self.spatial_pars['RA']['value'] = radec[0]
            self.spatial_pars['DEC']['value'] = radec[1]

    def _set_spatial_width(self):

         if self['SpatialModel'] in ['GaussianSource','RadialGaussian']:

            if self['SpatialWidth'] is None:
                self.data.setdefault('Sigma',0.5)                
                self['SpatialWidth'] = self['Sigma']*1.5095921854516636
            else:
                self.data.setdefault('Sigma',self['SpatialWidth']/1.5095921854516636)   

         elif self['SpatialModel'] in ['DiskSource','RadialDisk']:

            if self['SpatialWidth'] is None:
                self.data.setdefault('Radius',0.5)                
                self['SpatialWidth'] = self['Radius']*0.8246211251235321
            else:
                self.data.setdefault('Radius',self['SpatialWidth']/0.8246211251235321)   
            
    def _init_spatial_pars(self):

        if self['SpatialType'] == 'SkyDirFunction':
            self._extended = False
            self._data['SourceType'] = 'PointSource'
        else:
            self._extended = True
            self._data['SourceType'] = 'DiffuseSource'
        
        self._set_spatial_width()
        
        if self['SpatialType'] == 'SpatialMap':
            self._data['spatial_pars'] = {
                'Prefactor': {'name': 'Prefactor', 'value': 1.0,
                              'free': False, 'min': 0.001, 'max': 1000.0,
                              'scale': 1.0}
                }
        else:
            self.spatial_pars.setdefault('RA',
                                         {'name': 'RA', 'value': self['ra'],
                                          'free': False,
                                          'min': -360.0, 'max': 360.0, 'scale': 1.0})
            self.spatial_pars.setdefault('DEC',
                                         {'name': 'DEC', 'value': self['dec'],
                                          'free': False,
                                          'min': -90.0, 'max': 90.0, 'scale': 1.0})
            
        if self['SpatialType'] == 'RadialGaussian':
            self.spatial_pars.setdefault('Sigma',
                                         {'name': 'Sigma', 'value': self['Sigma'],
                                          'free': False, 'min': 0.001, 'max': 10,
                                          'scale': '1.0'})
        elif self['SpatialType'] == 'RadialDisk':            
            self.spatial_pars.setdefault('Radius',
                                         {'name': 'Radius', 'value': self['Radius'],
                                          'free': False, 'min': 0.001, 'max': 10,
                                          'scale': 1.0})
       

    def load_from_catalog(self):
        """Load spectral parameters from catalog values."""
        
        self._data['spectral_pars'] = \
            gtutils.get_function_pars_dict(self['SpectrumType'])
        sp = self['spectral_pars']

        catalog = self.data.get('catalog', {})

        if self['SpectrumType'] == 'PowerLaw':

            sp['Prefactor']['value'] = catalog['Flux_Density']
            sp['Prefactor']['scale'] = None
            sp['Scale']['value'] = catalog['Pivot_Energy']
            sp['Scale']['scale'] = 1.0
            sp['Index']['value'] = catalog['Spectral_Index']
            sp['Index']['max'] = max(5.0, sp['Index']['value'] + 1.0)
            sp['Index']['min'] = min(0.0, sp['Index']['value'] - 1.0)
            sp['Index']['scale'] = -1.0
            
            sp['Prefactor'] = gtutils.make_parameter_dict(sp['Prefactor'])
            sp['Scale'] = gtutils.make_parameter_dict(sp['Scale'], True)
            sp['Index'] = gtutils.make_parameter_dict(sp['Index'])

        elif self['SpectrumType'] == 'LogParabola':

            sp['norm']['value'] = catalog['Flux_Density']
            sp['norm']['scale'] = None
            sp['Eb']['value'] = catalog['Pivot_Energy']
            sp['alpha']['value'] = catalog['Spectral_Index']
            sp['beta']['value'] = catalog['beta']

            sp['norm'] = gtutils.make_parameter_dict(sp['norm'])
            sp['Eb'] = gtutils.make_parameter_dict(sp['Eb'], True)
            sp['alpha'] = gtutils.make_parameter_dict(sp['alpha'])
            sp['beta'] = gtutils.make_parameter_dict(sp['beta'])

        elif self['SpectrumType'] == 'PLSuperExpCutoff':

            flux_density = catalog['Flux_Density']
            flux_density *= np.exp(
                (catalog['Pivot_Energy'] / catalog['Cutoff']) ** catalog[
                    'Exp_Index'])

            sp['Prefactor']['value'] = flux_density
            sp['Prefactor']['scale'] = None
            sp['Index1']['value'] = catalog['Spectral_Index']
            sp['Index1']['scale'] = -1.0
            sp['Index2']['value'] = catalog['Exp_Index']
            sp['Index2']['scale'] = 1.0
            sp['Scale']['value'] = catalog['Pivot_Energy']
            sp['Cutoff']['value'] = catalog['Cutoff']

            sp['Prefactor'] = gtutils.make_parameter_dict(sp['Prefactor'])
            sp['Scale'] = gtutils.make_parameter_dict(sp['Scale'], True)
            sp['Index1'] = gtutils.make_parameter_dict(sp['Index1'])
            sp['Index2'] = gtutils.make_parameter_dict(sp['Index2'])
            sp['Cutoff'] = gtutils.make_parameter_dict(sp['Cutoff'])

        else:
            raise Exception('Unsupported spectral type:' + self['SpectrumType'])

    def update_data(self, d):
        self._data = utils.merge_dict(self._data, d, add_new_keys=True)
        if 'ra' in d and 'dec' in d:
            self._set_radec([d['ra'],d['dec']])
        #if self.params:
        #    self._sync_spectral_pars()
                          
    def set_position(self, skydir):
        """
        Set the position of the source.
        
        Parameters
        ----------
        skydir : `~astropy.coordinates.SkyCoord` 

        """

        if not isinstance(skydir, SkyCoord):
            skydir = SkyCoord(ra=skydir[0], dec=skydir[1], unit=u.deg)

        if not skydir.isscalar:
            skydir = np.ravel(skydir)[0]

        radec = np.array([skydir.icrs.ra.deg, skydir.icrs.dec.deg])
        self._set_radec(radec)
                
    def set_roi_direction(self,roidir):

        offset = roidir.separation(self.skydir).deg
        offset_cel = wcs_utils.sky_to_offset(roidir,self['ra'], self['dec'], 'CEL')
        offset_gal = wcs_utils.sky_to_offset(roidir,self['glon'], self['glat'], 'GAL')

        self['offset'] = offset
        self['offset_ra'] = offset_cel[0, 0]
        self['offset_dec'] = offset_cel[0, 1]
        self['offset_glon'] = offset_gal[0, 0]
        self['offset_glat'] = offset_gal[0, 1]

    def set_spatial_model(self, spatial_model, spatial_width=None):

        self._data['SpatialModel'] = spatial_model
        self._data['SpatialWidth'] = spatial_width
        self._data['SpatialType'] = gtutils.get_spatial_type(self['SpatialModel'])
        self._data['spatial_pars'] = {}
        self._init_spatial_pars()

    def separation(self, src):

        if isinstance(src, Source):
            return self.radec.separation(src.skydir)
        else:
            return self.radec.separation(src)

    @property
    def diffuse(self):
        return False

    @property
    def extended(self):
        return self._extended

    @property
    def associations(self):
        return self._names

    @property
    def radec(self):
        return self['radec']

    @property
    def skydir(self):
        """Return a SkyCoord representation of the source position.

        Returns
        -------
        skydir : `~astropy.coordinates.SkyCoord` 
        """
        return SkyCoord(self.radec[0] * u.deg, self.radec[1] * u.deg)

    @property
    def data(self):
        return self._data

    @staticmethod
    def create_from_dict(src_dict, roi_skydir=None):
        """Create a source object from a python dictionary.

        Parameters
        ----------
        src_dict : dict
           Dictionary defining the properties of the source.

        """

        src_dict = copy.deepcopy(src_dict)
        src_dict.setdefault('SpatialModel','PointSource')
        spectrum_type = src_dict.setdefault('SpectrumType','PowerLaw')
        spatial_type = \
            src_dict.setdefault('SpatialType',
                                gtutils.get_spatial_type(src_dict['SpatialModel']))
        
        spectral_pars = \
            src_dict.setdefault('spectral_pars',
                                gtutils.get_function_pars_dict(spectrum_type))

        spatial_pars = \
            src_dict.setdefault('spatial_pars',
                                gtutils.get_function_pars_dict(src_dict['SpatialType']))

        for k in ['RA','DEC','Prefactor']:
            if k in spatial_pars:
                del spatial_pars[k]
            
        for k, v in spectral_pars.items():

            if k not in src_dict: 
                continue

            if not isinstance(src_dict[k], dict):
                spectral_pars[k].update({'name': k,
                                         'value': src_dict.pop(k)})
            else:
                spectral_pars[k].update(src_dict.pop(k))


        for k, v in spatial_pars.items():

            if k not in src_dict: 
                continue

            if not isinstance(src_dict[k], dict):
                spatial_pars[k].update({'name': k, 'value': src_dict[k]})
            else:
                spatial_pars[k].update(src_dict.pop(k))
                
                
        for k, v in spectral_pars.items():
            spectral_pars[k] = gtutils.make_parameter_dict(spectral_pars[k])

        for k, v in spatial_pars.items():
            spatial_pars[k] = gtutils.make_parameter_dict(spatial_pars[k],
                                                          rescale=False)
            
        src_dict['spectral_pars'] = gtutils.cast_pars_dict(spectral_pars)
        src_dict['spatial_pars'] = gtutils.cast_pars_dict(spatial_pars)
        #        validate_config(src_dict,default_src_dict)
            
        if 'name' in src_dict:
            name = src_dict['name']
            src_dict['Source_Name'] = src_dict.pop('name')
        elif 'Source_Name' in src_dict:
            name = src_dict['Source_Name']
        else:
            raise Exception('Source name undefined.')

        skydir = wcs_utils.get_target_skydir(src_dict, roi_skydir)
        
        src_dict['RAJ2000'] = skydir.ra.deg
        src_dict['DEJ2000'] = skydir.dec.deg

        radec = np.array([skydir.ra.deg, skydir.dec.deg])
        
        return Source(name, src_dict, radec=radec)

    @staticmethod
    def create_from_xml(root, extdir=None):
        """Create a Source object from an XML node."""

        spec = utils.load_xml_elements(root, 'spectrum')
        spat = utils.load_xml_elements(root, 'spatialModel')
        spectral_pars = utils.load_xml_elements(root, 'spectrum/parameter')
        spatial_pars = utils.load_xml_elements(root, 'spatialModel/parameter')

        spectral_pars = gtutils.cast_pars_dict(spectral_pars)
        spatial_pars = gtutils.cast_pars_dict(spatial_pars)
        
        src_type = root.attrib['type']
        spatial_type = spat['type']
        spectral_type = spec['type']

        xml_dict = copy.deepcopy(root.attrib)
        src_dict = {'catalog': xml_dict}

        src_dict['Source_Name'] = xml_dict['name']
        src_dict['SpectrumType'] = spec['type']
        src_dict['SpatialType'] = spatial_type
        src_dict['SourceType'] = src_type

        if src_type == 'PointSource':
            src_dict['SpatialModel'] = 'PointSource'
        elif spatial_type == 'SpatialMap':
            src_dict['SpatialModel'] = 'SpatialMap'

        if src_type == 'PointSource' or spatial_type == 'SpatialMap':

            if 'file' in spat:
                src_dict['Spatial_Filename'] = spat['file']
                if not os.path.isfile(src_dict['Spatial_Filename']) \
                        and extdir is not None:
                    src_dict['Spatial_Filename'] = \
                        os.path.join(extdir, 'Templates',
                                     src_dict['Spatial_Filename'])

            if 'RA' in src_dict:
                src_dict['RAJ2000'] = float(xml_dict['RA'])
                src_dict['DEJ2000'] = float(xml_dict['DEC'])
            elif 'RA' in spatial_pars:
                src_dict['RAJ2000'] = float(spatial_pars['RA']['value'])
                src_dict['DEJ2000'] = float(spatial_pars['DEC']['value'])
            else:
                hdu = pyfits.open(
                    os.path.expandvars(src_dict['Spatial_Filename']))
                src_dict['RAJ2000'] = float(hdu[0].header['CRVAL1'])
                src_dict['DEJ2000'] = float(hdu[0].header['CRVAL2'])

            radec = np.array([src_dict['RAJ2000'], src_dict['DEJ2000']])

            src_dict['spectral_pars'] = spectral_pars
            src_dict['spatial_pars'] = spatial_pars            
            return Source(src_dict['Source_Name'],
                          src_dict, radec=radec)

        elif src_type == 'DiffuseSource' and spatial_type == 'ConstantValue':
            return IsoSource(src_dict['Source_Name'],
                             {'filefunction' : spec['file'],
                              'spectral_pars' : spectral_pars,
                              'spatial_pars' : spatial_pars})
        elif src_type == 'DiffuseSource' and spatial_type == 'MapCubeFunction':
            return MapCubeSource(src_dict['Source_Name'],
                                 {'mapcube' : spat['file'],
                                  'spectral_pars' : spectral_pars,
                                  'spatial_pars' : spatial_pars})
        else:
            raise Exception(
                'Unrecognized type for source: %s' % src_dict['Source_Name'])

    def write_xml(self, root):
        """Write this source to an XML node."""

        if not self.extended:
            source_element = utils.create_xml_element(root, 'source',
                                                      dict(name=self[
                                                          'Source_Name'],
                                                           type='PointSource'))

            spat_el = ElementTree.SubElement(source_element, 'spatialModel')
            spat_el.set('type', 'SkyDirFunction')

        elif self['SpatialType'] == 'SpatialMap':
            source_element = utils.create_xml_element(root, 'source',
                                                      dict(name=self[
                                                          'Source_Name'],
                                                           type='DiffuseSource'))

            filename = self['Spatial_Filename']
            filename = re.sub(r'\$([a-zA-Z\_]+)', r'$(\1)', filename)

            spat_el = utils.create_xml_element(source_element, 'spatialModel',
                                               dict(map_based_integral='True',
                                                    type='SpatialMap',
                                                    file=filename))
        else:
            source_element = utils.create_xml_element(root, 'source',
                                                      dict(name=self['Source_Name'],
                                                           type='DiffuseSource'))
            spat_el = utils.create_xml_element(source_element, 'spatialModel',
                                               dict(type=self['SpatialType']))

            
        for k, v in self.spatial_pars.items():
            utils.create_xml_element(spat_el, 'parameter', v)

        el = ElementTree.SubElement(source_element, 'spectrum')

        stype = self['SpectrumType'].strip()
        el.set('type', stype)

        for k, v in self.spectral_pars.items():
            utils.create_xml_element(el, 'parameter', v)


class ROIModel(fermipy.config.Configurable):
    """This class is responsible for managing the ROI model (both sources
    and diffuse components).  Source catalogs can be read
    from either FITS or XML files.  Individual components are
    represented by instances of `~fermipy.roi_model.Model` and can be
    accessed by name using the bracket operator.

        * Create an ROI with all 3FGL sources and print a summary of its contents:

        >>> skydir = astropy.coordinates.SkyCoord(0.0,0.0,unit='deg')
        >>> roi = ROIModel({'catalogs' : ['3FGL'],'src_roiwidth' : 10.0},skydir=skydir)
        >>> print roi
        name                SpatialModel   SpectrumType     offset        ts       npred
        --------------------------------------------------------------------------------
        3FGL J2357.3-0150   PointSource    PowerLaw          1.956       nan         0.0
        3FGL J0006.2+0135   PointSource    PowerLaw          2.232       nan         0.0
        3FGL J0016.3-0013   PointSource    PowerLaw          4.084       nan         0.0
        3FGL J0014.3-0455   PointSource    PowerLaw          6.085       nan         0.0

        * Print a summary of an individual source

        >>> print roi['3FGL J0006.2+0135']

        * Get the SkyCoord for a source

        >>> dir = roi['SourceA'].skydir

        * Loop over all sources and print their names

        >>> for s in roi.sources: print s.name

    """

    defaults = dict(defaults.model.items(),
                    logfile=(None, '', str),
                    fileio=defaults.fileio,
                    logging=defaults.logging)

    src_name_cols = ['Source_Name',
                     'ASSOC', 'ASSOC1', 'ASSOC2', 'ASSOC_GAM',
                     '1FHL_Name', '2FGL_Name', '3FGL_Name',
                     'ASSOC_GAM1', 'ASSOC_GAM2', 'ASSOC_TEV']

    def __init__(self, config=None, **kwargs):
        # Coordinate for ROI center (defaults to 0,0)
        self._skydir = kwargs.pop('skydir', SkyCoord(0.0, 0.0, unit=u.deg))
        coordsys = kwargs.pop('coordsys', 'CEL')
        super(ROIModel, self).__init__(config, **kwargs)

        self.logger = Logger.get(self.__class__.__name__,
                                 self.config['logfile'],
                                 ll(self.config['logging']['verbosity']))

        if self.config['extdir'] is not None and \
                not os.path.isdir(os.path.expandvars(self.config['extdir'])):
            self._config['extdir'] = \
                os.path.join('$FERMIPY_DATA_DIR',
                             'catalogs', self.config['extdir'])

        self._src_radius = self.config['src_radius']
        if self.config['src_roiwidth'] is not None:
            self._config['src_radius_roi'] = self.config['src_roiwidth'] * 0.5

        self._srcs = []
        self._diffuse_srcs = []
        self._src_dict = collections.defaultdict(set)
        self._src_radius = []

        self.load(coordsys=coordsys)

    def __contains__(self, key):
        key = key.replace(' ', '').lower()        
        return key in self._src_dict.keys()
        
    def __getitem__(self, key):
        return self.get_source_by_name(key)

    def __iter__(self):
        return iter(self._srcs + self._diffuse_srcs)

    def __str__(self):

        o = ''
        o += '%-20s%-15s%-15s%8s%10s%12s\n' % (
        'name', 'SpatialModel', 'SpectrumType', 'offset',
        'ts', 'npred')
        o += '-' * 80 + '\n'

        for s in sorted(self.sources, key=lambda t: t['offset']):

            if s.diffuse: continue
            o += '%-20.19s%-15.14s%-15.14s%8.3f%10.2f%12.1f\n' % (
            s['name'], s['SpatialModel'],
            s['SpectrumType'],
            s['offset'], s['ts'], s['npred'])

        for s in sorted(self.sources, key=lambda t: t['offset']):

            if not s.diffuse: continue
            o += '%-20.19s%-15.14s%-15.14s%8s%10.2f%12.1f\n' % (
            s['name'], s['SpatialModel'],
            s['SpectrumType'],
            '-----', s['ts'], s['npred'])

        return o

    @property
    def skydir(self):
        """Return the sky direction objection corresponding to the
        center of the ROI."""
        return self._skydir

    @property
    def sources(self):
        return self._srcs + self._diffuse_srcs

    @property
    def point_sources(self):
        return self._srcs

    @property
    def diffuse_sources(self):
        return self._diffuse_srcs

    def clear(self):
        """Clear the contents of the ROI."""
        self._srcs = []
        self._diffuse_srcs = []
        self._src_dict = collections.defaultdict(set)
        self._src_radius = []
    
    def load_diffuse_srcs(self):

        self._load_diffuse_src('isodiff')
        self._load_diffuse_src('galdiff')
        self._load_diffuse_src('limbdiff')
        self._load_diffuse_src('diffuse')

    def _load_diffuse_src(self, name, src_type='FileFunction'):

        if 'FERMI_DIR' in os.environ and 'FERMI_DIFFUSE_DIR' not in os.environ:
            os.environ['FERMI_DIFFUSE_DIR'] = \
                os.path.expandvars('$FERMI_DIR/refdata/fermi/galdiffuse')
        if 'FERMIPY_WORKDIR' not in os.environ:

            if self.config['fileio']['workdir'] is not None:
                os.environ['FERMIPY_WORKDIR'] = self.config['fileio']['workdir']
            else:
                os.environ['FERMIPY_WORKDIR'] = os.getcwd()

        srcs = []
        if self.config[name] is not None:
            srcs = self.config[name]

        for i, t in enumerate(srcs):

            if isinstance(t, str):
                src_dict = {'file': t}
            elif isinstance(t, dict):
                src_dict = copy.deepcopy(t)

            src_dict['file'] = \
                resolve_file_path(src_dict['file'],
                                  search_dirs=['$FERMIPY_WORKDIR',
                                               os.path.join('$FERMIPY_ROOT',
                                                            'data'),
                                               '$FERMI_DIFFUSE_DIR'])

            if 'name' not in src_dict:
                if len(srcs) == 1:
                    src_dict['name'] = name
                else:
                    src_dict['name'] = name + '%02i' % i

            if re.search(r'(\.txt$)', src_dict['file']):
                src_type = 'FileFunction'
            elif re.search(r'(\.fits$|\.fit$|\.fits.gz$|\.fit.gz$)',
                           src_dict['file']):
                src_type = 'MapCubeFunction'
            else:
                raise Exception(
                    'Unrecognized file format for diffuse model: %s' % src_dict[
                        'file'])

            # Extract here


            if src_type == 'FileFunction':
                src = IsoSource(src_dict['name'], {'filefunction' : src_dict['file']})
                altname = os.path.basename(src_dict['file'])
                altname = re.sub(r'(\.txt$)', '', altname)
            else:
                src = MapCubeSource(src_dict['name'], {'mapcube' : src_dict['file']})
                altname = os.path.basename(src_dict['file'])
                altname = re.sub(r'(\.fits$|\.fit$|\.fits.gz$|\.fit.gz$)',
                                 '', altname)

            src.add_name(altname)
            self.load_source(src, False, self.config['merge_sources'])

    def create_source(self, name, src_dict, build_index=True,
                      merge_sources=True):
        """Add a new source to the ROI model from a dictionary or an
        existing source object.

        Parameters
        ----------

        name : str

        src_dict : dict or `~fermipy.roi_model.Source`

        Returns
        -------

        src : `~fermipy.roi_model.Source`
        """

        src_dict = copy.deepcopy(src_dict)

        if isinstance(src_dict,dict):
            src_dict['name'] = name
            src = Model.create_from_dict(src_dict,self.skydir)
        else:
            src = src_dict

        if isinstance(src,Source):
            src.set_roi_direction(self.skydir)

        self.logger.debug('Creating source ' + src.name)
        self.load_source(src, build_index=build_index,
                         merge_sources=merge_sources)
        
        return self.get_source_by_name(name)

    def copy_source(self, name):
        src = self.get_source_by_name(name)
        return copy.deepcopy(src)
    
    def load_sources(self, sources):
        """Delete all sources in the ROI and load the input source list."""

        self.logger.debug('Loading sources')
        
        self.clear()
        for s in sources:

            if isinstance(s, dict):
                s = Model.create_from_dict(s)
            
            self.load_source(s, build_index=False)
        self._build_src_index()

        self.logger.debug('Finished')

    def load_source(self, src, build_index=True, merge_sources=True,
                    **kwargs):
        """
        Load a single source.

        Parameters
        ----------

        src : `~fermipy.roi_model.Source`
           Source object that will be added to the ROI.

        merge_sources : bool        
           When a source matches an existing source in the model
           update that source with the properties of the new source.

        build_index : bool 
           Re-make the source index after loading this source.
        
        """
        src = copy.deepcopy(src)
        name = src.name.replace(' ', '').lower()

        min_sep = kwargs.get('min_separation',None)

        if min_sep is not None:
        
            sep = src.skydir.separation(self._src_skydir).deg            
            if len(sep) > 0 and np.min(sep) < min_sep:
                return        
        
        match_srcs = self.match_source(src)
        
        if len(match_srcs) == 1:

            self.logger.debug('Found matching source for %s : %s'
                              %( src.name, match_srcs[0].name ) )
            
            if merge_sources:
                self.logger.debug('Updating source model for %s' % src.name)
                match_srcs[0].update_from_source(src)
            else:
                match_srcs[0].add_name(src.name)
                self.logger.debug('Skipping source model for %s' % src.name)

            self._src_dict[src.name.replace(' ', '').lower()].add(match_srcs[0])
                
            return
        elif len(match_srcs) > 2:
            self.logger.warning('Multiple sources matching %s' % name)
            return
            
        self._src_dict[src.name].add(src)

        for name in src.names:
            self._src_dict[name.replace(' ', '').lower()].add(src)

        if isinstance(src, Source):
            self._srcs.append(src)
        else:
            self._diffuse_srcs.append(src)

        if build_index:
            self._build_src_index()

    def match_source(self,src):
        """Look for source or sources in the model that match the
        given source.  Sources are matched by name and any association
        columns defined in the assoc_xmatch_columns parameter.
        """
        
        srcs = set()

        names = [src.name]
        for col in self.config['assoc_xmatch_columns']:
            if col in src.assoc and src.assoc[col]:
                names += [src.assoc[col]]

        for name in names:
            name = name.replace(' ', '').lower()
            if name in self._src_dict and self._src_dict[name]:
                srcs.update(self._src_dict[name])
                
        return list(srcs)
        
    def load(self, **kwargs):
        """Load both point source and diffuse components."""

        self.logger.debug('Starting')
        
        coordsys = kwargs.get('coordsys', 'CEL')
        extdir = kwargs.get('extdir', self.config['extdir'])

        self.clear()
        self.load_diffuse_srcs()

        for c in self.config['catalogs']:

            extname = os.path.splitext(c)[1]
            if extname != '.xml':
                self.load_fits_catalog(c, extdir=extdir, coordsys=coordsys)
            elif extname == '.xml':
                self.load_xml(c, extdir=extdir, coordsys=coordsys)
            else:
                raise Exception('Unrecognized catalog file extension: %s' % c)

        for c in self.config['sources']:

            if 'name' not in c:
                raise Exception('No name field in source dictionary:\n ' + str(c))

            self.create_source(c['name'],c, build_index=False)

        self._build_src_index()        

    def delete_sources(self, srcs):

        for k, v in self._src_dict.items():
            for s in srcs:
                if s in v:
                    self._src_dict[k].remove(s)
            if not v:
                del self._src_dict[k]

        self._srcs = [s for s in self._srcs if s not in srcs]
        self._diffuse_srcs = [s for s in self._diffuse_srcs if s not in srcs]
        self._build_src_index()

    @staticmethod
    def create_from_roi_data(datafile):
        """Create an ROI model."""
        data = np.load(datafile).flat[0]

        roi = ROIModel()
        roi.load_sources(data['sources'].values())

        return roi
        
    @staticmethod
    def create(selection, config, **kwargs):
        """Create an ROIModel instance."""
        
        if selection['target'] is not None:
            return ROIModel.create_from_source(selection['target'],
                                               config, **kwargs)
        else:
            target_skydir = wcs_utils.get_target_skydir(selection)
            return ROIModel.create_from_position(target_skydir,
                                                 config, **kwargs)


    @staticmethod
    def create_from_position(skydir, config, **kwargs):
        """Create an ROIModel instance centered on a sky direction.

        Parameters
        ----------
        
        skydir : `~astropy.coordinates.SkyCoord` 
            Sky direction on which the ROI will be centered.

        config : dict
            Model configuration dictionary.
        """

        coordsys = kwargs.pop('coordsys', 'CEL')
        roi = ROIModel(config, skydir=skydir, coordsys=coordsys, **kwargs)
        return roi

        srcs_dict = {}

        if roi.config['src_radius'] is not None:
            rsrc, srcs = roi.get_sources_by_position(skydir,
                                                     roi.config['src_radius'])
            for s, r in zip(srcs, rsrc):
                srcs_dict[s.name] = (s, r)

        if roi.config['src_roiwidth'] is not None:
            rsrc, srcs = \
                roi.get_sources_by_position(skydir,
                                            roi.config['src_roiwidth'] / 2.,
                                            square=True, coordsys=coordsys)

            for s, r in zip(srcs, rsrc):
                srcs_dict[s.name] = (s, r)

        srcs = []
        rsrc = []

        for k, v in srcs_dict.items():
            srcs.append(v[0])
            rsrc.append(v[1])

        return ROIModel(config, srcs=srcs,
                        diffuse_srcs=roi._diffuse_srcs,
                        skydir=skydir, **kwargs)

    @staticmethod
    def create_from_source(name, config, **kwargs):
        """Create an ROI centered on the given source."""

        coordsys = kwargs.pop('coordsys', 'CEL')

        roi = ROIModel(config, src_radius=None, src_roiwidth=None, **kwargs)
        src = roi.get_source_by_name(name)

        return ROIModel.create_from_position(src.skydir, config,
                                             coordsys=coordsys, **kwargs)

    @staticmethod
    def create_roi_from_ft1(ft1file, config):
        """Create an ROI model by extracting the sources coordinates
        form an FT1 file."""
        pass

    def has_source(self, name):

        index_name = name.replace(' ', '').lower()
        if index_name in self._src_dict:
            return True
        else:
            return False

    def get_source_by_name(self, name):
        """Return a single source in the ROI with the given name.  The
        input name string can match any of the strings in the names
        property of the source object.  Case and whitespace are
        ignored when matching name strings.  If no sources are found
        or multiple sources then an exception is thrown.

        Parameters
        ----------
        name : str 
           Name string.
        
        Returns
        -------
        srcs : `~fermipy.roi_model.Model`
           A source object.
        
        """
        srcs = self.get_sources_by_name(name)

        if len(srcs) == 1:
            return srcs[0]        
        elif len(srcs) == 0:
            raise Exception('No source matching name: ' + name)
        elif len(srcs) > 1:
            raise Exception('Multiple sources matching name: ' + name)
        
    def get_sources_by_name(self, name):
        """Return a list of sources in the ROI matching the given
        name.  The input name string can match any of the strings in
        the names property of the source object.  Case and whitespace
        are ignored when matching name strings.

        Parameters
        ----------
        name : str 

        Returns
        -------
        srcs : list
           A list of `~fermipy.roi_model.Model` objects.        
        """

        index_name = name.replace(' ', '').lower()

        if index_name in self._src_dict:

            return list(self._src_dict[index_name])

            if len(srcs) == 1 and unique:
                return srcs[0]
            elif not unique:
                return srcs
            else:
                raise Exception('Multiple sources matching name: ' + name)
        else:
            raise Exception('No source matching name: ' + name)

    def get_nearby_sources(self, name, dist, min_dist=None,
                           square=False):

        src = self.get_source_by_name(name)
        return self.get_sources_by_position(src.skydir,
                                            dist, min_dist,
                                            square)

    def get_sources(self, cuts=None, distance=None,
                    minmax_ts=None, minmax_npred=None, square=False,
                    exclude_diffuse=False,
                    coordsys='CEL'):
        """Retrieve list of sources satisfying the given selections.

        Returns
        -------

        srcs : list
            List of source objects.
        """
        rsrc, srcs = self.get_sources_by_position(self.skydir,
                                                  distance,
                                                  square=square,
                                                  coordsys=coordsys)
        o = []
        for s, r in zip(srcs, rsrc):
            if not s.check_cuts(cuts):
                continue
            ts = s['ts']
            npred = s['npred']

            if not utils.apply_minmax_selection(ts, minmax_ts):
                continue
            if not utils.apply_minmax_selection(npred, minmax_npred):
                continue

            o.append(s)

        for s in self.diffuse_sources:

            if exclude_diffuse:
                continue            
            if not s.check_cuts(cuts):
                continue

            ts = s['ts']
            npred = s['npred']

            if not utils.apply_minmax_selection(ts, minmax_ts):
                continue
            if not utils.apply_minmax_selection(npred, minmax_npred):
                continue

            o.append(s)

        return o

    def get_sources_by_property(self, pname, pmin, pmax=None):

        srcs = []
        for i, s in enumerate(self._srcs):
            if pname not in s: 
                continue
            if pmin is not None and s[pname] < pmin: 
                continue
            if pmax is not None and s[pname] > pmax: 
                continue
            srcs.append(s)
        return srcs

    def get_sources_by_position(self, skydir, dist, min_dist=None,
                                square=False, coordsys='CEL'):
        """Retrieve sources within a certain angular distance of a sky
        coordinate.  This function supports two types of geometric
        selections: circular (square=False) and square (square=True).
        The circular selection finds all sources with a given angular
        distance of the target position.  The square selection finds
        sources within an ROI-like region of size R x R where R = 2 x
        dist.

        Parameters
        ----------

        skydir : `~astropy.coordinates.SkyCoord` 
            Sky direction with respect to which the selection will be applied.

        dist : float
            Maximum distance in degrees from the sky coordinate.

        square : bool
            Choose whether to apply a circular or square selection.

        coordsys : str
            Coordinate system to use when applying a selection with square=True.

        """

        msk = get_skydir_distance_mask(self._src_skydir, skydir, dist,
                                       min_dist=min_dist, square=square,
                                       coordsys=coordsys)

        radius = self._src_skydir.separation(skydir).deg
        radius = radius[msk]

        srcs = [self._srcs[i] for i in np.nonzero(msk)[0]]

        isort = np.argsort(radius)
        radius = radius[isort]
        srcs = [srcs[i] for i in isort]

        return radius, srcs

    def load_fits_catalog(self, name, **kwargs):
        """Load sources from a FITS catalog file.

        Parameters
        ----------

        name : str
            Catalog name or path to a catalog FITS file.
        """

        self.logger.debug('Loading FITS catalog: %s'%name)
        
        coordsys = kwargs.get('coordsys', 'CEL')
        extdir = kwargs.get('extdir', self.config['extdir'])

        cat = catalog.Catalog.create(name)

        m0 = get_skydir_distance_mask(cat.skydir, self.skydir,
                                      self.config['src_radius'])
        m1 = get_skydir_distance_mask(cat.skydir, self.skydir,
                                      self.config['src_radius_roi'],
                                      square=True, coordsys=coordsys)
        m = (m0 & m1)
        
        offset = self.skydir.separation(cat.skydir).deg
        offset_cel = wcs_utils.sky_to_offset(self.skydir,
                                         cat.radec[:, 0], cat.radec[:, 1],
                                         'CEL')
        offset_gal = wcs_utils.sky_to_offset(self.skydir,
                                         cat.glonlat[:, 0], cat.glonlat[:, 1],
                                         'GAL')

        for i, (row, radec) in enumerate(zip(cat.table[m],
                                             cat.radec[m])):
            
            catalog_dict = catalog.row_to_dict(row)
            src_dict = {'catalog': catalog_dict}
            src_dict['Source_Name'] = row['Source_Name']
            src_dict['SpectrumType'] = row['SpectrumType']

            if row['extended']:
                src_dict['SourceType'] = 'DiffuseSource'
                src_dict['SpatialType'] = 'SpatialMap'
                src_dict['SpatialModel'] = 'SpatialMap'

                search_dirs = []
                if extdir is not None:
                    search_dirs += [extdir, os.path.join(extdir, 'Templates')]
                
                search_dirs += [row['extdir'],
                                os.path.join(row['extdir'], 'Templates')]
                
                src_dict['Spatial_Filename'] = resolve_file_path(
                    row['Spatial_Filename'],
                    search_dirs=search_dirs)

            else:
                src_dict['SourceType'] = 'PointSource'
                src_dict['SpatialType'] = 'SkyDirFunction'
                src_dict['SpatialModel'] = 'PointSource'

            src = Source(src_dict['Source_Name'], src_dict, radec=radec)
            src.load_from_catalog()
            src.data['offset'] = offset[m][i]
            src.data['offset_ra'] = offset_cel[:, 0][m][i]
            src.data['offset_dec'] = offset_cel[:, 1][m][i]
            src.data['offset_glon'] = offset_gal[:, 0][m][i]
            src.data['offset_glat'] = offset_gal[:, 1][m][i]
            self.load_source(src, False,
                             merge_sources=self.config['merge_sources'])

        self._build_src_index()

    def load_xml(self, xmlfile, **kwargs):
        """Load sources from an XML file."""

        extdir = kwargs.get('extdir', self.config['extdir'])
        coordsys = kwargs.get('coordsys', 'CEL')
        if not os.path.isfile(xmlfile):
            xmlfile = os.path.join(fermipy.PACKAGE_DATA, 'catalogs', xmlfile)

        self.logger.info('Reading XML Model: ' + xmlfile)

        root = ElementTree.ElementTree(file=xmlfile).getroot()

        diffuse_srcs = []
        srcs = []
        ra, dec = [], []

        for s in root.findall('source'):
            src = Source.create_from_xml(s, extdir=extdir)            
            if src.diffuse:
                diffuse_srcs += [src]
            else:
                srcs += [src]
                ra += [src['RAJ2000']]
                dec += [src['DEJ2000']]

        src_skydir = SkyCoord(ra=np.array(ra) * u.deg,
                              dec=np.array(dec) * u.deg)
        radec = np.vstack((src_skydir.ra.deg, src_skydir.dec.deg)).T
        glonlat = np.vstack((src_skydir.galactic.l.deg,
                             src_skydir.galactic.b.deg)).T

        offset = self.skydir.separation(src_skydir).deg
        offset_cel = wcs_utils.sky_to_offset(self.skydir,
                                         radec[:, 0], radec[:, 1], 'CEL')
        offset_gal = wcs_utils.sky_to_offset(self.skydir,
                                         glonlat[:, 0], glonlat[:, 1], 'GAL')

        m0 = get_skydir_distance_mask(src_skydir, self.skydir,
                                      self.config['src_radius'])
        m1 = get_skydir_distance_mask(src_skydir, self.skydir,
                                      self.config['src_radius_roi'],
                                      square=True, coordsys=coordsys)
        m = (m0 & m1)
        srcs = np.array(srcs)[m]
        for i, s in enumerate(srcs):
            s.data['offset'] = offset[m][i]
            s.data['offset_ra'] = offset_cel[:, 0][m][i]
            s.data['offset_dec'] = offset_cel[:, 1][m][i]
            s.data['offset_glon'] = offset_gal[:, 0][m][i]
            s.data['offset_glat'] = offset_gal[:, 1][m][i]
            self.load_source(s, False,
                             merge_sources=self.config['merge_sources'])

        for i, s in enumerate(diffuse_srcs):
            self.load_source(s, False,
                             merge_sources=self.config['merge_sources'])

        self._build_src_index()

    def _build_src_index(self):
        """Build an indices for fast lookup of a source given its name
        or coordinates."""

        self._srcs = sorted(self._srcs, key=lambda t: t['offset'])
        nsrc = len(self._srcs)
        radec = np.zeros((2, nsrc))

        for i, src in enumerate(self._srcs):
            radec[:, i] = src.radec

        self._src_skydir = SkyCoord(ra=radec[0], dec=radec[1], unit=u.deg)
        self._src_radius = self._src_skydir.separation(self.skydir)

    def write_xml(self, xmlfile):
        """Save the ROI model as an XML file."""

        root = ElementTree.Element('source_library')
        root.set('title', 'source_library')

        for s in self._srcs:
            s.write_xml(root)

        for s in self._diffuse_srcs:
            s.write_xml(root)

        output_file = open(xmlfile, 'w')
        output_file.write(utils.prettify_xml(root))

    def write_fits(self,fitsfile):
        """Write the ROI model to a FITS file."""

        scan_shape = (1,)
        for src in self._srcs:
            scan_shape = max(scan_shape,src['dloglike_scan'].shape)
            
        cols_dict = collections.OrderedDict()
        cols_dict['Source_Name'] = dict(dtype='S20', format='%s')
        cols_dict['name'] = dict(dtype='S20', format='%s')
        cols_dict['class'] = dict(dtype='S20', format='%s')
        cols_dict['SpectrumType'] = dict(dtype='S20', format='%s')
        cols_dict['SpatialType'] = dict(dtype='S20', format='%s')
        cols_dict['SourceType'] = dict(dtype='S20', format='%s')
        cols_dict['SpatialModel'] = dict(dtype='S20', format='%s')
        cols_dict['RAJ2000'] = dict(dtype='f8', format='%.3f',unit='deg')
        cols_dict['DEJ2000'] = dict(dtype='f8', format='%.3f',unit='deg')
        cols_dict['GLON'] = dict(dtype='f8', format='%.3f',unit='deg')
        cols_dict['GLAT'] = dict(dtype='f8', format='%.3f',unit='deg')
        cols_dict['ts'] = dict(dtype='f8', format='%.3f')
        cols_dict['loglike'] = dict(dtype='f8', format='%.3f')
        cols_dict['npred'] = dict(dtype='f8', format='%.3f')
        cols_dict['offset'] = dict(dtype='f8', format='%.3f',unit='deg')
        cols_dict['offset_ra'] = dict(dtype='f8', format='%.3f',unit='deg')
        cols_dict['offset_dec'] = dict(dtype='f8', format='%.3f',unit='deg')
        cols_dict['offset_glon'] = dict(dtype='f8', format='%.3f',unit='deg')
        cols_dict['offset_glat'] = dict(dtype='f8', format='%.3f',unit='deg')
        cols_dict['pivot_energy'] = dict(dtype='f8', format='%.3f',unit='MeV')
        cols_dict['flux_scan'] = dict(dtype='f8', format='%.3f',
                                      shape=scan_shape)
        cols_dict['eflux_scan'] = dict(dtype='f8', format='%.3f',
                                       shape=scan_shape)
        cols_dict['dloglike_scan'] = dict(dtype='f8', format='%.3f',
                                          shape=scan_shape)

        # Catalog Parameters
        cols_dict['Flux_Density'] = dict(dtype='f8', format='%.5g',unit='1 / (MeV cm2 s)')
        cols_dict['Spectral_Index'] = dict(dtype='f8', format='%.3f')
        cols_dict['Pivot_Energy'] = dict(dtype='f8', format='%.3f',unit='MeV')
        cols_dict['beta'] = dict(dtype='f8', format='%.3f')
        cols_dict['Exp_Index'] = dict(dtype='f8', format='%.3f')
        cols_dict['Cutoff'] = dict(dtype='f8', format='%.3f',unit='MeV')

        cols_dict['Conf_68_PosAng'] = dict(dtype='f8', format='%.3f',unit='deg')
        cols_dict['Conf_68_SemiMajor'] = dict(dtype='f8', format='%.3f',unit='deg')
        cols_dict['Conf_68_SemiMinor'] = dict(dtype='f8', format='%.3f',unit='deg')
        cols_dict['Conf_95_PosAng'] = dict(dtype='f8', format='%.3f',unit='deg')
        cols_dict['Conf_95_SemiMajor'] = dict(dtype='f8', format='%.3f',unit='deg')
        cols_dict['Conf_95_SemiMinor'] = dict(dtype='f8', format='%.3f',unit='deg')
        
        
        for t in ['eflux','eflux100','eflux1000','eflux10000']:
            cols_dict[t] = dict(dtype='f8', format='%.3f',unit='MeV / (cm2 s)',shape=(2,))

        for t in ['eflux_ul95','eflux100_ul95','eflux1000_ul95','eflux10000_ul95']:
            cols_dict[t] = dict(dtype='f8', format='%.3f',unit='MeV / (cm2 s)')
            
        for t in ['flux','flux100','flux1000','flux10000']:
            cols_dict[t] = dict(dtype='f8', format='%.3f',unit='1 / (cm2 s)',shape=(2,))

        for t in ['flux_ul95','flux100_ul95','flux1000_ul95','flux10000_ul95']:
            cols_dict[t] = dict(dtype='f8', format='%.3f',unit='1 / (cm2 s)')
            
        for t in ['dfde','dfde100','dfde1000','dfde10000']:
            cols_dict[t] = dict(dtype='f8', format='%.3f',unit='1 / (MeV cm2 s)',shape=(2,))

#        for t in ['e2dfde','e2dfde100','e2dfde1000','e2dfde10000']:
#            cols_dict[t] = dict(dtype='f8', format='%.3f',unit='MeV / (cm2 s)',shape=(2,))

        cols = [Column(name=k, **v) for k,v in cols_dict.items()]
        tab = Table(cols)

        row_dict = {} 
        
        for s in self._srcs:

            row_dict['Source_Name'] = s['name']
            row_dict['RAJ2000'] = s['ra']
            row_dict['DEJ2000'] = s['dec']
            row_dict['GLON'] = s['glon']
            row_dict['GLAT'] = s['glat']

            r68_semimajor = s['pos_sigma_semimajor']*s['pos_r68']/s['pos_sigma']
            r68_semiminor = s['pos_sigma_semiminor']*s['pos_r68']/s['pos_sigma']
            r95_semimajor = s['pos_sigma_semimajor']*s['pos_r95']/s['pos_sigma']
            r95_semiminor = s['pos_sigma_semiminor']*s['pos_r95']/s['pos_sigma']
            
            row_dict['Conf_68_PosAng'] = s['pos_angle']
            row_dict['Conf_68_SemiMajor'] = r68_semimajor
            row_dict['Conf_68_SemiMinor'] = r68_semiminor
            row_dict['Conf_95_PosAng'] = s['pos_angle']
            row_dict['Conf_95_SemiMajor'] = r95_semimajor
            row_dict['Conf_95_SemiMinor'] = r95_semiminor
            
            row_dict.update(s.get_catalog_dict())
                            
            for t in s.data.keys():
                if t in cols_dict.keys():
                    row_dict[t] = s[t]
            
            row  = [row_dict[k] for k in cols_dict.keys()]
            tab.add_row(row)

            
        tab.write(fitsfile,format='fits',overwrite=True)

        hdulist = pyfits.open(fitsfile)
        for h in hdulist:
            h.header['CREATOR'] = 'fermipy ' + fermipy.__version__
        hdulist.writeto(fitsfile, clobber=True)
        
if __name__ == '__main__':
    roi = ROIModel()

    roi.load_fits('gll_fssc_psc_v14.fit')

    src = roi.get_source_by_name('lmc')

    import pprint
    pprint.pprint(src.data)
    print(src)
    srcs = roi.get_nearby_sources('lmc', 10.0)
    roi.create_roi_from_source('test.xml', 'lmc', 'test', 'test', 90.0)
