# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import os
import copy
import re
import collections
import numpy as np
import xml.etree.cElementTree as ElementTree

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, Column

import fermipy
import fermipy.config
from fermipy import utils
from fermipy import wcs_utils
from fermipy import catalog
from fermipy import defaults
from fermipy import model_utils
from fermipy.logger import Logger, log_level
from fermipy.model_utils import make_parameter_dict
from fermipy.model_utils import cast_pars_dict
from fermipy.model_utils import get_function_defaults
from fermipy.model_utils import get_spatial_type
from fermipy.model_utils import get_function_norm_par_name
from fermipy.model_utils import get_function_par_names
from fermipy.model_utils import extract_pars_from_dict
from fermipy.model_utils import create_pars_from_dict


def create_source_table(scan_shape):
    """Create an empty source table.

    Returns
    -------
    tab : `~astropy.table.Table`
    """

    cols_dict = collections.OrderedDict()
    cols_dict['Source_Name'] = dict(dtype='S48', format='%s')
    cols_dict['name'] = dict(dtype='S48', format='%s')
    cols_dict['class'] = dict(dtype='S32', format='%s')
    cols_dict['SpectrumType'] = dict(dtype='S32', format='%s')
    cols_dict['SpatialType'] = dict(dtype='S32', format='%s')
    cols_dict['SourceType'] = dict(dtype='S32', format='%s')
    cols_dict['SpatialModel'] = dict(dtype='S32', format='%s')
    cols_dict['RAJ2000'] = dict(dtype='f8', format='%.3f', unit='deg')
    cols_dict['DEJ2000'] = dict(dtype='f8', format='%.3f', unit='deg')
    cols_dict['GLON'] = dict(dtype='f8', format='%.3f', unit='deg')
    cols_dict['GLAT'] = dict(dtype='f8', format='%.3f', unit='deg')
    cols_dict['ts'] = dict(dtype='f8', format='%.3f')
    cols_dict['loglike'] = dict(dtype='f8', format='%.3f')
    cols_dict['npred'] = dict(dtype='f8', format='%.3f')
    cols_dict['offset'] = dict(dtype='f8', format='%.3f', unit='deg')
    cols_dict['offset_ra'] = dict(dtype='f8', format='%.3f', unit='deg')
    cols_dict['offset_dec'] = dict(dtype='f8', format='%.3f', unit='deg')
    cols_dict['offset_glon'] = dict(dtype='f8', format='%.3f', unit='deg')
    cols_dict['offset_glat'] = dict(dtype='f8', format='%.3f', unit='deg')
    cols_dict['offset_roi_edge'] = dict(dtype='f8', format='%.3f', unit='deg')
    cols_dict['pivot_energy'] = dict(dtype='f8', format='%.3f', unit='MeV')
    cols_dict['flux_scan'] = dict(dtype='f8', format='%.3f',
                                  shape=scan_shape)
    cols_dict['eflux_scan'] = dict(dtype='f8', format='%.3f',
                                   shape=scan_shape)
    cols_dict['norm_scan'] = dict(dtype='f8', format='%.3f',
                                  shape=scan_shape)
    cols_dict['dloglike_scan'] = dict(dtype='f8', format='%.3f',
                                      shape=scan_shape)
    cols_dict['loglike_scan'] = dict(dtype='f8', format='%.3f',
                                     shape=scan_shape)

    # Add source dictionary columns
    for k, v in sorted(defaults.source_output.items()):
        if not k in cols_dict.keys():
            if v[2] == float:
                cols_dict[k] = dict(dtype='f8', format='%f')
            elif k == 'Spectrum_Filename' or k == 'Spatial_Filename':
                cols_dict[k] = dict(dtype='S128', format='%s')
            elif v[2] == str:
                cols_dict[k] = dict(dtype='S32', format='%s')

    cols_dict['param_names'] = dict(dtype='S32', format='%s', shape=(10,))
    cols_dict['param_values'] = dict(dtype='f8', format='%f', shape=(10,))
    cols_dict['param_errors'] = dict(dtype='f8', format='%f', shape=(10,))

    # Catalog Parameters
    cols_dict['Flux_Density'] = dict(
        dtype='f8', format='%.5g', unit='1 / (MeV cm2 s)')
    cols_dict['Spectral_Index'] = dict(dtype='f8', format='%.3f')
    cols_dict['Pivot_Energy'] = dict(dtype='f8', format='%.3f', unit='MeV')
    cols_dict['beta'] = dict(dtype='f8', format='%.3f')
    cols_dict['Exp_Index'] = dict(dtype='f8', format='%.3f')
    cols_dict['Cutoff'] = dict(dtype='f8', format='%.3f', unit='MeV')

    cols_dict['Conf_68_PosAng'] = dict(dtype='f8', format='%.3f', unit='deg')
    cols_dict['Conf_68_SemiMajor'] = dict(
        dtype='f8', format='%.3f', unit='deg')
    cols_dict['Conf_68_SemiMinor'] = dict(
        dtype='f8', format='%.3f', unit='deg')
    cols_dict['Conf_95_PosAng'] = dict(dtype='f8', format='%.3f', unit='deg')
    cols_dict['Conf_95_SemiMajor'] = dict(
        dtype='f8', format='%.3f', unit='deg')
    cols_dict['Conf_95_SemiMinor'] = dict(
        dtype='f8', format='%.3f', unit='deg')

    for t in ['eflux', 'eflux100', 'eflux1000', 'eflux10000']:
        cols_dict[t] = dict(dtype='f8', format='%.3f', unit='MeV / (cm2 s)')
        cols_dict[t + '_err'] = dict(dtype='f8',
                                     format='%.3f', unit='MeV / (cm2 s)')

    for t in ['eflux_ul95', 'eflux100_ul95', 'eflux1000_ul95', 'eflux10000_ul95']:
        cols_dict[t] = dict(dtype='f8', format='%.3f', unit='MeV / (cm2 s)')

    for t in ['flux', 'flux100', 'flux1000', 'flux10000']:
        cols_dict[t] = dict(dtype='f8', format='%.3f', unit='1 / (cm2 s)')
        cols_dict[t + '_err'] = dict(dtype='f8',
                                     format='%.3f', unit='1 / (cm2 s)')

    for t in ['flux_ul95', 'flux100_ul95', 'flux1000_ul95', 'flux10000_ul95']:
        cols_dict[t] = dict(dtype='f8', format='%.3f', unit='1 / (cm2 s)')

    for t in ['dnde', 'dnde100', 'dnde1000', 'dnde10000']:
        cols_dict[t] = dict(dtype='f8', format='%.3f', unit='1 / (MeV cm2 s)')
        cols_dict[t + '_err'] = dict(dtype='f8',
                                     format='%.3f', unit='1 / (MeV cm2 s)')

    cols = [Column(name=k, **v) for k, v in cols_dict.items()]
    tab = Table(cols)
    return tab


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

    if dist is None:
        dist = 180.

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
    if min_dist is not None:
        msk &= (dtheta > np.radians(min_dist))
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


def get_true_params_dict(pars_dict):

    params = {}
    for k, p in pars_dict.items():
        val = p['value'] * p['scale']
        err = np.nan
        if 'error' in p:
            err = p['error'] * np.abs(p['scale'])
        params[k] = {'value': val, 'error': err}

    return params


def spectral_pars_from_catalog(cat):
    """Create spectral parameters from 3FGL catalog columns."""

    spectrum_type = cat['SpectrumType']
    pars = get_function_defaults(cat['SpectrumType'])

    if spectrum_type == 'PowerLaw':

        pars['Prefactor']['value'] = cat['Flux_Density']
        pars['Scale']['value'] = cat['Pivot_Energy']
        pars['Scale']['scale'] = 1.0
        pars['Index']['value'] = cat['Spectral_Index']
        pars['Index']['max'] = max(5.0, pars['Index']['value'] + 1.0)
        pars['Index']['min'] = min(0.0, pars['Index']['value'] - 1.0)
        pars['Index']['scale'] = -1.0

        pars['Prefactor'] = make_parameter_dict(pars['Prefactor'])
        pars['Scale'] = make_parameter_dict(pars['Scale'], True, False)
        pars['Index'] = make_parameter_dict(pars['Index'], False, False)

    elif spectrum_type == 'LogParabola':

        pars['norm']['value'] = cat['Flux_Density']
        pars['Eb']['value'] = cat['Pivot_Energy']
        pars['alpha']['value'] = cat['Spectral_Index']
        pars['beta']['value'] = cat['beta']

        pars['norm'] = make_parameter_dict(pars['norm'], False, True)
        pars['Eb'] = make_parameter_dict(pars['Eb'], True, False)
        pars['alpha'] = make_parameter_dict(pars['alpha'], False, False)
        pars['beta'] = make_parameter_dict(pars['beta'], False, False)

    elif spectrum_type == 'PLSuperExpCutoff':

        flux_density = cat['Flux_Density']
        prefactor = (cat['Flux_Density'] *
                     np.exp((cat['Pivot_Energy'] / cat['Cutoff']) **
                            cat['Exp_Index']))

        pars['Prefactor']['value'] = prefactor
        pars['Index1']['value'] = cat['Spectral_Index']
        pars['Index1']['scale'] = -1.0
        pars['Index2']['value'] = cat['Exp_Index']
        pars['Index2']['scale'] = 1.0
        pars['Scale']['value'] = cat['Pivot_Energy']
        pars['Cutoff']['value'] = cat['Cutoff']

        pars['Prefactor'] = make_parameter_dict(pars['Prefactor'])
        pars['Scale'] = make_parameter_dict(pars['Scale'], True, False)
        pars['Index1'] = make_parameter_dict(pars['Index1'], False, False)
        pars['Index2'] = make_parameter_dict(pars['Index2'], False, False)
        pars['Cutoff'] = make_parameter_dict(pars['Cutoff'], False, True)

    else:
        raise Exception('Unsupported spectral type:' + spectrum_type)

    return pars


class Model(object):
    """Base class for point-like and diffuse source components.  This
    class is a container for spectral and spatial parameters as well
    as other source properties such as TS, Npred, and location within
    the ROI.
    """

    def __init__(self, name, data):

        self._data = defaults.make_default_dict(defaults.source_output)
        self._data['spectral_pars'] = get_function_defaults(
            data['SpectrumType'])
        self._data['spatial_pars'] = get_function_defaults(data['SpatialType'])
        self._data.setdefault('catalog', data.pop('catalog', {}))
        self._data.setdefault('assoc', data.pop('assoc', {}))
        self._data.setdefault('class', '')
        self._data['name'] = name
        self._data.setdefault('psf_scale_fn', None)
        self._data = utils.merge_dict(self._data, data)
        self._names = [name]
        catalog = self._data['catalog']

        if 'CLASS1' in catalog:
            self['class'] = catalog['CLASS1'].strip()
        elif 'CLASS' in catalog:
            self['class'] = catalog['CLASS'].strip()

        for k in ROIModel.src_name_cols:

            if k not in catalog:
                continue
            name = catalog[k].strip()
            if name != '' and name not in self._names:
                self._names.append(name)

            self._data['assoc'][k] = name

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

        for i, p in enumerate(self['param_names']):
            if not p:
                break
            val = self['param_values'][i]
            err = self['param_errors'][i]
            output += ['{:15s}: {:10.4g} +/- {:10.4g}'.format(p, val, err)]

        return '\n'.join(output).format(**data)

    def items(self):
        return self._data.items()

    @property
    def data(self):
        return self._data

    @property
    def spectral_pars(self):
        return self._data['spectral_pars']

    @property
    def spatial_pars(self):
        return self._data['spatial_pars']

    @property
    def params(self):
        return get_true_params_dict(self._data['spectral_pars'])

    @property
    def name(self):
        return self._data['name']

    @property
    def names(self):
        return self._names

    @property
    def assoc(self):
        return self._data['assoc']

    @property
    def psf_scale_fn(self):
        return self._data['psf_scale']

    @staticmethod
    def create_from_dict(src_dict, roi_skydir=None, rescale=False):

        src_dict = copy.deepcopy(src_dict)
        src_dict.setdefault('SpatialModel', 'PointSource')
        src_dict.setdefault('SpatialType',
                            get_spatial_type(src_dict['SpatialModel']))

        # Need this to handle old conventions for
        # MapCubeFunction/ConstantValue sources
        if src_dict['SpatialModel'] == 'DiffuseSource':
            src_dict['SpatialModel'] = src_dict['SpatialType']

            if 'filefunction' in src_dict:
                src_dict['Spectrum_Filename'] = src_dict.pop('filefunction')

            if 'mapcube' in src_dict:
                src_dict['Spatial_Filename'] = src_dict.pop('mapcube')

        if 'spectral_pars' in src_dict:
            src_dict['spectral_pars'] = cast_pars_dict(
                src_dict['spectral_pars'])

        if 'spatial_pars' in src_dict:
            src_dict['spatial_pars'] = cast_pars_dict(src_dict['spatial_pars'])

        if src_dict['SpatialModel'] == 'ConstantValue':
            return IsoSource(src_dict['name'], src_dict)
        elif src_dict['SpatialModel'] == 'CompositeSource':
            return CompositeSource(src_dict['name'], src_dict)
        elif src_dict['SpatialModel'] == 'MapCubeFunction':
            return MapCubeSource(src_dict['name'], src_dict)
        else:
            return Source.create_from_dict(src_dict, roi_skydir,
                                           rescale=rescale)

    def _sync_params(self):
        pars = model_utils.pars_dict_to_vectors(self['SpectrumType'],
                                                self.spectral_pars)
        self._data.update(pars)

    def get_norm(self):

        par_name = get_function_norm_par_name(self['SpectrumType'])
        val = self.spectral_pars[par_name]['value']
        scale = self.spectral_pars[par_name]['scale']
        return float(val) * float(scale)

    def add_to_table(self, tab):

        row_dict = {}
        row_dict['Source_Name'] = self['name']
        row_dict['RAJ2000'] = self['ra']
        row_dict['DEJ2000'] = self['dec']
        row_dict['GLON'] = self['glon']
        row_dict['GLAT'] = self['glat']

        if not 'param_names' in self.data:
            pars = model_utils.pars_dict_to_vectors(self['SpectrumType'],
                                                    self.spectral_pars)
            row_dict.update(pars)

        r68_semimajor = self['pos_sigma_semimajor'] * \
            self['pos_r68'] / self['pos_sigma']
        r68_semiminor = self['pos_sigma_semiminor'] * \
            self['pos_r68'] / self['pos_sigma']
        r95_semimajor = self['pos_sigma_semimajor'] * \
            self['pos_r95'] / self['pos_sigma']
        r95_semiminor = self['pos_sigma_semiminor'] * \
            self['pos_r95'] / self['pos_sigma']

        row_dict['Conf_68_PosAng'] = self['pos_angle']
        row_dict['Conf_68_SemiMajor'] = r68_semimajor
        row_dict['Conf_68_SemiMinor'] = r68_semiminor
        row_dict['Conf_95_PosAng'] = self['pos_angle']
        row_dict['Conf_95_SemiMajor'] = r95_semimajor
        row_dict['Conf_95_SemiMinor'] = r95_semiminor

        row_dict.update(self.get_catalog_dict())

        for t in self.data.keys():

            if t == 'params':
                continue
            if t in tab.columns:
                row_dict[t] = self[t]

        row = [row_dict[k] for k in tab.columns]
        tab.add_row(row)
    
    def get_catalog_dict(self):

        o = {'Spectral_Index': np.nan,
             'Flux_Density': np.nan,
             'Pivot_Energy': np.nan,
             'beta': np.nan,
             'Exp_Index': np.nan,
             'Cutoff': np.nan}

        params = get_true_params_dict(self.spectral_pars)
        if self['SpectrumType'] == 'PowerLaw':
            o['Spectral_Index'] = -1.0 * params['Index']['value']
            o['Flux_Density'] = params['Prefactor']['value']
            o['Pivot_Energy'] = params['Scale']['value']
        elif self['SpectrumType'] == 'LogParabola':
            o['Spectral_Index'] = params['alpha']['value']
            o['Flux_Density'] = params['norm']['value']
            o['Pivot_Energy'] = params['Eb']['value']
            o['beta'] = params['beta']['value']
        elif self['SpectrumType'] == 'PLSuperExpCutoff':
            o['Spectral_Index'] = -1.0 * params['Index1']['value']
            o['Exp_Index'] = params['Index2']['value']
            o['Flux_Density'] = params['Prefactor']['value']
            o['Pivot_Energy'] = params['Scale']['value']
            o['Cutoff'] = params['Cutoff']['value']

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

    def set_psf_scale_fn(self, fn):
        self._data['psf_scale_fn'] = fn

    def set_spectral_pars(self, spectral_pars):

        self._data['spectral_pars'] = copy.deepcopy(spectral_pars)
        self._sync_params()

    def update_spectral_pars(self, spectral_pars):

        self._data['spectral_pars'] = utils.merge_dict(
            self.spectral_pars, spectral_pars)
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
                                  'free': False}}

        super(IsoSource, self).__init__(name, data)

        self._init_spatial_pars()

    @property
    def filefunction(self):
        return self._data['Spectrum_Filename']

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

        filename = utils.path_to_xmlpath(self.filefunction)
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

        data.setdefault('SpectrumType', 'PowerLaw')
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
        return self._data['Spatial_Filename']

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
                                           dict(type=self.data['SpectrumType']))

        filename = utils.path_to_xmlpath(self.mapcube)
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

        data.setdefault('SpatialModel', 'PointSource')
        data.setdefault('SpectrumType', 'PowerLaw')
        data.setdefault(
            'SpatialType', model_utils.get_spatial_type(data['SpatialModel']))
        data.setdefault(
            'SourceType', model_utils.get_source_type(data['SpatialType']))

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

        self._init_spatial_pars(SpatialWidth=self['SpatialWidth'])

    def __str__(self):

        data = copy.deepcopy(self.data)
        data['names'] = self.names
        output = []
        output += ['{:15s}:'.format('Name') + ' {name:s}']
        output += ['{:15s}:'.format('Associations') + ' {names:s}']
        output += ['{:15s}:'.format('RA/DEC') + ' {ra:10.3f}/{dec:10.3f}']
        output += ['{:15s}:'.format('GLON/GLAT') +
                   ' {glon:10.3f}/{glat:10.3f}']
        output += ['{:15s}:'.format('TS') + ' {ts:.2f}']
        output += ['{:15s}:'.format('Npred') + ' {npred:.2f}']
        output += ['{:15s}:'.format('Flux') +
                   ' {flux:9.4g} +/- {flux_err:8.3g}']
        output += ['{:15s}:'.format('EnergyFlux') +
                   ' {eflux:9.4g} +/- {eflux_err:8.3g}']
        output += ['{:15s}:'.format('SpatialModel') + ' {SpatialModel:s}']
        output += ['{:15s}:'.format('SpectrumType') + ' {SpectrumType:s}']
        output += ['Spectral Parameters']

        for i, p in enumerate(self['param_names']):
            if not p:
                break
            val = self['param_values'][i]
            err = self['param_errors'][i]
            output += ['{:15s}: {:10.4g} +/- {:10.4g}'.format(p, val, err)]

        return '\n'.join(output).format(**data)

    def _set_radec(self, radec):

        self['radec'] = np.array(radec, ndmin=1)
        self['RAJ2000'] = radec[0]
        self['DEJ2000'] = radec[1]
        self['ra'] = radec[0]
        self['dec'] = radec[1]
        glonlat = utils.eq2gal(radec[0], radec[1])
        self['glon'], self['glat'] = glonlat[0][0], glonlat[1][0]
        if 'RA' in self.spatial_pars:
            self.spatial_pars['RA']['value'] = radec[0]
            self.spatial_pars['DEC']['value'] = radec[1]

    def _set_spatial_width(self, spatial_width):
        self.data['SpatialWidth'] = spatial_width
        if self['SpatialType'] in ['RadialGaussian']:
            self.spatial_pars['Sigma'][
                'value'] = spatial_width / 1.5095921854516636
        elif self['SpatialType'] in ['RadialDisk']:
            self.spatial_pars['Radius'][
                'value'] = spatial_width / 0.8246211251235321

    def _init_spatial_pars(self, **kwargs):

        spatial_pars = copy.deepcopy(kwargs)
        spatial_width = spatial_pars.pop('SpatialWidth', None)

        if self['SpatialType'] == 'SkyDirFunction':
            self._extended = False
            self._data['SourceType'] = 'PointSource'
        else:
            self._extended = True
            self._data['SourceType'] = 'DiffuseSource'

        spatial_pars.setdefault('RA', spatial_pars.pop('ra', self['ra']))
        spatial_pars.setdefault('DEC', spatial_pars.pop('dec', self['dec']))

        for k, v in spatial_pars.items():

            if not isinstance(v, dict):
                spatial_pars[k] = {'name': k, 'value': v}

            if k in self.spatial_pars:
                self.spatial_pars[k].update(spatial_pars[k])

        if spatial_width is not None:
            self._set_spatial_width(spatial_width)
        elif self['SpatialType'] == 'RadialDisk':
            self['SpatialWidth'] = self.spatial_pars[
                'Radius']['value'] * 0.8246211251235321
        elif self['SpatialType'] == 'RadialGaussian':
            self['SpatialWidth'] = self.spatial_pars[
                'Sigma']['value'] * 1.5095921854516636

        if 'RA' in spatial_pars or 'DEC' in spatial_pars:
            self._set_radec([spatial_pars['RA']['value'],
                             spatial_pars['DEC']['value']])

    def update_data(self, d):
        self._data = utils.merge_dict(self._data, d, add_new_keys=True)
        if 'ra' in d and 'dec' in d:
            self._set_radec([d['ra'], d['dec']])

    def set_radec(self, ra, dec):
        self._set_radec(np.array([ra, dec]))

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

    def set_roi_direction(self, roidir):

        offset = roidir.separation(self.skydir).deg
        offset_cel = wcs_utils.sky_to_offset(
            roidir, self['ra'], self['dec'], 'CEL')
        offset_gal = wcs_utils.sky_to_offset(
            roidir, self['glon'], self['glat'], 'GAL')

        self['offset'] = offset
        self['offset_ra'] = offset_cel[0, 0]
        self['offset_dec'] = offset_cel[0, 1]
        self['offset_glon'] = offset_gal[0, 0]
        self['offset_glat'] = offset_gal[0, 1]

    def set_roi_projection(self, proj):

        if proj is None:
            return

        self['offset_roi_edge'] = proj.distance_to_edge(self.skydir)

    def set_spatial_model(self, spatial_model, spatial_pars):

        update_pars = False
        if spatial_model != self['SpatialModel']:
            update_pars = True
        self._data['SpatialModel'] = spatial_model
        self._data['SpatialType'] = get_spatial_type(self['SpatialModel'])
        if update_pars:
            self._data['spatial_pars'] = get_function_defaults(
                self['SpatialType'])

        if spatial_model == 'PointSource':
            self._data['SpatialWidth'] = None

        self._init_spatial_pars(**spatial_pars)

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
    def create_from_dict(src_dict, roi_skydir=None, rescale=False):
        """Create a source object from a python dictionary.

        Parameters
        ----------
        src_dict : dict
           Dictionary defining the properties of the source.

        """
        src_dict = copy.deepcopy(src_dict)
        src_dict.setdefault('SpatialModel', 'PointSource')
        src_dict.setdefault('Spectrum_Filename', None)
        src_dict.setdefault('SpectrumType', 'PowerLaw')
        src_dict['SpatialType'] = get_spatial_type(src_dict['SpatialModel'])

        spectrum_type = src_dict['SpectrumType']
        spatial_type = src_dict['SpatialType']

        spectral_pars = src_dict.pop('spectral_pars', {})
        spatial_pars = src_dict.pop('spatial_pars', {})

        if not spectral_pars:
            spectral_pars = extract_pars_from_dict(spectrum_type, src_dict)
            norm_par_name = get_function_norm_par_name(spectrum_type)
            if norm_par_name is not None:
                spectral_pars[norm_par_name].setdefault('free', True)

        if not spatial_pars:
            spatial_pars = extract_pars_from_dict(spatial_type, src_dict)
            for k in ['RA', 'DEC', 'Prefactor']:
                if k in spatial_pars:
                    del spatial_pars[k]

        spectral_pars = create_pars_from_dict(spectrum_type, spectral_pars,
                                              rescale)
        spatial_pars = create_pars_from_dict(spatial_type, spatial_pars,
                                             False)

        if 'file' in src_dict:
            src_dict['Spectrum_Filename'] = src_dict.pop('file')

        if spectrum_type == 'DMFitFunction' and src_dict['Spectrum_Filename'] is None:
            src_dict['Spectrum_Filename'] = os.path.join('$FERMIPY_DATA_DIR',
                                                         'gammamc_dif.dat')

        src_dict['spectral_pars'] = cast_pars_dict(spectral_pars)
        src_dict['spatial_pars'] = cast_pars_dict(spatial_pars)

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
    def create_from_xmlfile(xmlfile, extdir=None):
        """Create a Source object from an XML file.

        Parameters
        ----------
        xmlfile : str
            Path to XML file.

        extdir : str
            Path to the extended source archive.
        """
        root = ElementTree.ElementTree(file=xmlfile).getroot()
        srcs = root.findall('source')
        if len(srcs) == 0:
            raise Exception('No sources found.')
        return Source.create_from_xml(srcs[0], extdir=extdir)

    @staticmethod
    def create_from_xml(root, extdir=None):
        """Create a Source object from an XML node.

        Parameters
        ----------
        root : `~xml.etree.ElementTree.Element`
            XML node containing the source.

        extdir : str
            Path to the extended source archive.
        """

        src_type = root.attrib['type']
        spec = utils.load_xml_elements(root, 'spectrum')
        spectral_pars = utils.load_xml_elements(root, 'spectrum/parameter')
        spectral_type = spec['type']
        spectral_pars = cast_pars_dict(spectral_pars)
        spat = {}
        spatial_pars = {}
        nested_sources = []

        if src_type == 'CompositeSource':
            spatial_type = 'CompositeSource'
            source_library = root.findall('source_library')[0]
            for node in source_library.findall('source'):
                nested_sources += [Source.create_from_xml(node, extdir=extdir)]
        else:
            spat = utils.load_xml_elements(root, 'spatialModel')
            spatial_pars = utils.load_xml_elements(
                root, 'spatialModel/parameter')
            spatial_pars = cast_pars_dict(spatial_pars)
            spatial_type = spat['type']

        xml_dict = copy.deepcopy(root.attrib)
        src_dict = {'catalog': xml_dict}

        src_dict['Source_Name'] = xml_dict['name']
        src_dict['SpectrumType'] = spectral_type
        src_dict['SpatialType'] = spatial_type
        src_dict['SourceType'] = src_type
        src_dict['Spatial_Filename'] = None
        src_dict['Spectrum_Filename'] = None
        if 'file' in spat:
            src_dict['Spatial_Filename'] = utils.xmlpath_to_path(spat['file'])
            if not os.path.isfile(src_dict['Spatial_Filename']) \
                    and extdir is not None:
                src_dict['Spatial_Filename'] = \
                    os.path.join(extdir, 'Templates',
                                 src_dict['Spatial_Filename'])

        if 'file' in spec:
            src_dict['Spectrum_Filename'] = utils.xmlpath_to_path(spec['file'])

        if src_type == 'PointSource':
            src_dict['SpatialModel'] = 'PointSource'
        elif src_type == 'CompositeSource':
            src_dict['SpatialModel'] = 'CompositeSource'
        elif spatial_type == 'SpatialMap':
            src_dict['SpatialModel'] = 'SpatialMap'
        else:
            src_dict['SpatialModel'] = spatial_type

        if src_type == 'PointSource' or \
                spatial_type in ['SpatialMap', 'RadialGaussian', 'RadialDisk']:

            if 'RA' in xml_dict:
                src_dict['RAJ2000'] = float(xml_dict['RA'])
                src_dict['DEJ2000'] = float(xml_dict['DEC'])
            elif 'RA' in spatial_pars:
                src_dict['RAJ2000'] = float(spatial_pars['RA']['value'])
                src_dict['DEJ2000'] = float(spatial_pars['DEC']['value'])
            else:
                skydir = wcs_utils.get_map_skydir(os.path.expandvars(
                    src_dict['Spatial_Filename']))
                src_dict['RAJ2000'] = skydir.ra.deg
                src_dict['DEJ2000'] = skydir.dec.deg

            radec = np.array([src_dict['RAJ2000'], src_dict['DEJ2000']])

            src_dict['spectral_pars'] = spectral_pars
            src_dict['spatial_pars'] = spatial_pars
            return Source(src_dict['Source_Name'],
                          src_dict, radec=radec)

        elif src_type == 'DiffuseSource' and spatial_type == 'ConstantValue':
            return IsoSource(src_dict['Source_Name'],
                             {'Spectrum_Filename': spec['file'],
                              'spectral_pars': spectral_pars,
                              'spatial_pars': spatial_pars})
        elif src_type == 'DiffuseSource' and spatial_type == 'MapCubeFunction':
            return MapCubeSource(src_dict['Source_Name'],
                                 {'Spatial_Filename': spat['file'],
                                  'SpectrumType': spectral_type,
                                  'spectral_pars': spectral_pars,
                                  'spatial_pars': spatial_pars})
        elif src_type == 'CompositeSource':
            return CompositeSource(src_dict['Source_Name'],
                                   {'SpectrumType': spectral_type,
                                    'nested_sources': nested_sources})
        else:
            raise Exception(
                'Unrecognized type for source: %s %s' % (src_dict['Source_Name'], src_type))

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

            filename = utils.path_to_xmlpath(self['Spatial_Filename'])
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

        if self['Spectrum_Filename'] is not None:
            filename = utils.path_to_xmlpath(self['Spectrum_Filename'])
            el.set('file', filename)

        for k, v in self.spectral_pars.items():
            utils.create_xml_element(el, 'parameter', v)


class CompositeSource(Model):

    def __init__(self, name, data):

        data.setdefault('SpectrumType', 'ConstantValue')
        data['SpatialType'] = 'CompositeSource'
        data['SpatialModel'] = 'CompositeSource'
        data['SourceType'] = 'CompositeSource'

        if not 'spectral_pars' in data:
            data['spectral_pars'] = {
                'Value': {'name': 'Value', 'scale': 1.0,
                          'value': 1.0, 'min': 0.1, 'max': '10.0',
                          'free': False},
            }

        super(CompositeSource, self).__init__(name, data)
        self._build_nested_sources(data)

    @property
    def nested_sources(self):
        return self._nested_sources

    @property
    def diffuse(self):
        return True

    def _build_nested_sources(self, data):
        self._nested_sources = []
        for nested_source in data.get('nested_sources', []):
            if isinstance(nested_source, Model):
                self._nested_sources.append(copy.deepcopy(nested_source))
            elif isinstance(nested_source, dict):
                self._nested_sources.append(
                    Source.create_from_dict(nested_source))

    def write_xml(self, root):

        source_element = utils.create_xml_element(root, 'source',
                                                  dict(name=self.name,
                                                       type='CompositeSource'))

        spec_el = utils.create_xml_element(source_element, 'spectrum',
                                           dict(type=self.data['SpectrumType']))

        for k, v in self.spectral_pars.items():
            utils.create_xml_element(spec_el, 'parameter', v)

        spat_el = utils.create_xml_element(
            source_element, 'source_library', dict(title=self.name))
        for nested_source in self._nested_sources:
            nested_source.write_xml(spat_el)


class ROIModel(fermipy.config.Configurable):
    """This class is responsible for managing the ROI model (both sources
    and diffuse components).  Source catalogs can be read
    from either FITS or XML files.  Individual components are
    represented by instances of `~fermipy.roi_model.Model` and can be
    accessed by name using the bracket operator.

        * Create an ROI with all 3FGL sources and print a summary of its contents:

        >>> skydir = astropy.coordinates.SkyCoord(0.0,0.0,unit='deg')
        >>> roi = ROIModel({'catalogs' : ['3FGL'],'src_roiwidth' : 10.0},skydir=skydir)
        >>> print(roi)
        name                SpatialModel   SpectrumType     offset        ts       npred
        --------------------------------------------------------------------------------
        3FGL J2357.3-0150   PointSource    PowerLaw          1.956       nan         0.0
        3FGL J0006.2+0135   PointSource    PowerLaw          2.232       nan         0.0
        3FGL J0016.3-0013   PointSource    PowerLaw          4.084       nan         0.0
        3FGL J0014.3-0455   PointSource    PowerLaw          6.085       nan         0.0

        * Print a summary of an individual source

        >>> print(roi['3FGL J0006.2+0135'])
        Name           : 3FGL J0006.2+0135
        Associations   : ['3FGL J0006.2+0135']
        RA/DEC         :      1.572/     1.585
        GLON/GLAT      :    100.400/   -59.297
        TS             : nan
        Npred          : nan
        Flux           :       nan +/-      nan
        EnergyFlux     :       nan +/-      nan
        SpatialModel   : PointSource
        SpectrumType   : PowerLaw
        Spectral Parameters
        Index          :         -2 +/-        nan
        Scale          :       1000 +/-        nan
        Prefactor      :      1e-12 +/-        nan

        * Get the SkyCoord for a source

        >>> dir = roi['SourceA'].skydir

        * Loop over all sources and print their names

        >>> for s in roi.sources: print(s.name)
        3FGL J2357.3-0150
        3FGL J0006.2+0135
        3FGL J0016.3-0013
        3FGL J0014.3-0455

    """

    defaults = dict(defaults.model.items(),
                    fileio=defaults.fileio)

    src_name_cols = ['Source_Name',
                     'ASSOC', 'ASSOC1', 'ASSOC2', 'ASSOC_GAM',
                     '1FHL_Name', '2FGL_Name', '3FGL_Name',
                     'ASSOC_GAM1', 'ASSOC_GAM2', 'ASSOC_TEV']

    def __init__(self, config=None, **kwargs):
        # Coordinate for ROI center (defaults to 0,0)
        self._skydir = kwargs.pop('skydir', SkyCoord(0.0, 0.0, unit=u.deg))
        self._projection = kwargs.get('projection', None)
        coordsys = kwargs.pop('coordsys', 'CEL')
        srcname = kwargs.pop('srcname', None)
        super(ROIModel, self).__init__(config, **kwargs)

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
        self._src_dict = collections.defaultdict(list)
        self._src_radius = []

        self.load(coordsys=coordsys, srcname=srcname)

#    def __getstate__(self):
#        d = self.__dict__.copy()
#        if 'logger' in d.keys():
#            d['logger'] = d['logger'].name
#        return d

#    def __setstate__(self, d):
#        if 'logger' in d.keys():
#            d['logger'] = \
#                Logger.get(self.__class__.__name__,
#                           d['_config']['logfile'],
#                           log_level(d['_config']['logging']['verbosity']))
#
#        self.__dict__.update(d)

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

            if s.diffuse:
                continue

            o += '%-20.19s%-15.14s%-15.14s%8.3f%10.2f%12.1f\n' % (
                s['name'], s['SpatialModel'],
                s['SpectrumType'],
                s['offset'], s['ts'], s['npred'])

        for s in sorted(self.sources, key=lambda t: t['offset']):

            if not s.diffuse:
                continue

            o += '%-20.19s%-15.14s%-15.14s%8s%10.2f%12.1f\n' % (
                s['name'], s['SpatialModel'],
                s['SpectrumType'],
                '-----', s['ts'], s['npred'])

        return o

    @property
    def skydir(self):
        """Return the sky direction corresponding to the center of the
        ROI."""
        return self._skydir

    @property
    def projection(self):
        return self._projection

    @property
    def sources(self):
        return self._srcs + self._diffuse_srcs

    @property
    def point_sources(self):
        return self._srcs

    @property
    def diffuse_sources(self):
        return self._diffuse_srcs

    def set_projection(self, proj):
        self._projection = proj
        for s in self._srcs:
            s.set_roi_projection(proj)

    def clear(self):
        """Clear the contents of the ROI."""
        self._srcs = []
        self._diffuse_srcs = []
        self._src_dict = collections.defaultdict(list)
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

        search_dirs = []
        search_dirs += self.config['diffuse_dir']
        search_dirs += [self.config['fileio']['outdir'],
                        os.path.join('$FERMIPY_ROOT', 'data'),
                        '$FERMI_DIFFUSE_DIR']

        srcs = []
        if self.config[name] is not None:
            srcs = self.config[name]

        for i, t in enumerate(srcs):

            if utils.isstr(t):
                src_dict = {'file': t}
            elif isinstance(t, dict):
                src_dict = copy.deepcopy(t)
            else:
                raise Exception(
                    'Invalid type in diffuse mode list: %s' % str(type(t)))

            src_dict['file'] = \
                utils.resolve_file_path(src_dict['file'],
                                        search_dirs=search_dirs)

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
                src = IsoSource(src_dict['name'], {
                                'Spectrum_Filename': src_dict['file']})
                altname = os.path.basename(src_dict['file'])
                altname = re.sub(r'(\.txt$)', '', altname)
            else:
                src = MapCubeSource(src_dict['name'], {
                                    'Spatial_Filename': src_dict['file']})
                altname = os.path.basename(src_dict['file'])
                altname = re.sub(r'(\.fits$|\.fit$|\.fits.gz$|\.fit.gz$)',
                                 '', altname)

            src.add_name(altname)
            self.load_source(src, False, self.config['merge_sources'])

    def create_source(self, name, src_dict, build_index=True,
                      merge_sources=True, rescale=True):
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

        if isinstance(src_dict, dict):
            src_dict['name'] = name
            src = Model.create_from_dict(src_dict, self.skydir,
                                         rescale=rescale)
        else:
            src = src_dict
            src.set_name(name)

        if isinstance(src, Source):
            src.set_roi_direction(self.skydir)
            src.set_roi_projection(self.projection)

        self.load_source(src, build_index=build_index,
                         merge_sources=merge_sources)

        return self.get_source_by_name(name)

    def copy_source(self, name):
        src = self.get_source_by_name(name)
        return copy.deepcopy(src)

    def load_sources(self, sources):
        """Delete all sources in the ROI and load the input source list."""

        self.clear()
        for s in sources:

            if isinstance(s, dict):
                s = Model.create_from_dict(s)

            self.load_source(s, build_index=False)
        self._build_src_index()

    def _add_source_alias(self, name, src):

        if src not in self._src_dict[name]:
            self._src_dict[name] += [src]

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

        min_sep = kwargs.get('min_separation', None)

        if min_sep is not None:

            sep = src.skydir.separation(self._src_skydir).deg
            if len(sep) > 0 and np.min(sep) < min_sep:
                return

        match_srcs = self.match_source(src)

        if len(match_srcs) == 1:

            # self.logger.debug('Found matching source for %s : %s',
            #                  src.name, match_srcs[0].name)

            if merge_sources:
                match_srcs[0].update_from_source(src)
            else:
                match_srcs[0].add_name(src.name)

            self._add_source_alias(src.name.replace(' ', '').lower(),
                                   match_srcs[0])
            return
        elif len(match_srcs) > 2:
            raise Exception('Multiple sources with name %s' % name)

        self._add_source_alias(src.name, src)

        for name in src.names:
            self._add_source_alias(name.replace(' ', '').lower(), src)

        if isinstance(src, Source):
            self._srcs.append(src)
        else:
            self._diffuse_srcs.append(src)

        if build_index:
            self._build_src_index()

    def match_source(self, src):
        """Look for source or sources in the model that match the
        given source.  Sources are matched by name and any association
        columns defined in the assoc_xmatch_columns parameter.
        """

        srcs = []

        names = [src.name]
        for col in self.config['assoc_xmatch_columns']:
            if col in src.assoc and src.assoc[col]:
                names += [src.assoc[col]]

        for name in names:
            name = name.replace(' ', '').lower()
            if name not in self._src_dict:
                continue
            srcs += [s for s in self._src_dict[name] if s not in srcs]

        return srcs

    def load(self, **kwargs):
        """Load both point source and diffuse components."""

        coordsys = kwargs.get('coordsys', 'CEL')
        extdir = kwargs.get('extdir', self.config['extdir'])
        srcname = kwargs.get('srcname', None)

        self.clear()
        self.load_diffuse_srcs()

        for c in self.config['catalogs']:

            if isinstance(c, catalog.Catalog):
                self.load_existing_catalog(c)
                continue

            extname = os.path.splitext(c)[1]
            if extname != '.xml':
                self.load_fits_catalog(c, extdir=extdir, coordsys=coordsys,
                                       srcname=srcname)
            elif extname == '.xml':
                self.load_xml(c, extdir=extdir, coordsys=coordsys)
            else:
                raise Exception('Unrecognized catalog file extension: %s' % c)

        for c in self.config['sources']:

            if 'name' not in c:
                raise Exception(
                    'No name field in source dictionary:\n ' + str(c))

            self.create_source(c['name'], c, build_index=False)

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

    @staticmethod
    def create_from_source(name, config, **kwargs):
        """Create an ROI centered on the given source."""

        coordsys = kwargs.pop('coordsys', 'CEL')

        roi = ROIModel(config, src_radius=None, src_roiwidth=None,
                       srcname=name, **kwargs)
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
        else:
            raise Exception('No source matching name: ' + name)

    def get_nearby_sources(self, name, distance, min_dist=None,
                           square=False):

        src = self.get_source_by_name(name)
        return self.get_sources_by_position(src.skydir,
                                            distance, min_dist,
                                            square)

    def get_sources(self, skydir=None, distance=None, cuts=None,
                    minmax_ts=None, minmax_npred=None,
                    exclude=None, square=False, coordsys='CEL'):
        """Retrieve list of source objects satisfying the following
        selections:

        * Angular separation from ``skydir`` or ROI center (if
             ``skydir`` is None) less than ``distance``.           

        * Cuts on source properties defined in ``cuts`` list.

        * TS and Npred in range specified by ``minmax_ts`` and ``minmax_npred``.

        Sources can be excluded from the selection by adding their
        name to the ``exclude`` list.

        Returns
        -------
        srcs : list
            List of source objects.
        """

        if skydir is None:
            skydir = self.skydir

        if exclude is None:
            exclude = []

        rsrc, srcs = self.get_sources_by_position(skydir,
                                                  distance,
                                                  square=square,
                                                  coordsys=coordsys)

        o = []
        for s in srcs + self.diffuse_sources:

            if s.name in exclude:
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
        # EAC split this function to make it easier to load an existing catalog
        cat = catalog.Catalog.create(name)
        self.load_existing_catalog(cat, **kwargs)

    def load_existing_catalog(self, cat, **kwargs):
        """Load sources from an existing catalog object.

        Parameters
        ----------
        cat : `~fermipy.catalog.Catalog`
            Catalog object.

        """
        coordsys = kwargs.get('coordsys', 'CEL')
        extdir = kwargs.get('extdir', self.config['extdir'])
        srcname = kwargs.get('srcname', None)

        m0 = get_skydir_distance_mask(cat.skydir, self.skydir,
                                      self.config['src_radius'])
        m1 = get_skydir_distance_mask(cat.skydir, self.skydir,
                                      self.config['src_radius_roi'],
                                      square=True, coordsys=coordsys)
        m = (m0 & m1)
        if srcname is not None:
            m &= utils.find_rows_by_string(cat.table, [srcname],
                                           self.src_name_cols)

        offset = self.skydir.separation(cat.skydir).deg
        offset_cel = wcs_utils.sky_to_offset(self.skydir,
                                             cat.radec[:, 0], cat.radec[:, 1],
                                             'CEL')
        offset_gal = wcs_utils.sky_to_offset(self.skydir,
                                             cat.glonlat[
                                                 :, 0], cat.glonlat[:, 1],
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

                src_dict['Spatial_Filename'] = utils.resolve_file_path(
                    row['Spatial_Filename'],
                    search_dirs=search_dirs)

            else:
                src_dict['SourceType'] = 'PointSource'
                src_dict['SpatialType'] = 'SkyDirFunction'
                src_dict['SpatialModel'] = 'PointSource'

            src_dict['spectral_pars'] = spectral_pars_from_catalog(
                catalog_dict)
            src = Source(src_dict['Source_Name'], src_dict, radec=radec)
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

    def create_source_table(self):

        cols_dict = collections.OrderedDict()
        cols_dict['source_name'] = dict(dtype='S48', format='%s')
        cols_dict['spectrum_type'] = dict(dtype='S48', format='%s')
        cols_dict['spatialModel_type'] = dict(dtype='S48', format='%s')
        cols_dict['spectrum_file'] = dict(dtype='S256', format='%s')
        cols_dict['spatialModel_file'] = dict(dtype='S256', format='%s')

        cols = [Column(name=k, **v) for k, v in cols_dict.items()]
        tab = Table(cols)

        row_dict = {}
        for s in self.sources:
            row_dict['source_name'] = s.name
            row_dict['spectrum_type'] = s['SpectrumType']
            row_dict['spatialModel_type'] = s['SpatialType']
            row_dict['spectrum_file'] = s['Spectrum_Filename']
            row_dict['spatialModel_file'] = s['Spatial_Filename']
            tab.add_row([row_dict[k] for k in tab.columns])

        return tab

    def create_param_table(self):

        cols_dict = collections.OrderedDict()
        cols_dict['source_name'] = dict(dtype='S48', format='%s')
        cols_dict['name'] = dict(dtype='S48', format='%s')
        cols_dict['group'] = dict(dtype='S48', format='%s')
        cols_dict['type'] = dict(dtype='S48', format='%s')
        cols_dict['value'] = dict(dtype='f8', format='%.3f')
        cols_dict['error'] = dict(dtype='f8', format='%.3f')
        cols_dict['scale'] = dict(dtype='f8', format='%.3f')
        cols_dict['min'] = dict(dtype='f8', format='%.3f')
        cols_dict['max'] = dict(dtype='f8', format='%.3f')
        cols_dict['free'] = dict(dtype='bool')

        cols = [Column(name=k, **v) for k, v in cols_dict.items()]
        tab = Table(cols)

        row_dict = {}
        for s in self.sources:
            row_dict['source_name'] = s.name

            row_dict['type'] = s['SpectrumType']
            row_dict['group'] = 'spectrum'
            for k, v in s.spectral_pars.items():
                row_dict['name'] = k
                row_dict.update(v)
                tab.add_row([row_dict[k] for k in tab.columns])

            row_dict['type'] = s['SpatialType']
            row_dict['group'] = 'spatialModel'
            for k, v in s.spatial_pars.items():
                row_dict['name'] = k
                row_dict.update(v)
                tab.add_row([row_dict[k] for k in tab.columns])

        return tab

    def create_table(self, names=None):
        """Create an astropy Table object with the contents of the ROI model.
        """

        scan_shape = (1,)
        for src in self._srcs:
            scan_shape = max(scan_shape, src['dloglike_scan'].shape)

        tab = create_source_table(scan_shape)
        for s in self._srcs:
            if names is not None and s.name not in names:
                continue
            s.add_to_table(tab)
            
        return tab

    def write_fits(self, fitsfile):
        """Write the ROI model to a FITS file."""

        tab = self.create_table()
        hdu_data = fits.table_to_hdu(tab)
        hdus = [fits.PrimaryHDU(), hdu_data]
        fits_utils.write_hdus(hdus, fitsfile)
