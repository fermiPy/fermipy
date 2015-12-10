import os
import copy
import re
import collections
import numpy as np
import defaults 
from fermipy.utils import (
    create_xml_element, merge_dict, sky_to_offset, prettify_xml,
    get_target_skydir, load_xml_elements, fits_recarray_to_dict
)
import fermipy
import fermipy.config
import fermipy.utils as utils
from fermipy.logger import Logger
from fermipy.logger import logLevel as ll


import xml.etree.cElementTree as ElementTree
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.io.fits as pyfits
from astropy.table import Table, Column

def xyz_to_lonlat(*args):

    if len(args) == 1:
        x, y, z = args[0][0],args[0][1],args[0][2]
    else:
        x, y, z = args[0], args[1], args[2]
        
    lat = np.pi/2. - np.arctan2(np.sqrt(x**2+y**2),z)
    lon = np.arctan2(y,x)
    return lon,lat
    
def lonlat_to_xyz(lon,lat):
    
    phi = lon
    theta = np.pi/2.-lat
    return np.array([np.sin(theta)*np.cos(phi),
                     np.sin(theta)*np.sin(phi),
                     np.cos(theta)])

def project(lon0,lat0,lon1,lat1):
    """This function performs a stereographic projection on the unit
    vector (lon1,lat1) with the pole defined at the reference unit
    vector (lon0,lat0)."""

    costh = np.cos(np.pi/2.-lat0)
    cosphi = np.cos(lon0)

    sinth = np.sin(np.pi/2.-lat0)
    sinphi = np.sin(lon0)

    xyz = lonlat_to_xyz(lon1,lat1)
    x1 = xyz[0]; y1 = xyz[1]; z1 = xyz[2]
    
    x1p = x1*costh*cosphi + y1*costh*sinphi - z1*sinth
    y1p = -x1*sinphi + y1*cosphi
    z1p = x1*sinth*cosphi + y1*sinth*sinphi + z1*costh
    
    r = np.arctan2(np.sqrt(x1p**2+y1p**2),z1p)
    phi = np.arctan2(y1p,x1p)

    return r*np.cos(phi), r*np.sin(phi)


def scale_parameter(p):
    
    if isinstance(p,str): p = float(p)
    
    if p > 0:    
        scale = 10**-np.round(np.log10(1./p))
        return p/scale, scale
    else:
        return p, 1.0

def resolve_file_path(path,filename):

    if filename is None: return None
    elif os.path.isfile(filename): return filename    
    elif os.path.isfile(os.path.join(path,filename)):
        return os.path.join(path,filename)
    else:
        return filename

spectrum_type_pars = {
    'PowerLaw' : ['Prefactor','Index','Scale'],
    'PowerLaw2' : ['Integral','Index','LowerLimit','UpperLimit'],
    'ConstantValue' : ['Value'],
    'BrokenPowerLaw' : ['Prefactor','Index1','Index2'],    
    'LogParabola' : ['norm','alpha','beta','Eb'],
    'PLSuperExpCutoff' : ['Prefactor','Index1','Index2','Cutoff','Scale'],
    'ExpCutoff' : ['Prefactor','Index1','Cutoff'],
    'FileFunction' : ['Normalization'],
    }
    
    
default_par_dict = {
    'Prefactor' :
        {'name' : 'Prefactor', 'value' : 1.0, 'scale' : None, 'min' : 0.01, 'max' : 100.0, 'free' : '0'},
    'norm' :
        {'name' : 'norm', 'value' : 1.0, 'scale' : None, 'min' : 0.01, 'max' : 100.0, 'free' : '0'},
    'Scale' :
        {'name' : 'Scale', 'value' : 1000.0, 'scale' : 1.0, 'min' : 1.0, 'max' : 1.0, 'free' : '0'},
    'Eb' :
        {'name' : 'Eb', 'value' : 1.0, 'scale' : None, 'min' : 0.01, 'max' : 10.0, 'free' : '0'},
    'Cutoff' :
        {'name' : 'Cutoff', 'value' : 1.0, 'scale' : None, 'min' : 0.01, 'max' : 10.0, 'free' : '0'},
    'Index' :
        {'name' : 'Index', 'value' : 2.0, 'scale' : -1.0, 'min' : 0.0, 'max' : 5.0, 'free' : '0'},
    'alpha' :
        {'name' : 'alpha', 'value' : 0.0, 'scale' : 1.0, 'min' : -5.0, 'max' : 5.0, 'free' : '0'},
    'beta' :
        {'name' : 'beta', 'value' : 0.0, 'scale' : 1.0, 'min' : -10.0, 'max' : 10.0, 'free' : '0'},
    'Index1' :
        {'name' : 'Index1', 'value' : 2.0, 'scale' : -1.0, 'min' : 0.0, 'max' : 5.0, 'free' : '0'},
    'Index2' :
        {'name' : 'Index2', 'value' : 1.0, 'scale' : 1.0, 'min' : 0.0, 'max' : 2.0, 'free' : '0'},
    'LowerLimit' :
        {'name' : 'LowerLimit', 'value' : 100.0, 'scale' : 1.0, 'min' : 20.0, 'max' : 1000000., 'free' : '0'},
    'UpperLimit' :
        {'name' : 'UpperLimit', 'value' : 100000.0, 'scale' : 1.0, 'min' : 20.0, 'max' : 1000000., 'free' : '0'},
    }
    
catalog_alias = {
    '3FGL' : {'file' : 'gll_psc_v16.fit',
              'extdir' : os.path.join('$(FERMIPY_ROOT)','catalogs','Extended_archive_v15'),
              'src_hduname' : 'LAT_Point_Source_Catalog',
              'extsrc_hduname' : 'ExtendedSources' },
    '2FGL' : {'file' : 'gll_psc_v08.fit',
              'extdir' : os.path.join('$(FERMIPY_ROOT)','catalogs','Extended_archive_v07'),
              'src_hduname' : 'LAT_Point_Source_Catalog',
              'extsrc_hduname' : 'ExtendedSources' },
    '2FHL' : {'file' : 'gll_psch_v08.fit',
              'extdir' : os.path.join('$(FERMIPY_ROOT)','catalogs','Extended_archive_v15'),
              'src_hduname' : '2FHL Source Catalog',
              'extsrc_hduname' : 'Extended Sources' },
    }

class PowerLaw(object):

    def __init__(self,phi0,x0,index):
        self._params = np.array([phi0,x0,index])

    @property
    def params(self):
        return self._params

    def dfde(self,x):
        return PowerLaw.eval_dfde(x,*self.params)

    @staticmethod
    def eval_dfde(x,phi0,x0,index):        
        return phi0*(x/x0)**index
    
    @staticmethod
    def eval_flux(phi0,x0,index,xmin,xmax):

        if np.allclose(index,-1.0):
            return phi0*x0**(-index)*(np.log(xmax)-np.log(xmin))

        y0 = x0*phi0*(xmin/x0)**(index+1)/(index+1)
        y1 = x0*phi0*(xmax/x0)**(index+1)/(index+1)
        v = y1 - y0

        return y1-y0

    @staticmethod
    def eval_norm(x0,index,xmin,xmax,flux):

        return flux/PowerLaw.eval_flux(1.0,x0,index,xmin,xmax)

def add_columns(t0,t1):
    """Add columns of table t1 to table t0."""
    
    for colname in t1.colnames:
        col = t1.columns[colname]
        if colname in t0.columns: continue
        new_col = Column(name=col.name,length=len(t0),dtype=col.dtype)#,shape=col.shape)
        t0.add_column(new_col)

def join_tables(t0,t1,key0,key1):
    
    v0, v1 = t0[key0],t1[key1]
    v0 = np.core.defchararray.strip(v0)
    v1 = np.core.defchararray.strip(v1)
    add_columns(t0,t1)

    # Get mask of elements in t0 that are shared with t0
    m0 = np.in1d(v0,v1)
    idx1 = np.searchsorted(v1,v0)[m0]    

    for colname in t1.colnames:
        if colname == 'Source_Name': continue
        t0[colname][m0] = t1[colname][idx1]

#    for i in np.nonzero(m0)[0]:
#        print i, t0[i]['Source_Name'], t0[i]['Spatial_Filename'], t0[i]['RAJ2000'], t0[i]['ASSOC1']
        
def strip_columns(t):    
    for colname in t.colnames:
        if not t[colname].dtype.type is np.string_: continue
        t[colname] = np.core.defchararray.strip(t[colname])

def row_to_dict(row):

    o = {}
    for colname in row.colnames:

        if isinstance(row[colname],np.string_):
            o[colname] = str(row[colname])
        else:
            o[colname] = row[colname]

    return o

class Catalog(object):

    def __init__(self,table,extdir=''):
        self._table = table
        self._extdir = extdir
        self._src_skydir = SkyCoord(ra=self.table['RAJ2000']*u.deg,
                                    dec=self.table['DEJ2000']*u.deg)
        self._radec = np.vstack((self._src_skydir.ra.deg,
                                 self._src_skydir.dec.deg)).T
        self._glonlat = np.vstack((self._src_skydir.galactic.l.deg,
                                    self._src_skydir.galactic.b.deg)).T

        m = self.table['Spatial_Filename'] != ''
        self.table['extended'] = False
        self.table['extended'][m] = True
        self.table['extdir'] = extdir

    @property
    def table(self):
        return self._table

    @property
    def skydir(self):
        return self._src_skydir

    @property
    def radec(self):
        return self._radec

    @property
    def glonlat(self):
        return self._glonlat

    @staticmethod
    def create(name):

        extname = os.path.splitext(name)[1]
        if extname == '.fits' or extname == '.fit':
            fitsfile = name
            if not os.path.isfile(fitsfile):
                fitsfile = os.path.join(fermipy.PACKAGE_ROOT,'catalogs',fitsfile)
            return Catalog3FGL(fitsfile)
        elif name == '3FGL': 
            return Catalog3FGL()
        elif name == '2FHL': 
            return Catalog2FHL()
        else:
            raise Exception('Unrecognized catalog type.')

class Catalog2FHL(Catalog):

    def __init__(self,fitsfile=None,extdir=''):

        if fitsfile is None:
            fitsfile = os.path.join(fermipy.PACKAGE_ROOT,'catalogs','gll_psch_v08.fit')
        
        hdulist = pyfits.open(fitsfile)
        table = Table(hdulist['2FHL Source Catalog'].data)
        table_extsrc = Table(hdulist['Extended Sources'].data)

        strip_columns(table)
        strip_columns(table_extsrc)

        join_tables(table,table_extsrc,'Source_Name','Source_Name')

        super(Catalog2FHL,self).__init__(table)

        self._table['Flux_Density'] = PowerLaw.eval_norm(50E3,-self.table['Spectral_Index'],
                                                         50E3,2000E3,self.table['Flux50'])
        self._table['Pivot_Energy'] = 50E3
        self._table['SpectrumType'] = 'PowerLaw'

        

class Catalog3FGL(Catalog):

    def __init__(self,fitsfile=None,extdir=''):
        
        if extdir is None:
            extdir = os.path.join('$(FERMIPY_ROOT)','catalogs','Extended_archive_v15')

        if fitsfile is None:
            fitsfile = os.path.join(fermipy.PACKAGE_ROOT,'catalogs','gll_psc_v16.fit')
        
        hdulist = pyfits.open(fitsfile)
        table = Table(hdulist['LAT_Point_Source_Catalog'].data)
        table_extsrc = Table(hdulist['ExtendedSources'].data)

        strip_columns(table)
        strip_columns(table_extsrc)

        self._table_extsrc = table_extsrc

        join_tables(table,table_extsrc,'Extended_Source_Name','Source_Name')

        super(Catalog3FGL,self).__init__(table,extdir)

        m = self.table['SpectrumType'] == 'PLExpCutoff'
        self.table['SpectrumType'][m] = 'PLSuperExpCutoff'

        self.table['TS_value'] = 0.0
        self.table['TS'] = 0.0

        ts_keys = ['Sqrt_TS30_100','Sqrt_TS100_300',
                   'Sqrt_TS300_1000','Sqrt_TS1000_3000',
                   'Sqrt_TS3000_10000','Sqrt_TS10000_100000']

        for k in ts_keys:
            m = np.isfinite(self.table[k])
            self._table['TS_value'][m] += self.table[k][m]**2
            self._table['TS'][m] += self.table[k][m]**2
        
#        if not os.path.isfile(src_dict['Spatial_Filename']) and extdir:
#            src_dict['Spatial_Filename'] = os.path.join(extdir,'Templates',
#                                                        src_dict['Spatial_Filename'])

#        m = self.table['extended']
#       src_dict['Spatial_Filename'] = os.path.join(extdir,'Templates',
#                                                    src_dict['Spatial_Filename'])

def make_parameter_dict(pdict,fixed_par=False):

    o = copy.deepcopy(pdict)
    
    if not 'scale' in o or o['scale'] is None:
        value, scale = scale_parameter(o['value'])        
        o['value'] = value
        o['scale'] = scale

    if fixed_par:
        o['min'] = o['value']
        o['max'] = o['value']
        
    if float(o['min'])>float(o['value']):
        o['min'] = o['value']

    if float(o['max'])<float(o['value']):
        o['max'] = o['value']

    for k,v in o.items():
        o[k] = str(v)
        
    return o
    
    
def get_skydir_distance_mask(src_skydir,skydir,dist,min_dist=None,
                             square=False,coordsys='CEL'):
    """Retrieve sources within a certain angular distance of an
    (ra,dec) coordinate.  This function supports two types of
    geometric selections: circular (square=False) and square
    (square=True).  The circular selection finds all sources with a given
    angular distance of the target position.  The square selection
    finds sources within an ROI-like region of size R x R where R
    = 2 x dist.

    Parameters
    ----------
    
    src_skydir : `~astropy.coord.SkyCoord` 
    Array of sky directions.

    skydir : `~astropy.coord.SkyCoord` 
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
        raise Exception('Unrecognized coordinate system: %s'%coordsys)

    msk = (dtheta < np.radians(dist))
    if min_dist is not None: msk &= (dtheta > np.radians(min_dist))
    return msk

def get_linear_dist(skydir,lon,lat,coordsys='CEL'):

    xy = sky_to_offset(skydir,np.degrees(lon),np.degrees(lat),
                       coordsys=coordsys)

    x = np.radians(xy[:,0])
    y = np.radians(xy[:,1])    
    delta = np.array([np.abs(x),np.abs(y)])
    dtheta = np.max(delta,axis=0)
    return dtheta
    
def get_dist_to_edge(skydir,lon,lat,width,coordsys='CEL'):

    xy = sky_to_offset(skydir,np.degrees(lon),np.degrees(lat),
                       coordsys=coordsys)

    x = np.radians(xy[:,0])
    y = np.radians(xy[:,1])
    
    delta_edge = np.array([np.abs(x) - width,np.abs(y) - width])
    dtheta = np.max(delta_edge,axis=0)
    return dtheta

def create_model_name(src):
    if src['SpectrumType'] == 'PowerLaw':
        return 'powerlaw_%04.2f'%src['Index']

class Model(object):
    """Base class for source objects."""
    
    def __init__(self,name,data=None,
                 spectral_pars=None,
                 spatial_pars=None):

        self._data = { 'SpatialModel' : None, 
                       'SpatialWidth' : None, 
                       'SpatialType' : None,
                       'SourceType' : None,
                       'SpectrumType' : None,
                       'RAJ2000' : 0.0,
                       'DEJ2000' : 0.0,
                       'ra'   : 0.0,
                       'dec'  : 0.0,
                       'glon' : 0.0,
                       'glat' : 0.0,
                       'offset_ra' : [0.0,0.0],
                       'offset_dec' : [0.0,0.0],
                       'offset_glon' : [0.0,0.0],
                       'offset_glat' : [0.0,0.0],
                       'offset' : 0.0,
                       'ts' : np.nan,
                       'Npred' : 0.0,
                       'params' : {}
                       }
        if data is not None:
            self._data.update(data)

        self._data['name'] = name

        self._data.setdefault('spectral_pars',{})
        self._data.setdefault('spatial_pars',{})
        self._data.setdefault('catalog',{})

        if spectral_pars is not None:
            self._data['spectral_pars'] = spectral_pars
            
        if spatial_pars is not None:
            self._data['spatial_pars'] = spatial_pars

        self._names = [name]
        self._names_dict = {}
        catalog = self._data['catalog']

        for k in ROIModel.src_name_cols:

            if not k in catalog: continue            
            name = catalog[k].strip()
            if name != '' and not name in self._names:
                self._names.append(name)

            self._names_dict[k] = name
        
        
    def __contains__(self,key):
        return key in self._data

    def __getitem__(self,key):
        return self._data[key]

    def __setitem__(self,key,value):
        self._data[key]=value

    def __eq__(self, other): 
        return self.name == other.name

    def __str__(self):

        data = copy.deepcopy(self.data)
        data['names'] = self.names

        output = []
        output += ['{:15s}:'.format('Name') + ' {name:s}']
        output += ['{:15s}:'.format('TS') + ' {ts:.2f}']
        output += ['{:15s}:'.format('Npred') + ' {Npred:.2f}']
        output += ['{:15s}:'.format('SpatialModel') + ' {SpatialModel:s}']
        output += ['{:15s}:'.format('SpectrumType') + ' {SpectrumType:s}']
        output += ['Spectral Parameters']

        for k,v in self['params'].items():
            if isinstance(v,np.ndarray):
                output += ['{:15s}: {:10.4g} +/- {:10.4g}'.format(k,v[0],v[1])]

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
    def name(self):
        return self._data['name']

    @property
    def names(self):
        return self._names

    def set_name(self,name,names=None):
        self._data['name'] = name
        if names is None:
            self._names = [name]
        else:
            self._names = names
    
    def add_name(self,name):
        self._names.append(name)
    
    def update_data(self,d):
        self._data = merge_dict(self._data,d,add_new_keys=True)

    def update(self,m):

        if 'SpectrumType' in m and self['SpectrumType'] != m['SpectrumType']:
            self.spectral_pars = {}
        
        self._data = merge_dict(self._data,m._data,add_new_keys=True)
        self._name = m.name
        self._names = list(set(self._names + m.names))        
        self._data['spectral_pars'] = merge_dict(self.spectral_pars,
                                         m.spectral_pars,add_new_keys=True)
        self._data['spatial_pars'] = merge_dict(self.spatial_pars,
                                        m.spatial_pars,add_new_keys=True)
        
class IsoSource(Model):

    def __init__(self,name,filefunction,spectral_pars=None,spatial_pars=None):
        super(IsoSource,self).__init__(name,None,spectral_pars,spatial_pars)
        
        self._filefunction = filefunction
        self._data['SpectrumType'] = 'FileFunction'
        self._data['SpatialType'] = 'ConstantValue'
        self._data['SpatialModel'] = 'DiffuseSource'
        self._data['SourceType'] = 'DiffuseSource'

        if not self.spectral_pars:
            self['spectral_pars'] = {
                'Normalization' : {'name' : 'Normalization', 'scale' : '1.0',
                                   'value' : '1.0',
                                   'min' : '0.001', 'max' : '1000.0',
                                   'free' : '0' } }

        if not self.spatial_pars:            
            self['spatial_pars'] = {
                'Value' : {'name' : 'Value', 'scale' : '1',
                           'value' : '1', 'min' : '0', 'max' : '10',
                           'free' : '0' } }

        
        
    @property
    def filefunction(self):
        return self._filefunction

    @property
    def diffuse(self):
        return True

    def write_xml(self,root):
        
        source_element = create_xml_element(root,'source',
                                            dict(name=self.name,
                                                 type='DiffuseSource'))

        filename = re.sub(r'\$([a-zA-Z\_]+)',r'$(\1)',self.filefunction)        
        spec_el = create_xml_element(source_element,'spectrum',
                                     dict(file=filename,
                                          type='FileFunction',
                                          ctype='-1'))
                        
        spat_el = create_xml_element(source_element,'spatialModel',
                                     dict(type='ConstantValue'))


        for k,v in self.spectral_pars.items():                
            create_xml_element(spec_el,'parameter',v)

        for k,v in self.spatial_pars.items():                
            create_xml_element(spat_el,'parameter',v)
        
class MapCubeSource(Model):

    def __init__(self,name,mapcube,spectral_pars=None,spatial_pars=None):
        super(MapCubeSource,self).__init__(name,None,spectral_pars,spatial_pars)

        self._mapcube = mapcube
        self._data['SpectrumType'] = 'PowerLaw'
        self._data['SpatialType'] = 'MapCubeFunction'
        self._data['SpatialModel'] = 'DiffuseSource'
        self._data['SourceType'] = 'DiffuseSource'
        
        if not self.spectral_pars:
            self['spectral_pars'] = {
                'Prefactor' : {'name' : 'Prefactor', 'scale' : '1',
                               'value' : '1.0', 'min' : '0.1', 'max' : '10.0',
                               'free' : '0' },
                'Index' : {'name' : 'Index', 'scale' : '-1',
                           'value' : '0.0', 'min' : '-1.0', 'max' : '1.0',
                           'free' : '0' },
                'Scale' : {'name' : 'Scale', 'scale' : '1',
                           'value' : '1000.0',
                           'min' : '1000.0', 'max' : '1000.0',
                           'free' : '0' },
                }

        if not self.spatial_pars:            
            self['spatial_pars'] = {
                'Normalization' :
                    {'name' : 'Normalization', 'scale' : '1',
                     'value' : '1', 'min' : '0', 'max' : '10',
                     'free' : '0' } }
        
    @property
    def mapcube(self):
        return self._mapcube

    @property
    def diffuse(self):
        return True

    def write_xml(self,root):
        
        source_element = create_xml_element(root,'source',
                                            dict(name=self.name,
                                                 type='DiffuseSource'))

        spec_el = create_xml_element(source_element,'spectrum',
                                     dict(type='PowerLaw'))

        filename = re.sub(r'\$([a-zA-Z\_]+)',r'$(\1)',self.mapcube)        
        spat_el = create_xml_element(source_element,'spatialModel',
                                     dict(type='MapCubeFunction',
                                          file=filename))


        for k,v in self.spectral_pars.items():                
            create_xml_element(spec_el,'parameter',v)

        for k,v in self.spatial_pars.items():                
            create_xml_element(spat_el,'parameter',v)
        
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

    def __init__(self,name,data=None,
                 radec=None,
                 spectral_pars=None,
                 spatial_pars=None):
        super(Source,self).__init__(name,data,spectral_pars,spatial_pars)
                    
        self._radec = radec
        catalog = self.data.get('catalog',{})

        if self._radec is None and 'RAJ2000' in catalog and 'DEJ2000' in catalog:
            self._radec = [catalog['RAJ2000'],catalog['DEJ2000']]
        elif self._radec is None and 'ra' in self.data and 'dec' in self.data:
            self._radec = [self.data['ra'],self.data['dec']]
        elif self._radec is None:
            raise Exception('Failed to infer RADEC for source: %s'%name)
                
        self['RAJ2000'] = self._radec[0]
        self['DEJ2000'] = self._radec[1]
        self['ra'] = self._radec[0]
        self['dec'] = self._radec[1]        
        glonlat = utils.eq2gal(self._radec[0],self._radec[1])
        self['glon'], self['glat'] = glonlat[0][0], glonlat[1][0]

        if self['SpatialModel'] is None:
            self._data['SpatialModel'] = self['SpatialType']
                    
        self.set_spatial_model(self.data['SpatialModel'],
                               self.data['SpatialWidth'])

        if not self.spectral_pars:
            self._update_spectral_pars()

        if not self.spatial_pars:
            self._update_spatial_pars()

    def __str__(self):

        data = copy.deepcopy(self.data)
        data['names'] = self.names

        output = []
        output += ['{:15s}:'.format('Name') + ' {name:s}']
        output += ['{:15s}:'.format('Associations') +   ' {names:s}']
        output += ['{:15s}:'.format('RA/DEC') + ' {ra:10.3f}/{dec:10.3f}']
        output += ['{:15s}:'.format('GLON/GLAT') + ' {glon:10.3f}/{glat:10.3f}']
        output += ['{:15s}:'.format('TS') + ' {ts:.2f}']
        output += ['{:15s}:'.format('Npred') + ' {Npred:.2f}']
        output += ['{:15s}:'.format('SpatialModel') + ' {SpatialModel:s}']
        output += ['{:15s}:'.format('SpectrumType') + ' {SpectrumType:s}']
        output += ['Spectral Parameters']

        for k,v in self['params'].items():
            if isinstance(v,np.ndarray):
                output += ['{:15s}: {:10.4g} +/- {:10.4g}'.format(k,v[0],v[1])]

        return '\n'.join(output).format(**data)

    def _update_spatial_pars(self):

        if self['SpatialModel'] == 'SpatialMap':
            self._data['spatial_pars'] = {
                'Prefactor' : {'name' : 'Prefactor', 'value' : '1',
                               'free' : '0', 'min' : '0.001', 'max' : '1000',
                               'scale' : '1.0'}
                }        
        else:
            self._data['spatial_pars'] = {
                'RA' : {'name' : 'RA',  'value' : str(self['RAJ2000']),
                        'free' : '0',
                        'min' : '-360.0','max' : '360.0','scale' : '1.0'},
                'DEC' : {'name' : 'DEC',  'value' : str(self['DEJ2000']),
                         'free' : '0',
                         'min' : '-90.0','max' : '90.0','scale' : '1.0'}
                }
            
            
    def _update_spectral_pars(self):

        self._data['spectral_pars'] = {}        
        sp = self['spectral_pars']

        catalog = self.data.get('catalog',{})
        
        if self['SpectrumType'] == 'PowerLaw':

            sp['Prefactor'] = copy.copy(default_par_dict['Prefactor'])
            sp['Scale'] = copy.copy(default_par_dict['Scale'])
            sp['Index'] = copy.copy(default_par_dict['Index'])
            
            sp['Prefactor']['value'] = catalog['Flux_Density']
            sp['Scale']['value'] = catalog['Pivot_Energy']
            sp['Index']['value'] = catalog['Spectral_Index']                           
            sp['Index']['max'] = max(5.0,sp['Index']['value']+1.0)
            sp['Index']['min'] = min(0.0,sp['Index']['value']-1.0)
                        
            sp['Prefactor'] = make_parameter_dict(sp['Prefactor'])
            sp['Scale'] = make_parameter_dict(sp['Scale'],True)
            sp['Index'] = make_parameter_dict(sp['Index'])

        elif self['SpectrumType'] == 'LogParabola':

            sp['norm'] = copy.copy(default_par_dict['norm'])
            sp['Eb'] = copy.copy(default_par_dict['Eb'])
            sp['alpha'] = copy.copy(default_par_dict['alpha'])
            sp['beta'] = copy.copy(default_par_dict['beta'])

            sp['norm']['value'] = catalog['Flux_Density']
            sp['Eb']['value'] = catalog['Pivot_Energy']
            sp['alpha']['value'] = catalog['Spectral_Index']
            sp['beta']['value'] = catalog['beta']
                        
            sp['norm'] = make_parameter_dict(sp['norm'])
            sp['Eb'] = make_parameter_dict(sp['Eb'],True)
            sp['alpha'] = make_parameter_dict(sp['alpha'])
            sp['beta'] = make_parameter_dict(sp['beta'])            
        
        elif self['SpectrumType'] == 'PLSuperExpCutoff':

            flux_density = catalog['Flux_Density']
            flux_density *= np.exp((catalog['Pivot_Energy']/catalog['Cutoff'])**catalog['Exp_Index'])
            
            sp['Prefactor'] = copy.copy(default_par_dict['Prefactor'])
            sp['Index1'] = copy.copy(default_par_dict['Index1'])
            sp['Index2'] = copy.copy(default_par_dict['Index2'])
            sp['Scale'] = copy.copy(default_par_dict['Scale'])
            sp['Cutoff'] = copy.copy(default_par_dict['Cutoff'])

            sp['Prefactor']['value'] = flux_density
            sp['Index1']['value'] = catalog['Spectral_Index']
            sp['Index2']['value'] = catalog['Exp_Index']
            sp['Scale']['value'] = catalog['Pivot_Energy']
            sp['Cutoff']['value'] = catalog['Cutoff']
            
            sp['Prefactor'] = make_parameter_dict(sp['Prefactor'])
            sp['Scale'] = make_parameter_dict(sp['Scale'],True)
            sp['Index1'] = make_parameter_dict(sp['Index1'])
            sp['Index2'] = make_parameter_dict(sp['Index2'])
            sp['Cutoff'] = make_parameter_dict(sp['Cutoff'])                

        else:
            import pprint
            pprint.pprint(self._data)            
            raise Exception('Unsupported spectral type:' + self['SpectrumType'])
        
    def set_position(self,skydir):

        if not isinstance(skydir,SkyCoord):
            skydir = SkyCoord(ra=skydir[0],dec=skydir[1],unit=u.deg)
        
        self._radec = np.array([skydir.ra.deg,skydir.dec.deg])
        self['RAJ2000'] = self._radec[0]
        self['DEJ2000'] = self._radec[1]
        self['ra'] = self._radec[0]
        self['dec'] = self._radec[1]

    def set_spatial_model(self,spatial_model,spatial_width=None):

        self._data['SpatialModel'] = spatial_model
        self._data['SpatialWidth'] = spatial_width
        
        if self['SpatialModel'] in ['PointSource','Gaussian','PSFSource']:
            self._extended = False
            self._data['SpatialType'] = 'SkyDirFunction'
            self._data['SourceType'] = 'PointSource'
        elif self['SpatialModel'] in ['GaussianSource','DiskSource','SpatialMap']:
            self._extended = True
            self._data['SpatialType'] = 'SpatialMap'
            self._data['SourceType'] = 'DiffuseSource'
        else:
            raise Exception('Unrecognized SpatialModel: ' + self['SpatialModel'])
        
        self._update_spatial_pars()
        
    def check_cuts(self,cuts):

        if isinstance(cuts,tuple): cuts = [cuts]
        
        for c in cuts:

            if not isinstance(c,tuple) or len(c) != 3:
                raise Exception('Wrong format for cuts tuple.')
            
            (pname,pmin,pmax) = c
            if not pname in self._data: return False
            if pmin is not None and self[pname] < pmin: return False
            if pmax is not None and self[pname] > pmax: return False

        return True
                               
    def separation(self,src):

        if isinstance(src,Source):
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
        return self._radec

    @property
    def skydir(self):
        """Return a SkyCoord representation of the source position."""
        return SkyCoord(self._radec[0]*u.deg,self._radec[1]*u.deg)
    
    @property
    def data(self):
        return self._data

    @staticmethod
    def create_from_dict(src_dict):
        """Create a source object from a python dictionary."""
        
        src_dict = copy.deepcopy(src_dict)
        spectrum_type = src_dict.get('SpectrumType','PowerLaw')
        
        default_src_dict = dict(name = None,
                                Source_Name = None,
                                SpatialModel = 'PointSource',
                                SpatialWidth = 0.5,
                                SpectrumType = 'PowerLaw',
                                ra = None,
                                dec = None,
                                glon = None,
                                glat = None,
                                spectral_pars = {})

        for p in spectrum_type_pars[spectrum_type]:
            default_src_dict['spectral_pars'][p] = copy.copy(default_par_dict[p])

        for k,v in default_par_dict.items():
            
            if not k in src_dict: continue
            src_dict.setdefault('spectral_pars',{})
            src_dict['spectral_pars'].setdefault(k,{})
            
            if not isinstance(src_dict[k],dict):
                src_dict['spectral_pars'][k] = {'name' : k, 'value' : src_dict.pop(k)}
            else:
                src_dict['spectral_pars'][k] = src_dict.pop(k)
            
        src_dict = merge_dict(default_src_dict,src_dict)        
        for k, v in src_dict['spectral_pars'].items():
            src_dict['spectral_pars'][k] = make_parameter_dict(v)
        
#        validate_config(src_dict,default_src_dict)

        if 'name' in src_dict:
            name = src_dict['name']
            src_dict['Source_Name'] = src_dict.pop('name')
        elif 'Source_Name' in src_dict:
            name = src_dict['Source_Name']
        else:
            raise Exception('Source name undefined.')
            
        skydir = get_target_skydir(src_dict)
        
        src_dict['RAJ2000'] = skydir.ra.deg
        src_dict['DEJ2000'] = skydir.dec.deg
        
        radec = np.array([skydir.ra.deg,skydir.dec.deg])
        return Source(name,src_dict,radec=radec,
                      spectral_pars=src_dict['spectral_pars'])
    
    @staticmethod
    def create_from_xml(root,extdir=None):
        """Create a Source object from an XML node."""

        spec = load_xml_elements(root,'spectrum')
        spat = load_xml_elements(root,'spatialModel')
        spectral_pars = load_xml_elements(root,'spectrum/parameter')
        spatial_pars = load_xml_elements(root,'spatialModel/parameter')

        src_type = root.attrib['type']
        spatial_type = spat['type']
        spectral_type = spec['type']
        
        xml_dict = copy.deepcopy(root.attrib) 
        src_dict = {'catalog' : xml_dict }

        src_dict['Source_Name'] = xml_dict['name']
        src_dict['SpectrumType'] = spec['type']
        src_dict['SpatialType'] = spatial_type
        src_dict['SourceType'] = src_type
        
        if src_type == 'PointSource':
            src_dict['SpatialModel'] = 'PointSource'
        elif spatial_type == 'SpatialMap':
            src_dict['SpatialModel'] = 'SpatialMap'

        if src_type =='PointSource' or spatial_type == 'SpatialMap':
        
            if 'file' in spat: 
                src_dict['Spatial_Filename'] = spat['file']
                if not os.path.isfile(src_dict['Spatial_Filename']) \
                        and extdir is not None:
                    src_dict['Spatial_Filename'] = \
                        os.path.join(extdir,'Templates',
                                     src_dict['Spatial_Filename'])
                        
            if 'RA' in src_dict:
                src_dict['RAJ2000'] = float(xml_dict['RA'])
                src_dict['DEJ2000'] = float(xml_dict['DEC'])
            elif 'RA' in spatial_pars:
                src_dict['RAJ2000'] = float(spatial_pars['RA']['value'])
                src_dict['DEJ2000'] = float(spatial_pars['DEC']['value'])
            else:
                hdu = pyfits.open(src_dict['Spatial_Filename'])
                src_dict['RAJ2000'] = float(hdu[0].header['CRVAL1'])
                src_dict['DEJ2000'] = float(hdu[0].header['CRVAL2'])

            radec = np.array([src_dict['RAJ2000'],src_dict['DEJ2000']])
                
            return Source(src_dict['Source_Name'],
                          src_dict,radec=radec,
                          spectral_pars=spectral_pars,
                          spatial_pars=spatial_pars)

        elif src_type == 'DiffuseSource' and spatial_type == 'ConstantValue':
            return IsoSource(src_dict['Source_Name'],spec['file'],
                             spectral_pars,spatial_pars)
        elif src_type == 'DiffuseSource' and spatial_type == 'MapCubeFunction':
            return MapCubeSource(src_dict['Source_Name'],spat['file'],
                                 spectral_pars,spatial_pars)
        else:
            raise Exception('Unrecognized type for source: %s'%src_dict['Source_Name'])
        
    def write_xml(self,root):
        """Write this source to an XML node."""

        if not self.extended:
            source_element = create_xml_element(root,'source',
                                                dict(name=self['Source_Name'],
                                                     type='PointSource'))
            
            spat_el = ElementTree.SubElement(source_element,'spatialModel')
            spat_el.set('type','SkyDirFunction')
             
        else:
            source_element = create_xml_element(root,'source',
                                                dict(name=self['Source_Name'],
                                                     type='DiffuseSource'))

            filename = self['Spatial_Filename']
            filename = re.sub(r'\$([a-zA-Z\_]+)',r'$(\1)',filename)
            
            spat_el = create_xml_element(source_element,'spatialModel',
                                         dict(map_based_integral='True',
                                              type='SpatialMap',
                                              file=filename))
                    
        for k,v in self.spatial_pars.items():                
            create_xml_element(spat_el,'parameter',v)

                
        el = ElementTree.SubElement(source_element,'spectrum')  
        
        stype = self['SpectrumType'].strip()            
        el.set('type',stype)
        
        for k,v in self.spectral_pars.items():                
            create_xml_element(el,'parameter',v)
    
class ROIModel(fermipy.config.Configurable):
    """This class is responsible for managing the ROI definition.
    Catalogs can be read from either FITS or XML files."""

    defaults = dict(defaults.model.items(),
                    logfile=(None,'',str),
                    fileio=defaults.fileio,
                    logging=defaults.logging)

    src_name_cols = ['Source_Name',
                     'ASSOC','ASSOC1','ASSOC2','ASSOC_GAM',
                     '1FHL_Name','2FGL_Name','3FGL_Name',
                     'ASSOC_GAM1','ASSOC_GAM2','ASSOC_TEV']

    def __init__(self,config=None,**kwargs):
        # Coordinate for ROI center (defaults to 0,0)
        self._skydir = kwargs.pop('skydir',SkyCoord(0.0,0.0,unit=u.deg)) 
        coordsys=kwargs.pop('coordsys','CEL')
        super(ROIModel,self).__init__(config,**kwargs)
        
        self.logger = Logger.get(self.__class__.__name__,
                                 self.config['logfile'],
                                 ll(self.config['logging']['verbosity']))
        
        if not os.path.isdir(self.config['extdir']):
            self._config['extdir'] = \
                os.path.join(fermipy.PACKAGE_ROOT,
                             'catalogs',self.config['extdir'])
        
        self._src_radius = self.config['src_radius']
        if self.config['src_roiwidth'] is not None:
            self._config['src_radius_roi'] = self.config['src_roiwidth']*0.5

        self._srcs = []
        self._diffuse_srcs = []
        self._src_dict = collections.defaultdict(set)
        self._src_radius = []

        self.load(coordsys=coordsys)

    def __getitem__(self,key):
        return self.get_source_by_name(key,True)

    def __iter__(self):
        return iter(self._srcs + self._diffuse_srcs)

    def __str__(self):

        o = ''
        o += '%-20s%-15s%-15s%8s%10s%12s\n'%('name','SpatialModel','SpectrumType','offset',
                                                   'ts','Npred')
        o += '-'*80 + '\n'
        
        for s in sorted(self.sources,key=lambda t:t['offset']):

            if s.diffuse: continue            
            o += '%-20.19s%-15.14s%-15.14s%8.3f%10.2f%12.1f\n'%(s['name'],s['SpatialModel'],
                                                                s['SpectrumType'],
                                                                s['offset'],s['ts'],s['Npred'])
        
        for s in sorted(self.sources,key=lambda t:t['offset']):

            if not s.diffuse: continue
            o += '%-20.19s%-15.14s%-15.14s%8s%10.2f%12.1f\n'%(s['name'],s['SpatialModel'],
                                                              s['SpectrumType'],
                                                              '-----',s['ts'],s['Npred'])

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
    
    def load_diffuse_srcs(self):

        self._load_diffuse_src('isodiff')
        self._load_diffuse_src('galdiff')
        self._load_diffuse_src('limbdiff')
        self._load_diffuse_src('diffuse')

    def _load_diffuse_src(self,name,src_type='FileFunction'):

        if 'FERMI_DIR' in os.environ and not 'FERMI_DIFFUSE_DIR' in os.environ:
            os.environ['FERMI_DIFFUSE_DIR'] = \
                os.path.expandvars('$FERMI_DIR/refdata/fermi/galdiffuse')
        
        srcs = []
        if self.config[name] is not None:
            srcs = self.config[name]

        for i, t in enumerate(srcs):

            if isinstance(t,str):
                src_dict = {'file' : t}
            elif isinstance(t,dict):
                src_dict = copy.deepcopy(t)
                
            if not os.path.isfile(os.path.expandvars(src_dict['file'])):
                src_dict['file'] = os.path.join('$FERMI_DIFFUSE_DIR',
                                                src_dict['file'])
                
            if not 'name' in src_dict:                
                if len(srcs) == 1:
                    src_dict['name'] = name
                else:
                    src_dict['name'] = name + '%02i'%i
                    
            if re.search(r'(\.txt$)',src_dict['file']):
                src_type = 'FileFunction'
            elif re.search(r'(\.fits$|\.fit$|\.fits.gz$|\.fit.gz$)',src_dict['file']):
                src_type = 'MapCubeFunction'
            else:
                raise Exception('Unrecognized file format for diffuse model: %s'%src_dict['file'])

            if src_type == 'FileFunction':                    
                src = IsoSource(src_dict['name'],src_dict['file'])
                altname = os.path.basename(src_dict['file'])            
                altname = re.sub(r'(\.txt$)','', altname)
            else:
                src = MapCubeSource(src_dict['name'],src_dict['file'])
                altname = os.path.basename(src_dict['file'])
                altname = re.sub(r'(\.fits$|\.fit$|\.fits.gz$|\.fit.gz$)',
                                 '', altname)    
                                
            src.add_name(altname)

            

            self.load_source(src,False)

    def create_source(self,src_dict,build_index=True):
        """Create a new source object from a source dictionary and
        load it in the ROI."""

        src = Source.create_from_dict(src_dict)
        src.set_spatial_model(src['SpatialModel'],src['SpatialWidth'])
        
        offset = self.skydir.separation(src.skydir).deg
        offset_cel = utils.sky_to_offset(self.skydir,
                                         src['ra'], src['dec'], 'CEL')

        offset_gal = utils.sky_to_offset(self.skydir,
                                         src['glon'], src['glat'], 'GAL') 

        src['offset'] = offset
        src['offset_ra'] = offset_cel[0,0]
        src['offset_dec'] = offset_cel[0,1]
        src['offset_glon'] = offset_gal[0,0]
        src['offset_glat'] = offset_gal[0,1]

        self.logger.debug('Creating source ' + src.name)
        self.logger.debug(src._data)

        self.load_source(src,build_index=build_index)
        return src

    def load_source_data(self,sources):

        # Sync source data
        for k,v in sources.items():
            if self.has_source(k):
                src = self.get_source_by_name(k,True)
                src.update_data(v)
            else:
                src = Source(k,data=v)
                self.load_source(src)

        # Prune sources not present in the sources dict
        for s in self.sources:
            if not s.name in sources.keys():
                self.delete_sources([s])

        self.build_src_index()
            
    def load_source(self,src,build_index=True):

        src = copy.deepcopy(src)        
        name = src.name.replace(' ','').lower()

        if name in self._src_dict and self._src_dict[name]:
            self.logger.debug('Updating source model for %s'%src.name)
            list(self._src_dict[name])[0].update(src)
            return

        self._src_dict[src.name].add(src)

        for name in src.names:
            self._src_dict[name.replace(' ','').lower()].add(src)

        if isinstance(src,Source):
            self._srcs.append(src)
        else:
            self._diffuse_srcs.append(src)

        if build_index: self.build_src_index()
            
    def load(self,**kwargs):
        """Load both point source and diffuse components."""

        coordsys = kwargs.get('coordsys','CEL')
        extdir = kwargs.get('extdir',self.config['extdir'])

        self._srcs = []
        self.load_diffuse_srcs()
            
        for c in self.config['catalogs']:

            extname = os.path.splitext(c)[1]            
            if extname != '.xml':
                self.load_fits_catalog(c,extdir=extdir,coordsys=coordsys)
            elif extname == '.xml':
                self.load_xml(c,extdir=extdir,coordsys=coordsys)
            else:
                raise Exception('Unrecognized catalog file extension: %s'%c)

        for c in self.config['sources']:
            self.create_source(c,build_index=False)

        self.build_src_index()        
        
    def delete_sources(self,srcs):
        
        for k,v in self._src_dict.items():
            for s in srcs:
                if s in v: 
                    self._src_dict[k].remove(s)
            if not v: del self._src_dict[k]

        self._srcs = [s for s in self._srcs if not s in srcs]
        self._diffuse_srcs = [s for s in self._diffuse_srcs if not s in srcs]
        self.build_src_index()

    @staticmethod
    def create(selection,config,**kwargs):

        if selection['target'] is not None:            
            return ROIModel.create_from_source(selection['target'],
                                               config,**kwargs)
        else:
            target_skydir = get_target_skydir(selection)
            return ROIModel.create_from_position(target_skydir,
                                                 config,**kwargs)
        
    # Creation Methods           
    @staticmethod
    def create_from_position(skydir,config,**kwargs):
        """Create an ROI centered on the given coordinates."""
        
        coordsys = kwargs.pop('coordsys','CEL')
        roi = ROIModel(config,skydir=skydir,coordsys=coordsys,**kwargs)
        return roi

        srcs_dict = {}
            
        if roi.config['src_radius'] is not None:        
            rsrc, srcs = roi.get_sources_by_position(skydir,
                                                     roi.config['src_radius'])
            for s,r in zip(srcs,rsrc):
                srcs_dict[s.name] = (s,r)

        if roi.config['src_roiwidth'] is not None:                
            rsrc, srcs = \
                roi.get_sources_by_position(skydir,
                                            roi.config['src_roiwidth']/2.,
                                            square=True,coordsys=coordsys)
                
            for s,r in zip(srcs,rsrc):
                srcs_dict[s.name] = (s,r)

                
        srcs = []
        rsrc = []
        
        for k, v in srcs_dict.items():
            srcs.append(v[0])
            rsrc.append(v[1])
        
        return ROIModel(config,srcs=srcs,
                        diffuse_srcs=roi._diffuse_srcs,
                        skydir=skydir,**kwargs)
        
        
    @staticmethod
    def create_from_source(name,config,**kwargs):
        """Create an ROI centered on the given source."""

        coordsys = kwargs.pop('coordsys','CEL')
        
        roi = ROIModel(config,src_radius=None,src_roiwidth=None,**kwargs)
        src = roi.get_source_by_name(name,True)

        return ROIModel.create_from_position(src.skydir,config,
                                             coordsys=coordsys,**kwargs)
        
    @staticmethod
    def create_roi_from_ft1(ft1file,config):
        """Create an ROI model by extracting the sources coordinates
        form an FT1 file."""
        pass            
                
    def has_source(self,name):
        
        index_name = name.replace(' ','').lower()
        if index_name in self._src_dict:
            return True
        else:
            return False

    def get_source_by_name(self,name,unique=False):
        """Return a source in the ROI by name.  The input name string
        can match any of the strings in the names property of the
        source object.  Case and whitespace are ignored when matching
        name strings.

        Parameters
        ----------
        name : str 

        unique : bool
           Require a unique match.  If more than one source exists
           with this name an exception is raised.
        """

        index_name = name.replace(' ','').lower()
        
        if index_name in self._src_dict:

            srcs = list(self._src_dict[index_name])

            if len(srcs) == 1 and unique:
                return srcs[0]
            elif not unique:
                return srcs
            else:
                raise Exception('Multiple sources matching name: ' + name)
        else:
            raise Exception('No source matching name: ' + name)

    def get_nearby_sources(self,name,dist,min_dist=None,
                           square=False):
        
        src = self.get_source_by_name(name,True)
        return self.get_sources_by_position(src.skydir,
                                            dist,min_dist,
                                            square)

    def get_sources(self,cuts=None,distance=None,square=False):
        """Retrieve list of sources satisfying the given selections."""
        rsrc, srcs = self.get_sources_by_position(self.skydir,
                                                  distance,
                                                  square=square)
        o = []
        if cuts is None: cuts = []        
        for s,r in zip(srcs,rsrc):
            if not s.check_cuts(cuts): continue            
            o.append(s)

        return o
    
    def get_sources_by_property(self,pname,pmin,pmax=None):

        srcs = []
        for i, s in enumerate(self._srcs):
            if not pname in s: continue
            if pmin is not None and s[pname] < pmin: continue
            if pmax is not None and s[pname] > pmax: continue
            srcs.append(s)
        return srcs
    
    def get_sources_by_position(self,skydir,dist,min_dist=None,
                                square=False,coordsys='CEL'):
        """Retrieve sources within a certain angular distance of an
        (ra,dec) coordinate.  This function supports two types of
        geometric selections: circular (square=False) and square
        (square=True).  The circular selection finds all sources with a given
        angular distance of the target position.  The square selection
        finds sources within an ROI-like region of size R x R where R
        = 2 x dist.

        Parameters
        ----------

        skydir : SkyCoord object
            Sky direction with respect to which the selection will be applied.

        dist : float
            Maximum distance in degrees from the sky coordinate.

        square : bool
            Choose whether to apply a circular or square selection.

        coordsys : str
            Coordinate system to use when applying a selection with square=True.
            
        """

        msk = get_skydir_distance_mask(self._src_skydir,skydir,dist,
                                       min_dist=min_dist,square=square,
                                       coordsys=coordsys)

        radius = self._src_skydir.separation(skydir).deg
        radius = radius[msk]

        srcs = [self._srcs[i] for i in np.nonzero(msk)[0]]
        
        isort = np.argsort(radius)
        radius = radius[isort]
        srcs = [srcs[i] for i in isort]
        
        return radius, srcs
    
    def load_fits_catalog(self,name,**kwargs):

        coordsys = kwargs.get('coordsys','CEL')
        extdir = kwargs.get('extdir',self.config['extdir'])

        cat = Catalog.create(name)

        m0 = get_skydir_distance_mask(cat.skydir,self.skydir,
                                      self.config['src_radius'])
        m1 = get_skydir_distance_mask(cat.skydir,self.skydir,
                                      self.config['src_radius_roi'],
                                      square=True,coordsys=coordsys)
        m = (m0 & m1)

        offset = self.skydir.separation(cat.skydir).deg
        offset_cel = utils.sky_to_offset(self.skydir,
                                         cat.radec[:,0], cat.radec[:,1], 'CEL')
        offset_gal = utils.sky_to_offset(self.skydir,
                                         cat.glonlat[:,0], cat.glonlat[:,1], 'GAL')        

        for i, (row,radec) in enumerate(zip(cat.table[m],
                                            cat.radec[m])):
            
            catalog_dict = row_to_dict(row)
            src_dict = {'catalog' : catalog_dict }
            src_dict['Source_Name'] = row['Source_Name']
            src_dict['SpectrumType'] = row['SpectrumType']

            if row['extended']:
                src_dict['SourceType'] = 'DiffuseSource'
                src_dict['SpatialType'] = 'SpatialMap'
                src_dict['SpatialModel'] = 'SpatialMap'
                src_dict['Spatial_Filename'] = os.path.join(row['extdir'],
                                                            row['Spatial_Filename'])
            else:
                src_dict['SourceType'] = 'PointSource'
                src_dict['SpatialType'] = 'SkyDirFunction'
                src_dict['SpatialModel'] = 'PointSource'                

            src = Source(src_dict['Source_Name'],src_dict,radec=radec)
            src.data['offset'] = offset[m][i]
            src.data['offset_ra'] = offset_cel[:, 0][m][i]
            src.data['offset_dec'] = offset_cel[:, 1][m][i]
            src.data['offset_glon'] = offset_gal[:, 0][m][i]
            src.data['offset_glat'] = offset_gal[:, 1][m][i]            
            self.load_source(src,False)

        self.build_src_index()
        
    def load_xml(self,xmlfile,**kwargs):
        """Load sources from an XML file."""
        
        extdir=kwargs.get('extdir',self.config['extdir'])
        coordsys=kwargs.get('coordsys','CEL')
        if not os.path.isfile(xmlfile):
            xmlfile = os.path.join(fermipy.PACKAGE_ROOT,'catalogs',xmlfile)

        self.logger.info('Reading XML Model: ' + xmlfile)
            
        root = ElementTree.ElementTree(file=xmlfile).getroot()

        diffuse_srcs = []
        srcs = []
        ra, dec = [], []

        for s in root.findall('source'):
            src = Source.create_from_xml(s,extdir=extdir)

            if src.diffuse:
                diffuse_srcs += [src]
            else:
                srcs += [src]
                ra += [src['RAJ2000']]
                dec += [src['DEJ2000']]

        src_skydir = SkyCoord(ra=np.array(ra)*u.deg,
                              dec=np.array(dec)*u.deg)
        radec = np.vstack((src_skydir.ra.deg,src_skydir.dec.deg)).T
        glonlat = np.vstack((src_skydir.galactic.l.deg,
                             src_skydir.galactic.b.deg)).T

        offset = self.skydir.separation(src_skydir).deg
        offset_cel = utils.sky_to_offset(self.skydir,
                                         radec[:,0], radec[:,1], 'CEL')
        offset_gal = utils.sky_to_offset(self.skydir,
                                         glonlat[:,0], glonlat[:,1], 'GAL')  

        m0 = get_skydir_distance_mask(src_skydir,self.skydir,
                                      self.config['src_radius'])
        m1 = get_skydir_distance_mask(src_skydir,self.skydir,
                                      self.config['src_radius_roi'],
                                      square=True,coordsys=coordsys)
        m = (m0 & m1)
        srcs = np.array(srcs)[m]
        for i,s in enumerate(srcs):

            s.data['offset'] = offset[m][i]
            s.data['offset_ra'] = offset_cel[:, 0][m][i]
            s.data['offset_dec'] = offset_cel[:, 1][m][i]
            s.data['offset_glon'] = offset_gal[:, 0][m][i]
            s.data['offset_glat'] = offset_gal[:, 1][m][i] 
            self.load_source(s,False)

        for i,s in enumerate(diffuse_srcs):
            self.load_source(s,False)

        self.build_src_index()

    def build_src_index(self):
        """Build an indices for fast lookup of a source given its name
        or coordinates."""
        
        nsrc = len(self._srcs)
        radec = np.zeros((2,nsrc))
        
        for i, s in enumerate(self._srcs):
            radec[:,i] = s.radec

        self._src_skydir = SkyCoord(ra=radec[0],dec=radec[1],unit=u.deg)
        self._src_radius = self._src_skydir.separation(self.skydir)
                
    def write_xml(self,xmlfile):
        """Save this ROI model as an XML file."""
        
        root = ElementTree.Element('source_library')
        root.set('title','source_library')

        for s in self._srcs:
            s.write_xml(root)
                
        for s in self._diffuse_srcs:
            s.write_xml(root)
                
        output_file = open(xmlfile,'w')
        output_file.write(prettify_xml(root))

if __name__ == '__main__':

    
    roi = ROIModel()


    roi.load_fits('gll_fssc_psc_v14.fit')


    src = roi.get_source_by_name('lmc')


    import pprint
    pprint.pprint(src.data)

    print src

    srcs = roi.get_nearby_sources('lmc',10.0)

#    for s in srcs:        
#        print s.name, s.associations, s.separation(src)

    roi.create_roi_from_source('test.xml','lmc','test','test',90.0)
