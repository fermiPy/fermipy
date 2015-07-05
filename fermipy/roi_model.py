import defaults 
from utils import *
import pyfits
import fermipy
from fermipy.logger import Logger
from fermipy.logger import logLevel as ll

import xml.etree.cElementTree as ElementTree
from astropy import units as u
from astropy.coordinates import SkyCoord


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

    if p > 0:    
        scale = 10**-np.round(np.log10(1./p))
        return p/scale, scale
    else:
        return p, 1.0
    
def get_dist_to_edge(skydir,lon,lat,width,coordsys='CEL'):

    xy = sky_to_offset(skydir,np.degrees(lon),np.degrees(lat),
                       coordsys=coordsys)

    x = np.radians(xy[:,0])
    y = np.radians(xy[:,1])
    
    delta_edge = np.array([np.abs(x) - width,np.abs(y) - width])
    dtheta = np.max(delta_edge,axis=0)
    return dtheta

class Model(object):

    def __init__(self,data=None,
                 spectral_pars=None,
                 spatial_pars=None):
        self._data = {} if data is None else data
        self._spectral_pars = {} if spectral_pars is None else spectral_pars
        self._spatial_pars = {} if spatial_pars is None else spatial_pars
        
    def __contains__(self,key):
        return key in self._data

    def __getitem__(self,key):
        return self._data[key]

    def __setitem__(self,key,value):
        self._data[key]=value

    def __eq__(self, other): 
        return self.name == other.name
        
    def update(self,m):

        if self['SpectrumType'] != m['SpectrumType']:
            self._spectral_pars = {}
        
        self._data = merge_dict(self._data,m._data,add_new_keys=True)
            
        self._spectral_pars = merge_dict(self._spectral_pars,
                                         m._spectral_pars,add_new_keys=True)
        self._spatial_pars = merge_dict(self._spatial_pars,
                                        m._spatial_pars,add_new_keys=True)
        
class IsoSource(Model):

    def __init__(self,filefunction,name,spectral_pars=None,spatial_pars=None):
        super(IsoSource,self).__init__(None,spectral_pars,spatial_pars)
        
        self._filefunction = filefunction
        self._name = name
        self['SpectrumType'] = 'FileFunction'

        if not self._spectral_pars:
            self._spectral_pars = {
                'Normalization' : {'name' : 'Normalization', 'scale' : '1.0',
                                   'value' : '1.0',
                                   'min' : '0.001', 'max' : '1000.0',
                                   'free' : '0' } }

        if not self._spatial_pars:            
            self._spatial_pars = {
                'Value' : {'name' : 'Value', 'scale' : '1',
                           'value' : '1', 'min' : '0', 'max' : '10',
                           'free' : '0' } }

    @property
    def name(self):
        return self._name        
        
    @property
    def filefunction(self):
        return self._filefunction

    def write_xml(self,root):
        
        source_element = create_xml_element(root,'source',
                                            dict(name=self.name,
                                                 type='DiffuseSource'))

        
        spec_el = create_xml_element(source_element,'spectrum',
                                     dict(file=self.filefunction,
                                          type='FileFunction',
                                          ctype='-1'))
                        
        spat_el = create_xml_element(source_element,'spatialModel',
                                     dict(type='ConstantValue'))


        for k,v in self._spectral_pars.items():                
            create_xml_element(spec_el,'parameter',v)

        for k,v in self._spatial_pars.items():                
            create_xml_element(spat_el,'parameter',v)
        
class MapCubeSource(Model):

    def __init__(self,mapcube,name,spectral_pars=None,spatial_pars=None):
        super(MapCubeSource,self).__init__(None,spectral_pars,spatial_pars)

        self._mapcube = mapcube
        self._name = name
        self['SpectrumType'] = 'PowerLaw'

        if not self._spectral_pars:
            self._spectral_pars = {
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

        if not self._spatial_pars:            
            self._spatial_pars = {
                'Normalization' :
                    {'name' : 'Normalization', 'scale' : '1',
                     'value' : '1', 'min' : '0', 'max' : '10',
                     'free' : '0' } }

    @property
    def name(self):
        return self._name        
        
    @property
    def mapcube(self):
        return self._mapcube

    def write_xml(self,root):
        
        source_element = create_xml_element(root,'source',
                                            dict(name=self.name,
                                                 type='DiffuseSource'))

        spec_el = create_xml_element(source_element,'spectrum',
                                     dict(type='PowerLaw'))
                        
        spat_el = create_xml_element(source_element,'spatialModel',
                                     dict(type='MapCubeFunction',
                                          file=self._mapcube))


        for k,v in self._spectral_pars.items():                
            create_xml_element(spec_el,'parameter',v)

        for k,v in self._spatial_pars.items():                
            create_xml_element(spat_el,'parameter',v)
        
class Source(Model):

    def __init__(self,data=None,
                 radec=None,
                 glonlat=None,
                 spectral_pars=None,
                 spatial_pars=None,
                 extended=False):
        super(Source,self).__init__(data,spectral_pars,spatial_pars)
                    
#        phi = np.radians(data['RAJ2000'])
#        theta = np.pi/2.-np.radians(data['DEJ2000'])

        self._radec = radec
        self._glonlat = glonlat

#        np.array([np.sin(theta)*np.cos(phi),
#                                np.sin(theta)*np.sin(phi),
#                                np.cos(theta)])


        ts_keys = ['Sqrt_TS30_100','Sqrt_TS100_300',
                   'Sqrt_TS300_1000','Sqrt_TS1000_3000',
                   'Sqrt_TS3000_10000','Sqrt_TS10000_100000']

        if ts_keys[0] in self:        
            self._data['TS_value'] = 0
            for k in ts_keys:
                if k in self and np.isfinite(self[k]):
                    self._data['TS_value'] += self[k]**2
        
        self._names = []
        self._names_dict = {}
        for k in ROIModel.src_name_cols:

            if not k in self._data: continue

            name = self._data[k].strip()
            if name != '':  self._names.append(name)

            self._names_dict[k] = name

        self._extended=extended

        if not self._spectral_pars:
            self._update_spectral_pars()

        if not self.extended and not self._spatial_pars:
            
            self._spatial_pars = {
                'RA' : {'name' : 'RA',  'value' : str(self['RAJ2000']),
                        'free' : '0',
                        'min' : '-360.0','max' : '360.0','scale' : '1.0'},
                'DEC' : {'name' : 'DEC',  'value' : str(self['DEJ2000']),
                         'free' : '0',
                         'min' : '-90.0','max' : '90.0','scale' : '1.0'}
                }
        elif self.extended and not self._spatial_pars:

            self._spatial_pars = {
                'Prefactor' : {'name' : 'Prefactor', 'value' : '1',
                               'free' : '0', 'min' : '0.001', 'max' : '1000',
                               'scale' : '1.0'}
                }
            
    def _update_spectral_pars(self):

        if self['SpectrumType'] == 'PowerLaw':

            if not 'Prefactor' in self:
                self._data['Prefactor'] = self['Flux_Density']
            
            if not 'Scale' in  self:
                self._data['Scale'] = self['Pivot_Energy']
            prefactor, prefactor_scale = scale_parameter(self['Prefactor'])

            index_max = max(5.0,self['Spectral_Index']+1.0)
            index_min = min(0.0,self['Spectral_Index']-1.0)
            
            self._spectral_pars = {
                'Prefactor' : {'name' : 'Prefactor', 'value' : str(prefactor),
                               'scale' : str(prefactor_scale),
                               'min' : '0.01', 'max' : '100.0', 'free' : '0'},
                'Index' : {'name' : 'Index',
                           'value' : str(self['Spectral_Index']),
                           'scale' : str(-1.0),
                           'min' : str(index_min), 'max' : str(index_max), 'free' : '0'},
                'Scale' :  {'name' : 'Scale',
                            'value' : str(self['Scale']),
                            'scale' : str(1.0),
                            'min' : str(self['Scale']),
                            'max' : str(self['Scale']), 'free' : '0'}
                }

        elif self['SpectrumType'] == 'LogParabola':

            norm_value, norm_scale = scale_parameter(self['Flux_Density'])
            eb_value, eb_scale = scale_parameter(self['Pivot_Energy'])

            self._spectral_pars = {
                'norm' : {'name' : 'norm', 'value' : str(norm_value),
                          'scale' : str(norm_scale),
                          'min' : '0.01', 'max' : '100.0', 'free' : '0'},
                'alpha' : {'name' : 'alpha',
                           'value' : str(self['Spectral_Index']),
                           'scale' : str(1.0),
                           'min' : '-5.0', 'max' : '5.0', 'free' : '0'},
                'beta' :  {'name' : 'beta', 'value' : str(self['beta']),
                           'scale' : str(1.0),
                           'min' : '-10.0', 'max' : '10.0', 'free' : '0'},
                'Eb' :  {'name' : 'Eb', 'value' : str(eb_value),
                         'scale' : str(eb_scale),
                         'min' : '0.01', 'max' : '100.0', 'free' : '0'},
                }
        elif self['SpectrumType'] == 'PLSuperExpCutoff':

            flux_density = self['Flux_Density']
            flux_density *= np.exp((self['Pivot_Energy']/self['Cutoff'])**self['Exp_Index'])
            
            prefactor, prefactor_scale = scale_parameter(flux_density)
            cutoff, cutoff_scale = scale_parameter(self['Cutoff'])
                
            self._spectral_pars = {
                'Prefactor' : {'name' : 'Prefactor', 'value' : str(prefactor),
                               'scale' : str(prefactor_scale),
                               'min' : '0.01', 'max' : '100.0', 'free' : '0'},
                'Index1' : {'name' : 'Index1', 'value' : str(self['Spectral_Index']),
                           'scale' : str(-1.0), 'min' : '0.0', 'max' : '5.0', 'free' : '0'},
                'Index2' : {'name' : 'Index2', 'value' : str(self['Exp_Index']),
                           'scale' : str(1.0), 'min' : '0.0', 'max' : '2.0', 'free' : '0'},
                'Cutoff' : {'name' : 'Cutoff', 'value' : str(cutoff),
                           'scale' : str(cutoff_scale), 'min' : '0.01', 'max' : '100.0', 'free' : '0'},
                'Scale' :  {'name' : 'Scale', 'value' : str(self['Pivot_Energy']),
                            'scale' : str(1.0),
                            'min' : str(self['Pivot_Energy']),
                            'max' : str(self['Pivot_Energy']), 'free' : '0'}
                }
        else:

            import pprint
            pprint.pprint(self._data)            
            raise Exception('Unsupported spectral type:' + self['SpectrumType'])
            
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
    def extended(self):
        return self._extended

    @property
    def name(self):
        return self._data['Source_Name']

    @property
    def associations(self):
        return self._names

    @property
    def radec(self):
        return self._radec

    @property
    def skydir(self):
        return SkyCoord(self._radec[0]*u.deg,self._radec[1]*u.deg)
    
    @property
    def data(self):
        return self._data

    @staticmethod
    def load_from_dict(src_dict):

        src_dict.setdefault('SpatialType','PointSource')
        src_dict.setdefault('SpectrumType','PowerLaw')
        src_dict.setdefault('Spectral_Index',2.0)
        src_dict.setdefault('Pivot_Energy',1000.0)

        if src_dict['SpatialType'] != 'PointSource':
            extended=True
        else:
            extended=False
        
        if 'name' in src_dict:
            src_dict['Source_Name'] = src_dict['name']
        
        skydir = get_target_skydir(src_dict)

        print 'getting position'
        print skydir
        
        
        src_dict['RAJ2000'] = skydir.ra.deg
        src_dict['DEJ2000'] = skydir.dec.deg

        radec = np.array([skydir.ra.deg,skydir.dec.deg])
        
        print src_dict
        
        return Source(src_dict,radec=radec,extended=extended)
    
    @staticmethod
    def load_from_xml(root,extdir=None):
        
        spec = load_xml_elements(root,'spectrum')
        spat = load_xml_elements(root,'spatialModel')
        spectral_pars = load_xml_elements(root,'spectrum/parameter')
        spatial_pars = load_xml_elements(root,'spatialModel/parameter')

        src_type = root.attrib['type']
        spatial_type = spat['type']
        spectral_type = spec['type']
        
        src_dict = copy.deepcopy(root.attrib)

        src_dict['Source_Name'] = src_dict['name']
        src_dict['SpectrumType'] = spec['type']
            
        if src_type =='PointSource' or spatial_type == 'SpatialMap':
        
            extflag=False        
            if 'file' in spat: 
                src_dict['Spatial_Filename'] = spat['file']
                extflag=True

                if not os.path.isfile(src_dict['Spatial_Filename']) \
                        and extdir is not None:
                    src_dict['Spatial_Filename'] = \
                        os.path.join(extdir,'Templates',
                                     src_dict['Spatial_Filename'])

            
            
            if 'RA' in src_dict:
                src_dict['RAJ2000'] = float(src_dict['RA'])
                src_dict['DEJ2000'] = float(src_dict['DEC'])
            elif 'RA' in spatial_pars:
                src_dict['RAJ2000'] = float(spatial_pars['RA']['value'])
                src_dict['DEJ2000'] = float(spatial_pars['DEC']['value'])
            else:
                hdu = pyfits.open(src_dict['Spatial_Filename'])
                src_dict['RAJ2000'] = float(hdu[0].header['CRVAL1'])
                src_dict['DEJ2000'] = float(hdu[0].header['CRVAL2'])
                                
            return Source(src_dict,
                          spectral_pars=spectral_pars,
                          spatial_pars=spatial_pars,extended=extflag)

        elif src_type == 'DiffuseSource' and spatial_type == 'ConstantValue':
            return IsoSource(spec['file'],'isodiff',spectral_pars,spatial_pars)
        elif src_type == 'DiffuseSource' and spatial_type == 'MapCubeFunction':
            return MapCubeSource(spat['file'],'galdiff',
                                 spectral_pars,spatial_pars)
        else:
            raise Exception('Unrecognized type for source: %s'%src_dict['Source_Name'])
        
    def write_xml(self,root):

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
            
            spat_el = create_xml_element(source_element,'spatialModel',
                                         dict(map_based_integral='True',
                                              type='SpatialMap',
                                              file=self['Spatial_Filename']))
                    
        for k,v in self._spatial_pars.items():                
            create_xml_element(spat_el,'parameter',v)

                
        el = ElementTree.SubElement(source_element,'spectrum')  
        
        stype = self['SpectrumType'].strip()            
        el.set('type',stype)

#        spec_element.set('type','PLSuperExpCutoff')
        
        for k,v in self._spectral_pars.items():                
            create_xml_element(el,'parameter',v)
    
class ROIModel(AnalysisBase):
    """This class is responsible for managing the ROI definition.
    Catalogs can be read from either FITS or XML files."""

    defaults = dict(defaults.model.items(),
                    logfile=(None,''),
                    skydir=(None,''),
                    fileio=defaults.fileio,
                    logging=defaults.logging)

    src_name_cols = ['Source_Name',
                     'ASSOC1','ASSOC2','ASSOC_GAM',
                     '1FHL_Name','2FGL_Name',
                     'ASSOC_GAM1','ASSOC_GAM2','ASSOC_TEV']

    def __init__(self,config=None,srcs=None,diffuse_srcs=None,**kwargs):
        super(ROIModel,self).__init__(config,**kwargs)
        
        self.logger = Logger.get(self.__class__.__name__,
                                 self.config['logfile'],
                                 ll(self.config['logging']['verbosity']))
        
        if not os.path.isdir(self.config['extdir']):
            self._config['extdir'] = \
                os.path.join(fermipy.PACKAGE_ROOT,
                             'catalogs',self.config['extdir'])
        
        # Coordinate for ROI center (defaults to 0,0)
        self._skydir = kwargs.get('skydir',SkyCoord(0.0,0.0,unit=u.deg))                
#        self._radec = SkyCoord(*kwargs.get('radec',[0.0,0.0]),unit=u.deg)
        self._srcs = []
        self._diffuse_srcs = []
        self._src_index = {}
        self._src_radius = []

        if srcs is None: srcs = []
        if diffuse_srcs is None: diffuse_srcs = []
            
        for s in srcs + diffuse_srcs:
            self.load_source(s)
        
        self.build_src_index()

    def __iter__(self):
        return iter(self._srcs + self._diffuse_srcs)

    @property
    def skydir(self):
        """Return the sky direction objection corresponding to the center of the ROI."""        
        return self._skydir

    def load_diffuse_srcs(self):
        
        if self.config['isodiff'] is not None:
            self.load_source(IsoSource(self.config['isodiff'],'isodiff'))

        if self.config['galdiff'] is not None:
            self.load_source(MapCubeSource(self.config['galdiff'],'galdiff'))

    def create_source(self,src_dict,build_index=True):
        
        src_dict.setdefault('SpatialType','PointSource')
        src_dict.setdefault('SpectrumType','PowerLaw')
        src_dict.setdefault('Spectral_Index',2.0)
        src_dict.setdefault('Pivot_Energy',1000.0)
                
        src = Source.load_from_dict(src_dict)

        if src['SpatialType'] == 'Gaussian':
            template_file = os.path.join(self.config['fileio']['workdir'],'template.fits')
            make_gaussian_spatial_map(src.skydir,0.5,template_file)
            src['Spatial_Filename'] = template_file
            
        self.load_source(src)

        if build_index: self.build_src_index()
        
    def load_source(self,src):
        
        if src.name in self._src_index:
            self.logger.info('Updating source model for %s'%src.name)
            self._src_index[src.name].update(src)
            return

        self._src_index[src.name] = src
        
        for c in ROIModel.src_name_cols:
            if not c in src: continue
            name = src[c].strip()
            self._src_index[name] = src
            self._src_index[name.replace(' ','')] = src
            self._src_index[name.replace(' ','').lower()] = src

        if isinstance(src,Source):
            self._srcs.append(src)
        else:
            self._diffuse_srcs.append(src)
            
    def load(self):

        self._srcs = []
        self.load_diffuse_srcs()
            
        for c in self.config['catalogs']:

            extname = os.path.splitext(c)[1]            
            if extname == '.fits' or extname == '.fit':
                self.load_fits(c)
            elif extname == '.xml':
                self.load_xml(c)
            else:
                raise Exception('Unrecognized catalog file extension: %s'%c)

        for c in self.config['sources']:
            self.create_source(c,build_index=False)

        self.build_src_index()        
        
    def delete_sources(self,srcs):

#        srcs = []
#        for n in names:        
#            if not n in self._src_index:
#                raise Exception('No source with name: %s'%n)
#            srcs.append(self._src_index[n])
        
        self._src_index = {k:v for k,v in self._src_index.items() if not v in srcs}
        self._srcs = [s for s in self._srcs if not s in srcs]
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

        roi = kwargs.pop('roi',None)
        if roi is None:        
            roi = ROIModel(config,**kwargs)
            roi.load()

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
                                            roilike=True)
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

        roi = ROIModel(config,**kwargs)
        roi.load()
        src = roi.get_source_by_name(name)

        return ROIModel.create_from_position(src.skydir,config,
                                             roi=roi,**kwargs)
        
    @staticmethod
    def create_roi_from_ft1(ft1file,config):
        """Create an ROI model by extracting the sources coordinates
        form an FT1 file."""
        pass            
                
    def get_source_by_name(self,name):
        """Retrieve source by name."""

        if name in self._src_index:
            return self._src_index[name]
        else:
            raise Exception('No source matching name: ' + name)

    def get_nearby_sources(self,name,dist,min_dist=None,
                           roilike=False):
        
        src = self.get_source_by_name(name)
        return self.get_sources_by_position(src.skydir,
                                            dist,min_dist,
                                            roilike)

    def get_sources_by_property(self,pname,pmin,pmax=None):

        srcs = []
        for i, s in enumerate(self._srcs):
            if not pname in s: continue
            if pmin is not None and s[pname] < pmin: continue
            if pmax is not None and s[pname] > pmax: continue
            srcs.append(s)
        return srcs
    
    def get_sources_by_position(self,skydir,dist,min_dist=None,
                                roilike=False):
        """Retrieve sources within a certain angular distance of an
        (ra,dec) coordinate.  This function supports two types of
        geometric selections: circle (roilike=False) and roi
        (roilike=True).  The former selects all sources with a given
        angular distance of the target position.  The roi selection
        finds sources within an ROI-like region of size R x R where R
        = 2 x dist.

        Parameters
        ----------

        skydir : SkyCoord object

        dist : float
        
        """

        if dist is None: dist = 180.
        
        radius = self._src_skydir.separation(skydir).rad
        
        if not roilike:                    
            dtheta = radius            
        else:
            dtheta = get_dist_to_edge(skydir,
                                      self._src_skydir.ra.rad,
                                      self._src_skydir.dec.rad,
                                      np.radians(dist))
            
            dtheta += np.radians(dist)            
        
        if min_dist is not None:
            msk = np.where((dtheta < np.radians(dist)) &
                           (dtheta > np.radians(min_dist)))[0]
        else:
            msk = np.where(dtheta < np.radians(dist))[0]

        radius = radius[msk]
        srcs = [ self._srcs[i] for i in msk ]

        isort = np.argsort(radius)

        radius = radius[isort]
        srcs = [srcs[i] for i in isort]
        
        return radius, srcs
    
    def load_fits(self,fitsfile,
                  src_hduname='LAT_Point_Source_Catalog',
                  extsrc_hduname='ExtendedSources'):
        """Load sources from a FITS catalog file."""

        if not os.path.isfile(fitsfile):
            fitsfile = os.path.join(fermipy.PACKAGE_ROOT,'catalogs',fitsfile)
        
        hdulist = pyfits.open(fitsfile)
        table_src = hdulist[src_hduname]
        table_extsrc = hdulist[extsrc_hduname]

        # Rearrange column data in a more convenient format
        cols = fits_recarray_to_dict(table_src)
        cols_extsrc = fits_recarray_to_dict(table_extsrc)

        extsrc_names = cols_extsrc['Source_Name'].tolist()
        extsrc_names = [s.strip() for s in extsrc_names]

        src_skydir = SkyCoord(ra=cols['RAJ2000']*u.deg,
                             dec=cols['DEJ2000']*u.deg)

        radec = np.vstack((src_skydir.ra.deg,src_skydir.dec.deg)).T
        glonlat = np.vstack((src_skydir.galactic.l.deg,src_skydir.galactic.b.deg)).T
        
        nsrc = len(table_src.data)
        for i in range(nsrc):

            src_dict = {}
            for icol, col in enumerate(cols):
                src_dict[col] = cols[col][i]

            extflag=False

            src_dict['Source_Name'] = src_dict['Source_Name'].strip()  
            extsrc_name = src_dict['Extended_Source_Name'].strip()

            if len(extsrc_name.strip()) > 0:
                extflag=True
                extsrc_index = extsrc_names.index(extsrc_name) 
                
                for icol, col in enumerate(cols_extsrc):
                    if col in cols: continue
                    src_dict[col] = cols_extsrc[col][extsrc_index]

                src_dict['Spatial_Filename'] = src_dict['Spatial_Filename'].strip()

                if not os.path.isfile(src_dict['Spatial_Filename']) and self.config['extdir']:
                    src_dict['Spatial_Filename'] = os.path.join(self.config['extdir'],
                                                                'Templates',
                                                                src_dict['Spatial_Filename'])
            
            src_dict['SpectrumType'] = src_dict['SpectrumType'].strip()
            if src_dict['SpectrumType'] == 'PLExpCutoff':
                src_dict['SpectrumType'] = 'PLSuperExpCutoff'
            
            src = Source(src_dict,radec=radec[i],glonlat=glonlat[i],extended=extflag)
            self.load_source(src)
            
        self.build_src_index()

#            cat.load_source(ROIModelSource(src))

    def build_src_index(self):
        """Build an indices for fast lookup of a source given its name
        or coordinates."""
        
        nsrc = len(self._srcs)
        radec = np.zeros((2,nsrc))
        
        for i, s in enumerate(self._srcs):
            radec[:,i] = s.radec

        self._src_skydir = SkyCoord(ra=radec[0],dec=radec[1],unit=u.deg)
        self._src_radius = self._src_skydir.separation(self.skydir)
        
        for i, s in enumerate(self._diffuse_srcs):
            pass
        
    def write_xml(self,xmlfile,isodiff=None,galdiff=None):
        """Save this ROI model as an XML file."""
        
        root = ElementTree.Element('source_library')
        root.set('title','source_library')

        for s in self._srcs:
            s.write_xml(root)
                
        for s in self._diffuse_srcs:
            s.write_xml(root)
                
        output_file = open(xmlfile,'w')
        output_file.write(prettify_xml(root))

    def load_xml(self,xmlfile):
        """Load sources from an XML file."""

        if not os.path.isfile(xmlfile):
            xmlfile = os.path.join(fermipy.PACKAGE_ROOT,'catalogs',xmlfile)

        self.logger.info('Reading XML Model: ' + xmlfile)
            
        root = ElementTree.ElementTree(file=xmlfile).getroot()

        for s in root.findall('source'):
            src = Source.load_from_xml(s,extdir=self.config['extdir'])
            self.load_source(src)

        self.build_src_index()

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
