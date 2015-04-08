import defaults 
from utils import *
import pyfits
import fermipy

import xml.etree.cElementTree as ElementTree


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
    
def get_dist_to_edge(lon0,lat0,lon1,lat1,width):

    x,y = project(lon0,lat0,lon1,lat1)
    delta_edge = np.array([np.abs(x) - width,np.abs(y) - width])
    dtheta = np.max(delta_edge,axis=0)
    return dtheta
    
class IsoSource(object):

    def __init__(self,filefunction,name,spectral_pars=None,spatial_pars=None):
        self._filefunction = filefunction
        self._name = name

        self._spectral_pars = {} if spectral_pars is None else spectral_pars
        self._spatial_pars = {} if spatial_pars is None else spatial_pars

        if not self._spectral_pars:
            self._spectral_pars = {
                'Normalization' : {'name' : 'Normalization', 'scale' : '1.0',
                                   'value' : '1.0', 'min' : '0.001', 'max' : '1000.0',
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
        
class MapCubeSource(object):

    def __init__(self,mapcube,name,spectral_pars=None,spatial_pars=None):
        self._mapcube = mapcube
        self._name = name

        self._spectral_pars = {} if spectral_pars is None else spectral_pars
        self._spatial_pars = {} if spatial_pars is None else spatial_pars

        if not self._spectral_pars:
            self._spectral_pars = {
                'Prefactor' : {'name' : 'Prefactor', 'scale' : '1',
                               'value' : '0.0', 'min' : '-1.0', 'max' : '1.0',
                               'free' : '0' },
                'Index' : {'name' : 'Index', 'scale' : '1',
                               'value' : '1.0', 'min' : '0.0', 'max' : '10.0',
                               'free' : '0' },
                'Scale' : {'name' : 'Scale', 'scale' : '1',
                           'value' : '1000.0', 'min' : '1000.0', 'max' : '10000.0',
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
        
class Source(object):

    def __init__(self,col_data=None,
                 spectral_pars=None,
                 spatial_pars=None,
                 extended=False):

        self._col_data = {} if col_data is None else col_data
        self._spectral_pars = {} if spectral_pars is None else spectral_pars
        self._spatial_pars = {} if spatial_pars is None else spatial_pars
                    
        phi = np.radians(col_data['RAJ2000'])
        theta = np.pi/2.-np.radians(col_data['DEJ2000'])

        self._radec = np.array([np.sin(theta)*np.cos(phi),
                                np.sin(theta)*np.sin(phi),
                                np.cos(theta)])
            
        self._names = []
        self._names_dict = {}
        for k in ROIManager.src_name_cols:

            if not k in self._col_data: continue

            name = self._col_data[k].strip()
            if name != '':  self._names.append(name)

            self._names_dict[k] = name

        self._extended=extended

        if not self._spectral_pars:
            self._update_spectral_pars()

        if not self.extended and not self._spatial_pars:
            
            self._spatial_pars = {
                'RA' : {'name' : 'RA',  'value' : str(self['RAJ2000']), 'free' : '0',
                        'min' : '-360.0','max' : '360.0','scale' : '1.0'},
                'DEC' : {'name' : 'DEC',  'value' : str(self['DEJ2000']), 'free' : '0',
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

            prefactor, prefactor_scale = scale_parameter(self['Flux_Density'])
            
            self._spectral_pars = {
                'Prefactor' : {'name' : 'Prefactor', 'value' : str(prefactor),
                               'scale' : str(prefactor_scale),
                               'min' : '0.01', 'max' : '100.0', 'free' : '0'},
                'Index' : {'name' : 'Index', 'value' : str(self['Spectral_Index']),
                           'scale' : str(-1.0), 'min' : '-5.0', 'max' : '5.0', 'free' : '0'},
                'Scale' :  {'name' : 'Scale', 'value' : str(self['Pivot_Energy']),
                            'scale' : str(1.0),
                            'min' : str(self['Pivot_Energy']),
                            'max' : str(self['Pivot_Energy']), 'free' : '0'}
                }

        elif self['SpectrumType'] == 'LogParabola':

            norm_value, norm_scale = scale_parameter(self['Flux_Density'])
            eb_value, eb_scale = scale_parameter(self['Pivot_Energy'])

            self._spectral_pars = {
                'norm' : {'name' : 'norm', 'value' : str(norm_value),
                          'scale' : str(norm_scale),
                          'min' : '0.01', 'max' : '100.0', 'free' : '0'},
                'alpha' : {'name' : 'alpha', 'value' : str(self['Spectral_Index']),
                           'scale' : str(1.0), 'min' : '-5.0', 'max' : '5.0', 'free' : '0'},
                'beta' :  {'name' : 'beta', 'value' : str(self['beta']),
                           'scale' : str(1.0),
                           'min' : '-10.0', 'max' : '10.0', 'free' : '0'},
                'Eb' :  {'name' : 'Eb', 'value' : str(eb_value),
                         'scale' : str(eb_scale),
                         'min' : '0.01', 'max' : '100.0', 'free' : '0'},
                }
        elif self['SpectrumType'] == 'PLSuperExpCutoff':

            prefactor, prefactor_scale = scale_parameter(self['Flux_Density'])
            cutoff, cutoff_scale = scale_parameter(self['Cutoff'])
                
            self._spectral_pars = {
                'Prefactor' : {'name' : 'Prefactor', 'value' : str(prefactor),
                               'scale' : str(prefactor_scale),
                               'min' : '0.01', 'max' : '100.0', 'free' : '0'},
                'Index1' : {'name' : 'Index1', 'value' : str(self['Spectral_Index']),
                           'scale' : str(-1.0), 'min' : '-5.0', 'max' : '5.0', 'free' : '0'},
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
            pprint.pprint(self._col_data)

            sys.exit(0)
            
            
                               
    def separation(self,src):

        if isinstance(src,Source):        
            costh = np.sum(self._radec*src.radec) 
        else:
            costh = np.sum(self._radec*src) 
            
        costh = min(1.0,costh)
        costh = max(costh,-1.0)
        return np.degrees(np.arccos(costh))
    
    @property
    def extended(self):
        return self._extended

    @property
    def name(self):
        return self._col_data['Source_Name']

    @property
    def associations(self):
        return self._names

    @property
    def radec(self):
        return self._radec

    @property
    def data(self):
        return self._col_data

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
            else:
                src_dict['RAJ2000'] = float(spatial_pars['RA']['value'])
                src_dict['DEJ2000'] = float(spatial_pars['DEC']['value'])
            
            return Source(src_dict,
                          spectral_pars=spectral_pars,
                          spatial_pars=spatial_pars,extended=extflag)

        elif src_type == 'DiffuseSource' and spatial_type == 'ConstantValue':
            return IsoSource(spec['file'],'isodiff',spectral_pars,spatial_pars)
        elif src_type == 'DiffuseSource' and spatial_type == 'MapCubeFunction':
            return MapCubeSource(spat['file'],'galdiff',spectral_pars,spatial_pars)
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

        
            
    
    def __contains__(self,key):
        return key in self._col_data

    def __getitem__(self,key):
        return self._col_data[key]

    def __setitem__(self,key,value):
        self._col_data[key]=value
    
class ROIManager(AnalysisBase):
    """This class is responsible for managing the ROI definition.
    Catalogs can be read from either FITS or XML files."""

    defaults = dict(defaults.roi.items())

    src_name_cols = ['Source_Name',
                     'ASSOC1','ASSOC2','ASSOC_GAM',
                     '1FHL_Name','2FGL_Name',
                     'ASSOC_GAM1','ASSOC_GAM2','ASSOC_TEV']

    def __init__(self,config=None,**kwargs):
        super(ROIManager,self).__init__(config,**kwargs)

        if not os.path.isdir(self.config['extdir']):
            self._config['extdir'] = os.path.join(fermipy.PACKAGE_ROOT,
                                                  'catalogs',self.config['extdir'])
        
        # Coordinate for ROI center (defaults to 0,0)
        self._radec = kwargs.get('radec',np.array([0.0,0.0]))    
        self._srcs = kwargs.get('srcs',[])
        self._diffuse_srcs = kwargs.get('diffuse_srcs',[])

        self._src_index = {}
        self._diffuse_src_index = {}
        self._src_radius = []
        
        self.build_src_index()

    def __iter__(self):
        return iter(self._srcs + self._diffuse_srcs)

    @property
    def radec(self):
        """Return the coordinates of the center of the ROI in deg."""        
        return self._radec

    def load_diffuse_srcs(self):
        self._diffuse_srcs = []
        
        if self.config['isodiff'] is not None:
            self._diffuse_srcs.append(IsoSource(self.config['isodiff'],
                                                'isodiff'))
#        else:
#            self._diffuse_srcs.append(IsoSource(None,'isodiff'))

        if self.config['galdiff'] is not None:
            self._diffuse_srcs.append(MapCubeSource(self.config['galdiff'],
                                                    'galdiff'))
            
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

    # Creation Methods           
    @staticmethod
    def create_roi_from_coords(name,config):
        """Create an ROI centered on the given coordinates."""
        pass
        
    @staticmethod
    def create_roi_from_source(name,config):
        """Create an ROI centered on the given source."""
        
        roi = ROIManager(config)
        roi.load()

        src = roi.get_source_by_name(name)

        srcs_dict = {}
        
        if roi.config['radius'] is not None:
        
            rsrc, srcs = roi.get_nearby_sources(name,roi.config['radius'])
            for s,r in zip(srcs,rsrc):
                srcs_dict[s.name] = (s,r)

        if roi.config['roisize'] is not None:                
            rsrc, srcs = roi.get_nearby_sources(name,roi.config['roisize']/2.,
                                                selection='roi')
            for s,r in zip(srcs,rsrc):
                srcs_dict[s.name] = (s,r)

        srcs = []
        rsrc = []
                
        for k, v in srcs_dict.items():
            srcs.append(v[0])
            rsrc.append(v[1])
               
        radec = np.array([src['RAJ2000'],src['DEJ2000']])
        
        return ROIManager(config,srcs=srcs,
                          diffuse_srcs=roi._diffuse_srcs,radec=radec)

    @staticmethod
    def create_roi_from_ft1(ft1file,config):
        """Create an ROI model by extracting the sources coordinates
        form an FT1 file."""
        pass        
            
    @staticmethod
    def create_isotropic(src,root,filefunction=None):

        default_norm = dict(name='Normalization',value='1.0',free='1',
                            max='10000.0',min='0.0001',scale='1.0')
        default_value = dict(name='Value',value='1.0',free='0',
                             max='10.0', min='0.0',scale='1.0')

        el = create_xml_element(root,'source',
                                dict(name=src.name,
                                     type='DiffuseSource'))

        if filefunction is None: filefunction = src.filefunction
        
        spec_el = create_xml_element(el,'spectrum',
                                     dict(file=filefunction,
                                          type='FileFunction',
                                          ctype='-1'))

        create_xml_element(spec_el,'parameter',default_norm)
                        
        spat_el = create_xml_element(el,'spatialModel',
                                     dict(type='ConstantValue'))

        create_xml_element(spat_el,'parameter',default_value)

        return el

    @staticmethod
    def create_mapcube(src,root,mapcube=None):
        
        el = create_xml_element(root,'source',
                                dict(name=src.name,
                                     type='DiffuseSource'))

        spec_el = create_xml_element(el,'spectrum',
                                     dict(type='PowerLaw'))
        
                
        create_xml_element(spec_el,'parameter',
                           dict(name='Prefactor',
                                value='1.0',
                                free='1',
                                max='10.0',
                                min='0.1',
                                scale='1.0'))
        
        create_xml_element(spec_el,'parameter',
                           dict(name='Index',
                                value='0.0',
                                free='0',
                                max='1.0',
                                min='-1.0',
                                scale='-1.0'))

        create_xml_element(spec_el,'parameter',
                           dict(name='Scale',
                                value='1000.0',
                                free='0',
                                max='1000.0',
                                min='1000.0',
                                scale='1.0'))

        spat_el = create_xml_element(el,'spatialModel',
                                     dict(type='MapCubeFunction',
                                          file=src.mapcube))
                
        create_xml_element(spat_el,'parameter',
                           dict(name='Normalization',
                                value='1.0',
                                free='0',
                                max='1E3',
                                min='1E-3',
                                scale='1.0'))

        return el
    
                
    def get_source_by_name(self,name):
        """Retrieve source by name."""

        if name in self._src_index:
            return self._srcs[self._src_index[name]]
        else:
            raise Exception('No source matching name: ',name)

    def get_nearby_sources(self,name,dist,min_dist=None,
                           selection='circle'):
        
        if dist is None: dist = 180.0
        
        src = self.get_source_by_name(name)
        return self.get_sources_by_position(src['RAJ2000'],
                                            src['DEJ2000'],
                                            dist,min_dist,selection)

    def get_sources_by_position(self,ra,dec,dist,min_dist=None,
                                selection='circle'):
        """Retrieve sources within a certain angular distance of an
        (ra,dec) coordinate.  This function currently supports two
        types of geometric selections: circle and roi.  The circle
        selection finds all sources within a circle of radius dist.
        The roi selection finds sources within a square box of R x
        R where R = 2 x dist.

        Parameters
        ----------

        ra : float

        dec : float

        dist : float
        
        """

        if dist is None: dist = 180.
        
        x = lonlat_to_xyz(np.radians(ra),np.radians(dec))
        costh = np.sum(x[:,np.newaxis]*self._src_radec,axis=0)        
        costh[costh>1.0] = 1.0
        costh[costh<-1.0] = -1.0
        radius = np.arccos(costh)
        
        if selection == 'circle':                    
            dtheta = radius            
        elif selection == 'roi':
            ra0, dec0 = xyz_to_lonlat(self._src_radec)
            dtheta = get_dist_to_edge(np.radians(ra),np.radians(dec),
                                      ra0,dec0,np.radians(dist))
            dtheta += np.radians(dist)            
        else:
            raise Exception('Unrecognized selection type: ' + selection)
        
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
                    src_dict['Spatial_Filename'] = os.path.join(self.config['extdir'],'Templates',
                                                                src_dict['Spatial_Filename'])
            
            src_dict['SpectrumType'] = src_dict['SpectrumType'].strip()
            if src_dict['SpectrumType'] == 'PLExpCutoff':
                src_dict['SpectrumType'] = 'PLSuperExpCutoff'
            
            src = Source(src_dict,extended=extflag)
            self._srcs.append(src)

        self.build_src_index()

#            cat.load_source(ROIManagerSource(src))

    def build_src_index(self):
        """Build an indices for fast lookup of a source given its name
        or coordinates."""
        
        self._src_index = {}
        nsrc = len(self._srcs)

        self._src_radec = np.zeros(shape=(3,nsrc))
        self._src_radius = np.zeros(nsrc)

        for i, s in enumerate(self._srcs):
 
            self._src_radec[:,i] = s.radec
            self._src_radius[i] = s.separation(lonlat_to_xyz(self._radec[0],
                                                             self._radec[1]))
            for c in ROIManager.src_name_cols:
                if not c in s: continue
                name = s[c].strip()
                self._src_index[name] = i
                self._src_index[name.replace(' ','')] = i
                self._src_index[name.replace(' ','').lower()] = i

    
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
#            if isodiff is not None:
#                ROIManager.create_isotropic(s,root)            
#            elif isinstance(s,IsoSource):
#                ROIManager.create_isotropic(s,root)
#            elif isinstance(s,MapCubeSource):
#                ROIManager.create_mapcube(s,root)
#            else:
#                raise Exception('Unkown diffuse source type: ' + type(s))
                
        output_file = open(xmlfile,'w')
        output_file.write(prettify_xml(root))

    def load_xml(self,xmlfile):
        """Load sources from an XML file."""

        if not os.path.isfile(xmlfile):
            xmlfile = os.path.join(fermipy.PACKAGE_ROOT,'catalogs',xmlfile)
        
        root = ElementTree.ElementTree(file=xmlfile).getroot()

        for s in root.findall('source'):

            src = Source.load_from_xml(s,extdir=self.config['extdir'])
            if isinstance(src,Source):
                self._srcs.append(src)
            else:
                self._diffuse_srcs.append(src)

        self.build_src_index()

if __name__ == '__main__':

    
    roi = ROIManager()


    roi.load_fits('gll_fssc_psc_v14.fit')


    src = roi.get_source_by_name('lmc')


    import pprint
    pprint.pprint(src.data)

    print src

    srcs = roi.get_nearby_sources('lmc',10.0)

#    for s in srcs:        
#        print s.name, s.associations, s.separation(src)

    roi.create_roi_from_source('test.xml','lmc','test','test',90.0)
