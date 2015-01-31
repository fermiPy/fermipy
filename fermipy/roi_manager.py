import defaults 
from utils import *
import pyfits

import xml.etree.cElementTree as et


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


def get_dist_to_edge(lon0,lat0,lon1,lat1,width):

    x,y = project(lon0,lat0,lon1,lat1)
    delta_edge = np.array([np.abs(x) - width,np.abs(y) - width])
    dtheta = np.max(delta_edge,axis=0)
    return dtheta
    
class IsoSource(object):

    def __init__(self,filefunction,name,norm=1.0):
        self._norm = norm
        self._filefunction = filefunction
        self._name = name

    @property
    def name(self):
        return self._name        
        
    @property
    def filefunction(self):
        return self._filefunction
        
class MapCubeSource(object):

    def __init__(self,mapcube,name,norm=1.0):
        self._norm = norm
        self._mapcube = mapcube
        self._name = name

    @property
    def name(self):
        return self._name        
        
    @property
    def mapcube(self):
        return self._mapcube
        
class Source(object):

    def __init__(self,col_data=None,radec=None,extended=False):

        if col_data is None:            
            self._col_data = {}
        else:
            self._col_data = col_data

        self._radec = radec

        self._names = []
        self._names_dict = {}
        for k in ROIManager.src_name_cols:

            if not k in self._col_data: continue

            name = self._col_data[k].strip()
            if name != '':  self._names.append(name)

            self._names_dict[k] = name

        self._extended=extended

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

    def __contains__(self,key):
        return key in self._col_data

    def __getitem__(self,key):
        return self._col_data[key]

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
        """Return the center of the ROI."""        
        return self._radec

    def load_diffuse_srcs(self):
        self._diffuse_srcs = []
        
        if self.config['isodiff'] is not None:
            self._diffuse_srcs.append(IsoSource(self.config['isodiff'],'isodiff'))

        if self.config['galdiff'] is not None:
            self._diffuse_srcs.append(MapCubeSource(self.config['galdiff'],'galdiff'))
            
    def load(self):

        self._srcs = []

        self.load_diffuse_srcs()
            
        for c in self.config['catalogs']:
            self.load_fits(c)

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
    
        
    @staticmethod
    def create_powerlaw(src,root):

        if src['Flux_Density'] > 0:        
            scale = np.round(np.log10(1./src['Flux_Density']))
        else:
            scale = 0.0
            
        value = src['Flux_Density']*10**scale
                
        create_xml_element(root,'parameter',
                           dict(name='Prefactor',
                                free='0',
                                min='0.01',
                                max='100.0',
                                value=str(value),
                                scale=str(10**-scale)))

        create_xml_element(root,'parameter',
                           dict(name='Index',
                                free='0',
                                min='-5.0',
                                max='5.0',
                                value=str(src['Spectral_Index']),
                                scale=str(-1.0)))
        
        create_xml_element(root,'parameter',
                           dict(name='Scale',
                                free='0',
                                min=str(src['Pivot_Energy']),
                                max=str(src['Pivot_Energy']),
                                value=str(src['Pivot_Energy']),
                                scale=str(1.0)))

    @staticmethod
    def create_logparabola(src,root):

        norm_scale = np.round(np.log10(1./src['Flux_Density']))
        norm_value = src['Flux_Density']*10**norm_scale

        eb_scale = np.round(np.log10(1./src['Pivot_Energy']))
        eb_value = src['Pivot_Energy']*10**eb_scale
        
        create_xml_element(root,'parameter',
                           dict(name='norm',
                                free='0',
                                min='0.01',
                                max='100.0',
                                value=str(norm_value),
                                scale=str(10**-norm_scale)))

        create_xml_element(root,'parameter',
                           dict(name='alpha',
                                free='0',
                                min='-5.0',
                                max='5.0',
                                value=str(src['Spectral_Index']),
                                scale=str(1.0)))

        create_xml_element(root,'parameter',
                           dict(name='beta',
                                free='0',
                                min='0.0',
                                max='5.0',
                                value=str(src['beta']),
                                scale=str(1.0)))

        
        create_xml_element(root,'parameter',
                           dict(name='Eb',
                                free='0',
                                min='0.01',
                                max='100.0',
                                value=str(eb_value),
                                scale=str(10**-eb_scale)))
        
    @staticmethod
    def create_plsuperexpcutoff(src,root):

        norm_scale = np.round(np.log10(1./src['Flux_Density']))
        norm_value = src['Flux_Density']*10**norm_scale

        eb_scale = np.round(np.log10(1./src['Pivot_Energy']))
        eb_value = src['Pivot_Energy']*10**eb_scale
        
        create_xml_element(root,'parameter',
                           dict(name='norm',
                                free='0',
                                min='0.01',
                                max='100.0',
                                value=str(norm_value),
                                scale=str(10**-norm_scale)))

        create_xml_element(root,'parameter',
                           dict(name='alpha',
                                free='0',
                                min='-5.0',
                                max='5.0',
                                value=str(src['Spectral_Index']),
                                scale=str(1.0)))

        create_xml_element(root,'parameter',
                           dict(name='beta',
                                free='0',
                                min='0.0',
                                max='5.0',
                                value=str(src['beta']),
                                scale=str(1.0)))
        
        create_xml_element(root,'parameter',
                           dict(name='Eb',
                                free='0',
                                min='0.01',
                                max='100.0',
                                value=str(eb_value),
                                scale=str(10**-eb_scale)))

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
        R where R = 2 x dist."""

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

            phi = np.radians(src_dict['RAJ2000'])
            theta = np.pi/2.-np.radians(src_dict['DEJ2000'])

            src_radec = np.array([np.sin(theta)*np.cos(phi),
                                  np.sin(theta)*np.sin(phi),
                                  np.cos(theta)])

            src = Source(src_dict,src_radec,extended=extflag)
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
            self._src_radius[i] = s.separation(lonlat_to_xyz(self._radec[0],self._radec[1]))
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
        
        root = et.Element('source_library')
        root.set('title','source_library')

        for s in self._srcs:

            if not s.extended:
                source_element = create_xml_element(root,'source',
                                                    dict(name=s['Source_Name'],
                                                         type='PointSource'))

                spat_el = et.SubElement(source_element,'spatialModel')
                spat_el.set('type','SkyDirFunction')

                create_xml_element(spat_el,'parameter',
                                   dict(name = 'RA',value = str(s['RAJ2000']),
                                        free='0',min='-360.0',max='360.0',
                                        scale='1.0'))

                create_xml_element(spat_el,'parameter',
                                   dict(name = 'DEC',value = str(s['DEJ2000']),
                                        free='0',min='-90.0',max='90.0',
                                        scale='1.0'))

            else:
                source_element = create_xml_element(root,'source',
                                                    dict(name=s['Source_Name'],
                                                         type='DiffuseSource'))

                spat_el = create_xml_element(source_element,'spatialModel',
                                             dict(map_based_integral='True',
                                                  type='SpatialMap',
                                                  file=s['Spatial_Filename']))

                create_xml_element(spat_el,'parameter',
                                   dict(name = 'Prefactor',value = '1',
                                        free='0',min='0.001',max='1000',
                                        scale='1.0'))

            spec_element = et.SubElement(source_element,'spectrum')

            stype = s['SpectrumType'].strip()            
            spec_element.set('type',stype)

            if stype == 'PowerLaw':
                ROIManager.create_powerlaw(s,spec_element)
            elif stype == 'LogParabola':
                ROIManager.create_logparabola(s,spec_element)
            elif stype == 'PLSuperExpCutoff' or stype == 'PLExpCutoff':
                spec_element.set('type','PLSuperExpCutoff')
                ROIManager.create_plsuperexpcutoff(s,spec_element)
            else:
                raise Exception('Unrecognized spectral type: ' + stype)
            

        for s in self._diffuse_srcs:

            if isodiff is not None:
                ROIManager.create_isotropic(s,root)            
            elif isinstance(s,IsoSource):
                ROIManager.create_isotropic(s,root)
            elif isinstance(s,MapCubeSource):
                ROIManager.create_mapcube(s,root)
            else:
                raise Exception('Unkown diffuse source type: ' + type(s))
                
        output_file = open(xmlfile,'w')
        output_file.write(prettify_xml(root))

    def load_xml(self):
        """Load sources from an XML file."""

        pass


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
