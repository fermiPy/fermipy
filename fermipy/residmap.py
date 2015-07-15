
import copy
import os
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from astropy import wcs 

import fermipy.defaults as defaults
from fermipy.utils import write_fits_image, AnalysisBase, make_gaussian_kernel
from fermipy.logger import Logger, StreamLogger
from fermipy.logger import logLevel as ll

def poisson_lnl(nc,mu):
    nc = np.array(nc,ndmin=1)
    mu = np.array(mu,ndmin=1)

    shape = max(nc.shape,mu.shape)

    lnl = np.zeros(shape)
    mu = mu*np.ones(shape)
    nc = nc*np.ones(shape)

    msk = nc>0

    lnl[msk] = nc[msk]*np.log(mu[msk])-mu[msk]
    lnl[~msk] = -mu[~msk]
    return lnl

def smooth(m,k,cpix,mode='constant',threshold=0.01):

    from scipy import ndimage
    
    o = np.zeros(m.shape)
    for i in range(m.shape[0]):

        ks = k[i,:,:]

        mx = ks[cpix[0],:] > ks[cpix[0],cpix[1]]*threshold
        my = ks[:,cpix[1]] > ks[cpix[0],cpix[1]]*threshold

        nx = max(3,np.round(np.sum(mx)/2.))
        ny = max(3,np.round(np.sum(my)/2.))
        
        sx = slice(cpix[0]-nx,cpix[0]+nx+1)
        sy = slice(cpix[1]-ny,cpix[1]+ny+1)

        ks = ks[sx,sy]
        
        origin=[0,0]        
        if ks.shape[0]%2==0: origin[0] += 1
        if ks.shape[1]%2==0: origin[1] += 1
            
        o[i,:,:] = ndimage.convolve(m[i,:,:],ks,mode=mode,
                                    origin=origin,cval=0.0)

#    o /= np.sum(k**2)
    return o

def create_model_name(src,spatial_type):

    o = spatial_type

    if spatial_type == 'gaussian':
        o += '_s%04.2f'%src['SpatialWidth']
    
    if src['SpectrumType'] == 'PowerLaw':
        o += '_powerlaw_%04.2f'%src['Index']

    return o

class ResidMapGenerator(AnalysisBase):
    """This class generates spatial residual maps from the difference
    of data and model maps smoothed with a user-defined
    spatial/spectral template.  The resulting map of source
    significance can be interpreted in the same way as the TS map (the
    likelihood of a source at the given location).  The algorithm
    approximates the best-fit source amplitude that would be derived
    from a least-squares fit to the data."""

    defaults = dict(defaults.residmap.items(),
                    fileio=defaults.fileio,
                    logging=defaults.logging)
    
    def __init__(self,config,gta,**kwargs):
        AnalysisBase.__init__(self,config,**kwargs)        
        self._gta = gta
        self._maps = {}
        
        self.logger = Logger.get(self.__class__.__name__,
                                 self.config['fileio']['logfile'],
                                 ll(self.config['logging']['verbosity']))
        

    def get_source_mask(self,name,kernel=None):

        sm = []
        zs = 0
        for c in self._gta.components:
            z = c.modelCountsMap(name).astype('float')
            if kernel is not None:
                shape = (z.shape[0],) + kernel.shape 
                z = np.apply_over_axes(np.sum,z,axes=[1,2])*np.ones(shape)*kernel[np.newaxis,:,:]
                zs += np.sum(z)
            else:
                zs += np.sum(z)

            sm.append(z)
            
        sm2 = 0
        for i, m in enumerate(sm):
            sm[i] /= zs
            sm2 += np.sum(sm[i]**2)
        
        for i, m in enumerate(sm):
            sm[i] /= sm2

        return sm

    def run(self,prefix):

        for m in self.config['models']:
            self.logger.info('Generating Residual map')
            self.logger.info(m)
            self.make_residual_map(copy.deepcopy(m),prefix)
    
    def make_residual_map(self,src_dict,prefix):
        
        # Put the test source at the pixel closest to the ROI center
        xpix, ypix = (np.round((self._gta.npix-1.0)/2.),
                      np.round((self._gta.npix-1.0)/2.))
        cpix = np.array([xpix,ypix])
        
        w = wcs.WCS(self._gta._wcs.to_header(),naxis=[1,2])        
        radec = w.wcs_pix2world(xpix,ypix,0)
        src_dict['ra'] = radec[0]
        src_dict['dec'] = radec[1]
        src_dict.setdefault('SpatialType','PointSource')
        src_dict.setdefault('SpatialWidth',0.3)
        
        kernel = None
        
        if src_dict['SpatialType'] == 'Gaussian':
            src_dict['SpatialType'] = 'PointSource'
            kernel = make_gaussian_kernel(src_dict['SpatialWidth'],
                                          cdelt=0.1,npix=101)
            kernel /= np.sum(kernel)
            cpix = [50,50]
            spatial_type = 'gaussian'
        elif src_dict['SpatialType'] == 'PointSource':
            spatial_type = 'ptsrc'
            
        self._gta.add_source('testsource',src_dict)        
        src = self._gta.roi.get_source_by_name('testsource')
        modelname = create_model_name(src,spatial_type)
        
        enumbins = self._gta.enumbins
        npix = self._gta.components[0].npix

        mmst = np.zeros((npix,npix))
        cmst = np.zeros((npix,npix))
        
        sm = self.get_source_mask('testsource',kernel)
        ts = np.zeros((npix,npix))
        sigma = np.zeros((npix,npix))
        excess = np.zeros((npix,npix))

        for i, c in enumerate(self._gta.components):
            
            mc = c.modelCountsMap().astype('float')
            cc = c.countsMap().astype('float')
            
            ccs = smooth(cc,sm[i],cpix)
            mcs = smooth(mc,sm[i],cpix)
            cms = np.sum(ccs,axis=0)
            mms = np.sum(mcs,axis=0)
            
            cmst += cms
            mmst += mms
            
            cts = 2.0*(poisson_lnl(cms,cms) - poisson_lnl(cms,mms))
            excess += cms - mms
            
        ts = 2.0*(poisson_lnl(cmst,cmst) - poisson_lnl(cmst,mmst))
        sigma = np.sqrt(ts)
        sigma[excess<0] *= -1
        
        sigma_map_file = os.path.join(self.config['fileio']['workdir'],
                                      '%s_residmap_%s_sigma.fits'%(prefix,modelname))

        data_map_file = os.path.join(self.config['fileio']['workdir'],
                                     '%s_residmap_%s_data.fits'%(prefix,modelname))

        model_map_file = os.path.join(self.config['fileio']['workdir'],
                                      '%s_residmap_%s_model.fits'%(prefix,modelname))

        excess_map_file = os.path.join(self.config['fileio']['workdir'],
                                       '%s_residmap_%s_excess.fits'%(prefix,modelname))
        
        write_fits_image(sigma,w,sigma_map_file)
        write_fits_image(cmst,w,data_map_file)
        write_fits_image(mmst,w,model_map_file)
        write_fits_image(excess,w,excess_map_file)
                       
        self._gta.delete_source('testsource')

        self._maps[modelname] = {
            'wcs'    : w,
            'sigma'  : sigma,
            'model'  : mmst,
            'data'   : cmst,
            'excess' : excess }
        
        self._gta._roi_model['roi']['residmap'][modelname] = {
            'sigma'  : os.path.basename(sigma_map_file),
            'model'  : os.path.basename(model_map_file),
            'data'   : os.path.basename(data_map_file),
            'excess' : os.path.basename(excess_map_file) }
            
    def run_lnl(self):
        
        cpix = 50
        npix = self._gta.components[0].npix

        mmt = np.zeros((npix,npix,enumbins))
        cmt = np.zeros((npix,npix,enumbins))
        
        sm = self.get_source_mask()
        ts = np.zeros((npix,npix))
        sigma = np.zeros((npix,npix))
        
        for i, c in enumerate(self._gta.components):
            mm = c.modelCountsMap().astype('float')
            cm = c.countsMap().astype('float')

            cms = smooth(cm,sm[i])
            mms = smooth(mm,sm[i])

            ct = np.sum(cms,axis=2)
            mt = np.sum(mms,axis=2)
            excess = ct-mt
            ts = 2.0*(poisson_lnl(ct,ct) - poisson_lnl(ct,mt))
            
            for j in range(10,npix):
                for k in range(10,npix):
                    continue

                    
                    nx = min(cpix+j,npix) - max(cpix-j,0)
                    ny = min(cpix+k,npix) - max(cpix-k,0)
                    
                    m = copy.copy(mm)
                    m0 = np.zeros((npix,npix,20))
                    sx0 = slice(max(j-cpix,0),min(j-cpix+npix,npix))
                    sx1 = slice(max(cpix-j,0),min(cpix-j+npix,npix))
                    sy0 = slice(max(k-cpix,0),min(k-cpix+npix,npix))
                    sy1 = slice(max(cpix-k,0),min(cpix-k+npix,npix))

                    if excess[j,k] < 0:
                        s = sm[i][sx1,sy1,:]*0.0
                    else:
                        s = sm[i][sx1,sy1,:]*excess[j,k]
                    
                    m[sx0,sy0,:] += s
                    m0[sx0,sy0,:] += s

#                    plt.figure()
#                    plt.imshow(np.sum(m,axis=2),interpolation='nearest',vmax=10)
#                    plt.figure()
#                    plt.imshow(np.sum(m0,axis=2),interpolation='nearest',vmax=10)                    
#                    plt.show()

                    lnl0 = np.sum(poisson_lnl(cm,mm))
                    lnl1 = np.sum(poisson_lnl(cm,m))
                    
                    dlnl = lnl1-lnl0

                    print '%5i %5i %10.3f %10.3f %10.3f %10.3f'%(j,k,lnl0, lnl1, dlnl,excess[j,k])
                    print 2*dlnl, ts[j,k] 
            
#            

            sigma = np.sqrt(ts)
            sigma[excess<=0] *= -1
            
        return sigma, excess, cms, mms

            
