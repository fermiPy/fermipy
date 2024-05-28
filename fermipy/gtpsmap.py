#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
###################################################################
#
# gtpsmap: check data/model agreement in Fermi-LAT analysis
#
# This python script compares data and model 3D maps by computing a PS map.
# It has been written by Philippe Bruel (LLR, CNRS/IN2P3, IPP Paris, France), based on a new method fully described in:
# https://www.aanda.org/articles/aa/full_html/2021/12/aa41553-21/aa41553-21.html
#
# The main inputs of the script are:
# - the data 3D count map (output of gtbin) or the list of data 3D count maps for multi-component analyses
# - the model 3D count map (output of gtmodel or gta.write_model_map() for fermipy users) or the list of model 3D count maps for multi-component analyses
# - if you use weighted likelihood to perform the fit, then you also have to provide the 3D weight map
# The user also has to provide the energy interval and binning (emin, emax, nbinloge) that are used to compute PS (which can be different from the 3D map binning). The recommendation is to set nbinloge in order to have a logE bin width in the range 0.2-0.3. 
#
# Usage:
# gtpsmap.py --cmap <data 3D map> --mmap <model 3D map> --wmap <weights 3D map (optional)> --emin <minimum energy/MeV> --emax <maximum energy/MeV> --nbinloge <number of log10 energy bins> --outfile <output file>
#
# Example for one component between 100 MeV and 100 GeV:
# gtpsmap.py --cmap ccube_00.fits --mmap mcube_00.fits --emin 100 --emax 100000 --nbinloge 15 --outfile psmap.fits
# Example for four components between 100 MeV and 100 GeV:
# gtpsmap.py --cmap ccube_00.fits:ccube_01.fits:ccube_02.fits:ccube_03.fits --mmap mcube_00.fits:mcube_01.fits:mcube_02.fits:mcube_03.fits --emin 100 --emax 100000 --nbinloge 15 --outfile psmap.fits
#
# The output file contains:
# - primary hdu: 2D map of PS
# - hdu #1 ('PS in sigma'): deviation in sigma units (with sigma = sqrt(2) * ErfInv(1-10^(-|PS|)))
#
# The definition of PS at each pixel of the map is:
# - |PS| = -log10(deviation probability), where the deviation probability is measured on the PSF-like integrated count data and model spectra
# - sign(PS) = sign of sum(data_k-model_k)/sqrt(max(1,model_k)/weight_k), where the sum runs over the spectral bins
#
# Here are some useful PS<->sigma conversions:
# - PS -> sigma: 3/4/5/6/7/8 -> 3.29/3.89/4.42/4.89/5.33/5.73
# - sigma -> PS: 3/4/5/6 -> 2.57/4.20/6.24/8.70
#
# The other (optional) parameters of the python script are:
# --psfpar0 <PSF-like par0; 4.0 deg> --psfpar1 <PSF-like par1; 100 MeV>  --psfpar2 <PSF-like par2; 0.9>  --psfpar3 <PSF-like par3; 0.1deg>
#   these parameters define the PSF-like integration radius = sqrt(p0^2*pow(E/p1,-2*p2) + p3^2)
# --psfpar0lst: in the case of multiple components, the user can set psfpar0 independently for each component, e.g. --psfpar0lst 2.0:3.0:4.0:5.0
# --chatter <output verbosity; 1>
# --prob_epsilon <precision parameter; 1e-7>
#   This parameter defines the k-interval over which the Poisson probability of each spectral bin is considered
#   when computing the log-likelihood distribution (the k-interval is such that ΣPoisson(k,model_i) = 1-prob_epsilon).
#   The default value 1e-7 provides a 1% precision on PS up to PS=20. If you want a better precision for larger PS values,
#   you need a smaller prob_epsilon (1e-11 provides a 5% precision up to 250) but that makes the script slower.
#
###################################################################

import sys, getopt
import numpy as np
import re
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
import scipy.signal
from scipy.stats import poisson
from scipy.special import gammainc
from scipy.special import gammaincc
from scipy.special import erfcinv

#from gammapy.maps import WcsNDMap, WcsGeom

def getDeviationProbability(datcts, modcts, weights, maxpoissoncount, prob_epsilon, nbinpdf, scaleaxis):
    # datcts = array of data counts
    # modcts = array of model counts
    # weights = array of weights
    # for simplicty's sake, LL (LogLikelihood) stands for -LL
    
    datcts=np.rint(datcts)

    msk = modcts>0
    zmodcts = modcts[msk]
    zdatcts = datcts[msk]
    zweights = weights[msk]

    # sort model counts in descending order
    sortindex = np.argsort(-zmodcts)
    smodcts = zmodcts[sortindex]
    sdatcts = zdatcts[sortindex]
    sweights = zweights[sortindex]

    # compute the sum of the residuals to estimate the sign of the deviation
    allres = (sdatcts-smodcts)/np.sqrt(np.maximum(smodcts,1)/sweights)
    
    # select the bins in gaussian mode
    mskgaus = smodcts > maxpoissoncount
    gmodcts = smodcts[mskgaus]
    gdatcts = sdatcts[mskgaus]
    gweights = sweights[mskgaus]
    gchi2 = np.power(gdatcts-gmodcts,2)/gmodcts*gweights
    gdat_LL = 0.5*np.sum(gchi2)
    ngaus = len(gmodcts)
    # return simple chi2 probability if all the bins are gaussian
    if ngaus==len(smodcts):        
        # second argument in gammaincc is gdat_LL = chi2/2
        return gammaincc(ngaus/2.0,gdat_LL), np.sum(allres)

    pmodcts = smodcts[~mskgaus]
    pdatcts = sdatcts[~mskgaus]
    pweights = sweights[~mskgaus]

    # compute loglikelihood for data counts
    dat_PP = poisson.pmf(pdatcts,pmodcts)
    checkmin = np.full(len(pdatcts),1e-323)
    dat_PP = np.maximum(dat_PP,checkmin)
    dat_LL = np.sum(-pweights*np.log(dat_PP))
    dat_LL += gdat_LL
    
    # for each bin, get [k_min,k_max] interval such that sum_{k_min,k_max}(poisson(k,model_counts)) = 1-prob_epsilon
    kmin = np.zeros(len(pmodcts),dtype=int)
    kmax = np.zeros(len(pmodcts),dtype=int)
    msk = pmodcts < 5
    kmin[msk] = 0
    kmax[msk] = np.rint(0.715029+0.049825*np.log10(prob_epsilon)+0.011768*pow(np.log10(prob_epsilon),2) + (-0.308206+-1.309547*np.log10(prob_epsilon)+-0.028455*pow(np.log10(prob_epsilon),2)) * np.exp((1.881824+0.117058*np.log10(prob_epsilon)+0.004208*pow(np.log10(prob_epsilon),2))*np.maximum(np.log10(pmodcts[msk]),-4))+1);
    #
    offsetlo = np.rint(-0.523564-0.75129*np.log10(prob_epsilon))
    offsetup = np.rint(-1.09374-0.716202*np.log10(prob_epsilon))
    kmin[~msk] = np.rint(pmodcts[~msk]-np.sqrt(pmodcts[~msk])*erfcinv(prob_epsilon)*np.sqrt(2)+offsetlo)
    kmax[~msk] = np.rint(pmodcts[~msk]+np.sqrt(pmodcts[~msk])*erfcinv(prob_epsilon)*np.sqrt(2)+offsetup)
    msk = kmin<0
    kmin[msk] = 0
    
    # prepare the binning of the LL histogram
    hnbin = nbinpdf*len(smodcts)
    hxmax = scaleaxis*(ngaus+np.sum(pweights))
    hstep = hxmax/hnbin

    result = np.zeros(hnbin,dtype=float)
    hbinedge = np.linspace(0,hxmax+hstep,num=hnbin+1,endpoint=False)
    hbincen = 0.5*(hbinedge[1:] + hbinedge[:-1])

    if ngaus>0:
        # compute the histogram of LL values with the incomplete gamma function
        # setting the incomplete gamma parameters so that it corresponds to the chi2 distribution with ngaus degrees of freedom
        # the theta parameter = 2, so the second argument of the gamma function is 2*x/theta=x
        result = gammainc(ngaus/2.0,hbinedge[1:])-gammainc(ngaus/2.0,hbinedge[:-1])
        resultc = np.cumsum(result)
        msk = resultc>1-prob_epsilon
        result[msk] = 0
    else:
        # prepare the histogram of LL values corresponding to the first bin with model counts = pmodcts[0]
        vecdat0 = np.arange(kmin[0],kmax[0]+1)
        vecPP0 = poisson.pmf(vecdat0,pmodcts[0])
        vecLL0 = -pweights[0]*np.log(vecPP0)
        result, hbinedge = np.histogram(vecLL0,bins=hnbin,range=(0,hxmax),weights=vecPP0)

    # build the LL distribution iteratively, starting from the first Poissonian bin (if ngaus>0) or the second bin (if ngaus==0)
    istart = int(0)
    if ngaus==0:
        istart = 1
    for i in np.arange(istart,len(pmodcts)):
        vecdat = np.arange(kmin[i],kmax[i]+1)
        # get prob and LL values for [kmin,kmax]
        vecLLw = poisson.logpmf(vecdat,pmodcts[i])
        vecPP = np.exp(vecLLw)
        vecLLw *= (-pweights[i]/hstep)
        # get shift = bin # of LL value along the x-axis
        binshift = np.rint(vecLLw,out=vecdat,casting='unsafe')

        # shift the current LL histogram by binshift to fill the 2d array
        # accumulate the 2d array as we go so we never have to actually
        # allocate it
        comp_binshift = len(result)-binshift
        tmp = np.zeros_like(result)
        for j in range(0,len(vecPP)):
            tmp[binshift[j]:] += vecPP[j]*result[:comp_binshift[j]]

        # project onto the x-axis to get the new LL histogram
        result = tmp

    # get the largest LL value of the LL histogram
    mskhist = result > 0
    xnz = hbincen[mskhist]
    LLmax = xnz[-1]
    # set data LL to the maximum one if needed
    if dat_LL > LLmax:
        dat_LL = LLmax
    ibin = int(dat_LL/hstep)
    pdat = np.sum(result[ibin:])
    #print('pdat %f PS %f' % (pdat,-np.log10(pdat)))

    return pdat, np.sum(allres)

def convolve_map(m, k, cpix, threshold=0.001):
    # piece of code copied from fermipy residmap (see https://fermipy.readthedocs.io/en/latest/)
    o = np.zeros(m.shape,dtype=np.float64)
    ix = int(cpix[0])
    iy = int(cpix[1])

    ks = k
    ms = m

    mx = ks[ix, :] > ks[ix, iy] * threshold
    my = ks[:, iy] > ks[ix, iy] * threshold

    nx = int(np.sum(mx)/2)
    ny = int(np.sum(mx)/2)

    # Ensure that there is an odd number of pixels in the kernel
    # array
    if ix + nx + 1 >= ms.shape[0] or ix - nx < 0:
        nx -= 1
    if iy + ny + 1 >= ms.shape[0] or iy - ny < 0:
        ny -= 1

    sx = slice(ix - nx, ix + nx + 1)
    sy = slice(iy - ny, iy + ny + 1)
    
    ks = ks[sx, sy]
    o = scipy.signal.fftconvolve(ms, ks, mode='same')

    return o

def checkenergyrange(cmapfilename,emin,emax):
    #
    # check that the energy range of the fits file is not outside [emin,emax]

    C0dat_hdulist = fits.open(cmapfilename)
    try:
        ene_hdu = C0dat_hdulist[1]
        ene_table = ene_hdu.data
        energymin = ene_table['E_MIN']
        energymax = ene_table['E_MAX']
        if ene_hdu.header['TUNIT2']=='keV':
            energymin = energymin/1000
            energymax = energymax/1000
    except: # for weight files, assuming MeV
        ene_hdu = C0dat_hdulist[0]
        naxis3 = ene_hdu.header['NAXIS3']
        e0 = ene_hdu.header['CRVAL3']
        estep = ene_hdu.header['CDELT3']
        logestep = np.log10(1+estep/e0)
        loge0 = np.log10(e0)
        axis3 = np.arange(0, naxis3)
        energymin = np.power(10.,loge0+logestep*axis3)
        energymax = np.power(10.,loge0+logestep*(axis3+1))
                
    mskle = energymax>emin
    mskhe = energymin<emax
    C0mskerange = mskle & mskhe
    C0dat_hdulist.close()
    return np.max(C0mskerange)

def get_event_class_type(cmapfilename):
    #
    C0dat_hdulist = fits.open(cmapfilename)
    hdu = C0dat_hdulist[0]
    evtclass = ''
    evttype = ''
    for i in range(1,20):
        key = ('DSTYP%d' %(i))
        value = hdu.header.get(key)
        if value==None:
            break
        if re.search('EVENT_CLASS',value):
            value2 = value.split(",",2)
            evtclass = value2[1]
        if re.search('EVENT_TYPE',value):
            value2 = value.split(",",2)
            evttype = value2[1]

    evtclasstype = ('%s/%s' %(evtclass,evttype))
    C0dat_hdulist.close()
    return evtclasstype

def prepPSFintegmap(componentname, cmapfilename, mmapfilename, wmapfilename, psfpar, emin, emax, chatter):
    #
    # compute the PSF-like integrated maps for data, model and weights

    optweight = 1
    if wmapfilename == '':
        optweight = 0
    
    C0dat_hdulist = fits.open(cmapfilename)
    C0dat_hdu = C0dat_hdulist[0]
    C0dat_map0 = C0dat_hdu.data

    ene_hdu = C0dat_hdulist[1]
    ene_table = ene_hdu.data
    energymin = ene_table['E_MIN']
    energymax = ene_table['E_MAX']
    energycenter = np.sqrt(ene_table['E_MIN']*ene_table['E_MAX'])
    if ene_hdu.header['TUNIT2']=='keV':
        energymin = energymin/1000
        energymax = energymax/1000
        energycenter = energycenter/1000
    C0psflikeroi = np.sqrt(psfpar[0]*psfpar[0]*np.power(psfpar[1]/energymin,2*psfpar[2])+psfpar[3]*psfpar[3])
    mskle = energymax>emin
    mskhe = energymin<emax
    C0mskerange = mskle & mskhe
    if np.max(C0mskerange)==False:
        print('The energy range emin=%f emax=%f is empty. Exiting!' % (emin,emax))
        return 0
    C0selenergymin = energymin[C0mskerange]
    C0selenergymax = energymax[C0mskerange]
    C0logselenergycenter = np.log10(np.sqrt(C0selenergymin*C0selenergymax))

    C0mod_hdulist = fits.open(mmapfilename)
    C0mod_hdu = C0mod_hdulist[0]
    C0mod_map = C0mod_hdu.data

    # the weights of the integrated count spectra correspond to the Ndata-averaged 1/weights over the PSF-like region
    if optweight==1:
        C0weight_hdulist = fits.open(wmapfilename)
        C0weight_map = C0weight_hdulist[0].data
        C0weight_map = 1/C0weight_map

    coordname1 = C0dat_hdu.header['CTYPE1']
    coordname2 = C0dat_hdu.header['CTYPE2']

    naxis1 = C0dat_hdu.header['NAXIS1']
    naxis2 = C0dat_hdu.header['NAXIS2']
    C0naxis3 = C0dat_hdu.header['NAXIS3']
    
    # get 2d part of wcs
    wcs3d = WCS(C0dat_hdu.header,fix=0)
    hdr3D = wcs3d.to_header()
    for key in hdr3D.keys():
        if '3' in key:
            del hdr3D[key]
    hdr3D['WCSAXES'] = 2
    wcs2d = WCS(hdr3D)

    # get the world coordinates of the pixel center array
    axis1 = np.arange(0, naxis1)
    axis2 = np.arange(0, naxis2)
    coord_x, coord_y = np.meshgrid(axis2, axis1, indexing='ij')
    wdir = SkyCoord.from_pixel(coord_x,coord_y,wcs=wcs2d,origin=0,mode='all')
    
    # get the array of the angular distance to the reference direction (map center)
    icenter = int(naxis2/2-1)
    jcenter = int(naxis1/2-1)
    refdir = SkyCoord.from_pixel(icenter,jcenter,wcs=wcs2d,origin=0)
    dist = refdir.separation(wdir)
        
    C0kernel_map=np.zeros((C0naxis3,naxis2,naxis1),dtype=np.float64)
    C0dat_summed_map=np.zeros((C0naxis3,naxis2,naxis1),dtype=np.float64)
    C0mod_summed_map=np.zeros((C0naxis3,naxis2,naxis1),dtype=np.float64)
    C0dat_map=np.zeros((C0naxis3,naxis2,naxis1),dtype=np.float64)
    C0dat_map = C0dat_map0

    # the weights of the integrated count spectra correspond to the Ndata-averaged 1/weights over the PSF-like region
    if optweight:
        C0prod_weightdat_map=C0weight_map*C0dat_map
        C0weight_avesummed_map=np.zeros((C0naxis3,naxis2,naxis1),dtype=np.float64)

    cpix=np.array([icenter,jcenter])
    for k in range(C0naxis3):
        C0kernel_map[k,:,:]=dist.degree < C0psflikeroi[k]
        C0dat_summed_map[k,:,:]=convolve_map(C0dat_map[k,:,:],C0kernel_map[k,:,:],cpix)
        C0mod_summed_map[k,:,:]=convolve_map(C0mod_map[k,:,:],C0kernel_map[k,:,:],cpix)
        if optweight:
            C0weight_avesummed_map[k,:,:]=convolve_map(C0prod_weightdat_map[k,:,:],C0kernel_map[k,:,:],cpix)

    msk = C0dat_summed_map > 0.1
    C0dat_summed_map[~msk] = 0

    # the weights of the integrated count spectra correspond to the Ndata-averaged 1/weights over the PSF-like region
    C0weight_ave_map = np.ones((C0naxis3,naxis2,naxis1),dtype=np.float64)

    if optweight:
        msk = C0dat_summed_map>0
        C0weight_ave_map[msk] = C0weight_avesummed_map[msk]/C0dat_summed_map[msk]
        C0weight_ave_map[msk] = 1/C0weight_ave_map[msk]

    # select energy range
    C0mod_summed_map = C0mod_summed_map[C0mskerange]
    C0dat_summed_map = C0dat_summed_map[C0mskerange]
    C0weight_ave_map = C0weight_ave_map[C0mskerange]
    if chatter>0:
        print("%s: selecting %d energy bins in [%f,%f]" %(componentname,C0mod_summed_map.shape[0],C0selenergymin[0],C0selenergymax[-1]))

    C0dat_hdulist.close()
    C0mod_hdulist.close()
    if optweight==1:
        C0weight_hdulist.close()

    return C0mod_summed_map, C0dat_summed_map, C0weight_ave_map, C0logselenergycenter

def rebin_spectrum(C0logselenergycenter,C0datcounts,C0modcounts,C0weights,hlogebin):
    C0aveinvwgt = C0datcounts/C0weights
    C0datres, hbinedge = np.histogram(C0logselenergycenter,bins=hlogebin,weights=C0datcounts)
    C0modres, hbinedge = np.histogram(C0logselenergycenter,bins=hlogebin,weights=C0modcounts)
    C0aveinvwgtres, hbinedge = np.histogram(C0logselenergycenter,bins=hlogebin,weights=C0aveinvwgt)
    mskdat = C0datres>0
    C0aveinvwgtres[mskdat] = C0aveinvwgtres[mskdat]/C0datres[mskdat]
    C0aveinvwgtres[~mskdat] = 1
    return C0datres,C0modres,C0aveinvwgtres

def run(args):

    o = {} # this sintax is to respect matt syntax where all the output are saved in the dictionary called "o"

    chatter = args['chatter']
    if chatter>0:
        print('Checking input files:')

    cmaplst0 = re.split('[:]',args['cmap'])
    mmaplst0 = re.split('[:]',args['mmap'])
    wmaplst0 = re.split('[:]',args['wmap'])
    par0lst0 = re.split('[:]',args['psfpar0lst'])

    # check if weights are provided
    optweight = True
    if wmaplst0[0] == '':
        optweight = False

    if len(cmaplst0)!=len(mmaplst0):
        print('The numbers of data and model components are not the same. Exiting.')
        return o
    if optweight and len(cmaplst0)!=len(wmaplst0):
        print('The numbers of data and weight components are not the same. Exiting.')
        return o
    if par0lst0[0]!='' and len(par0lst0)!=len(cmaplst0):
        print('The number of parameters listed in psfpar0lst does not match the number of data components. Exiting.')
        return o

    emin = args['emin']
    emax = args['emax']
    nbinloge = args['nbinloge']
    logebinwidth = np.log10(emax/emin)/nbinloge
    if logebinwidth<0.2 or logebinwidth>0.3:
        print('Warning: emin, emax, and nbinloge are such that the log10(E) bin width is %f. It is recommended that it is between 0.2 and 0.3.' %(logebinwidth))
    #if nbinloge<1: nbinloge = np.maximum(1,int((np.log10(emax/emin)/0.2)))
    #print('nbinloge %d' %(nbinloge))
    out_filename    = args['outfile']
    fixedradius     = args['fixedradius']
    psfpar0 = args['psfpar0']
    psfpar1 = args['psfpar1']
    psfpar2 = args['psfpar2']
    psfpar3 = args['psfpar3']
    maxpoissoncount = args['maxpoissoncount']
    prob_epsilon = np.maximum(args['prob_epsilon'],1e-15)
    nbinpdf = args['nbinpdf']
    scaleaxis = args['scaleaxis']
    ipix = args['ipix']
    jpix = args['jpix']
    dpix = args['dpix']
    #write_fits = args['write_fits']

    # Remove the components that are outside the energy range [emin,emax]
    cmaplst = []
    mmaplst = []
    wmaplst = []
    par0lst = []
    for i in range(0,len(cmaplst0)):
        if(checkenergyrange(cmaplst0[i],emin,emax)==True): cmaplst.append(cmaplst0[i])
        if(checkenergyrange(mmaplst0[i],emin,emax)==True): mmaplst.append(mmaplst0[i])
        if optweight:
            if(checkenergyrange(wmaplst0[i],emin,emax)==True): wmaplst.append(wmaplst0[i])
        if par0lst0[0]!='':
            par0lst.append(par0lst0[i])

    if len(cmaplst)==0:
        print('All components are outside the energy range [%f,%f]. Exiting.' %(emin,emax))
        return o
    if len(cmaplst0)!=len(mmaplst0):
        print('The numbers of data and model components after energy range verification are not the same. Exiting.')
        return o
    if optweight and len(cmaplst0)!=len(wmaplst0):
        print('The numbers of data and weight components after energy range verification are not the same. Exiting.')
        return o
    if len(cmaplst)>4:
        print('Considering only the first 4 components.')

    psfpar0c = np.zeros(4,dtype=float)
    for i in range(0,len(cmaplst)):
        psfpar0c[i] = psfpar0
        if(len(par0lst)>0):
            psfpar0c[i] = float(par0lst[i])
    
    optcomponent = np.zeros(4,dtype=int)
    for i in range(0,len(cmaplst)):
        optcomponent[i] = 1
        if optweight==False:
            wmaplst.append('')
        if chatter>0:
            mystr = ('Component%d: data=%s (class/type=%-6s) model=%s' % (i,cmaplst[i],get_event_class_type(cmaplst[i]),mmaplst[i]))
            if optweight:
                mystr = ('%s weight=%s' %(mystr,wmaplst[i]))
            print(mystr)

    # prepare logE histogram on which the fits energy binning will be projected
    hlogebin = np.linspace(np.log10(emin),np.log10(emax),nbinloge+1)
    #print(hlogebin)
    
    # prepare the 
    C0dat_hdulist = fits.open(cmaplst[0])
    C0dat_hdu = C0dat_hdulist[0]
    C0dat_map0 = C0dat_hdu.data

    coordname1 = C0dat_hdu.header['CTYPE1']
    coordname2 = C0dat_hdu.header['CTYPE2']

    naxis1 = C0dat_hdu.header['NAXIS1']
    naxis2 = C0dat_hdu.header['NAXIS2']
    
    # get 2d part of wcs
    wcs3d = WCS(C0dat_hdu.header,fix=0)
    hdr3D = wcs3d.to_header()
    for key in hdr3D.keys():
        if '3' in key:
            del hdr3D[key]
    hdr3D['WCSAXES'] = 2
    wcs2d = WCS(hdr3D)

    # get the world coordinates of the pixel center array
    axis1 = np.arange(0, naxis1)
    axis2 = np.arange(0, naxis2)
    coord_x, coord_y = np.meshgrid(axis2, axis1, indexing='ij')
    wdir = SkyCoord.from_pixel(coord_x,coord_y,wcs=wcs2d,origin=0,mode='all')
    
    # get the array of the angular distance to the reference direction (map center)
    icenter = int(naxis2/2-1)
    jcenter = int(naxis1/2-1)
    refdir = SkyCoord.from_pixel(icenter,jcenter,wcs=wcs2d,origin=0)
    dist = refdir.separation(wdir)

    C0dat_hdulist.close()

    if fixedradius>0:
        for i in range(0,len(cmaplst)):
            psfpar0c[i] = 0
        psfpar3 = fixedradius
        if chatter>0:
            print ('Computing PSF-like summed maps with fixed radius %f deg' % (fixedradius))
    else:
        if chatter>0:
            print ('Computing PSF-like summed maps with energy dependent selection:')
            print ('Component0: (%f deg, %f MeV, %f, %f deg)' % (psfpar0c[0],psfpar1,psfpar2,psfpar3))
            if optcomponent[1]:
                print ('Component1: (%f deg, %f MeV, %f, %f deg)' % (psfpar0c[1],psfpar1,psfpar2,psfpar3))
            if optcomponent[2]:
                print ('Component2: (%f deg, %f MeV, %f, %f deg)' % (psfpar0c[2],psfpar1,psfpar2,psfpar3))
            if optcomponent[3]:
                print ('Component3: (%f deg, %f MeV, %f, %f deg)' % (psfpar0c[3],psfpar1,psfpar2,psfpar3))


    psfpar = np.zeros(4,dtype=float)

    psfpar[0] = psfpar0c[0]; psfpar[1] = psfpar1; psfpar[2] = psfpar2; psfpar[3] = psfpar3
    C0mod_summed_map, C0dat_summed_map, C0weight_ave_map, C0logselenergycenter = prepPSFintegmap('Component0',cmaplst[0],mmaplst[0],wmaplst[0],psfpar,emin,emax,chatter)

    if optcomponent[1]:
        psfpar[0] = psfpar0c[1]; psfpar[1] = psfpar1; psfpar[2] = psfpar2; psfpar[3] = psfpar3
        C1mod_summed_map, C1dat_summed_map, C1weight_ave_map, C1logselenergycenter = prepPSFintegmap('Component1',cmaplst[1],mmaplst[1],wmaplst[1],psfpar,emin,emax,chatter)

    if optcomponent[2]:
        psfpar[0] = psfpar0c[2]; psfpar[1] = psfpar1; psfpar[2] = psfpar2; psfpar[3] = psfpar3
        C2mod_summed_map, C2dat_summed_map, C2weight_ave_map, C2logselenergycenter = prepPSFintegmap('Component2',cmaplst[2],mmaplst[2],wmaplst[2],psfpar,emin,emax,chatter)

    if optcomponent[3]:
        psfpar[0] = psfpar0c[3]; psfpar[1] = psfpar1; psfpar[2] = psfpar2; psfpar[3] = psfpar3
        C3mod_summed_map, C3dat_summed_map, C3weight_ave_map, C3logselenergycenter = prepPSFintegmap('Component3',cmaplst[3],mmaplst[3],wmaplst[3],psfpar,emin,emax,chatter)

    if chatter>0:
        print('Computing the PS map (nbinpdf=%d, scaleaxis=%3.2f, maxpoissoncount=%3.2f, prob_epsilon=%g) between %f MeV and %f MeV with %d bins (log10(E) bin width=%.2f):\n(for each column of the map, the integer part of the maximum value of abs(PS) in sigma is given as one character [1,..,9,A=10,B=11,..,Y=34,Z>=35])' %(nbinpdf,scaleaxis,maxpoissoncount,prob_epsilon,emin,emax,nbinloge,logebinwidth))
        
    psmap      = np.zeros((naxis2,naxis1))
    totresmap  = np.zeros((naxis2,naxis1))

    psmin = 1e9;
    psmax = -1e9;
    imax = 0
    jmax = 0
    imin = 0
    jmin = 0
    psvalstr = ['0','1','2','3','4','5','6','7','8','9']
    psvalstr.extend(['A','B','C','D','E','F','G','H','I','J'])
    psvalstr.extend(['K','L','M','N','O','P','Q','R','S','T'])
    psvalstr.extend(['U','V','W','X','Y','Z'])

    nbnan = int(0)
    psmaxcol = 0
    axisrange1 = range(naxis1)
    axisrange2 = range(naxis2)
    # change range if ipix and jpix are provided
    if ipix>0 and jpix>0 and dpix>=0:
        axisrange1 = range(np.maximum(0,ipix-1-dpix),np.minimum(naxis1,ipix+dpix))
        axisrange2 = range(np.maximum(0,jpix-1-dpix),np.minimum(naxis2,jpix+dpix))
    
    # loop over the pixels and compute PS
    for i in axisrange1:
        psmaxcol = 0;
        for j in axisrange2:
            #
            # project C0 on hlogebin
            C0datres,C0modres,C0aveinvwgtres = rebin_spectrum(C0logselenergycenter,C0dat_summed_map[:,j,i],C0mod_summed_map[:,j,i],C0weight_ave_map[:,j,i],hlogebin)
            datcounts = C0datres
            modcounts = C0modres
            aveinvwgt = C0datres*C0aveinvwgtres # equivalent to dat/(1/invwgt)
            #
            # project C1 on hlogebin
            if optcomponent[1]:
                C1datres,C1modres,C1aveinvwgtres = rebin_spectrum(C1logselenergycenter,C1dat_summed_map[:,j,i],C1mod_summed_map[:,j,i],C1weight_ave_map[:,j,i],hlogebin)
                datcounts = datcounts + C1datres
                modcounts = modcounts + C1modres
                aveinvwgt = aveinvwgt + C1datres*C1aveinvwgtres
            #
            # project C2 on hlogebin
            if optcomponent[2]:
                C2datres,C2modres,C2aveinvwgtres = rebin_spectrum(C2logselenergycenter,C2dat_summed_map[:,j,i],C2mod_summed_map[:,j,i],C2weight_ave_map[:,j,i],hlogebin)
                datcounts = datcounts + C2datres
                modcounts = modcounts + C2modres
                aveinvwgt = aveinvwgt + C2datres*C2aveinvwgtres
            #
            # project C3 on hlogebin
            if optcomponent[3]:
                C3datres,C3modres,C3aveinvwgtres = rebin_spectrum(C3logselenergycenter,C3dat_summed_map[:,j,i],C3mod_summed_map[:,j,i],C3weight_ave_map[:,j,i],hlogebin)
                datcounts = datcounts + C3datres
                modcounts = modcounts + C3modres
                aveinvwgt = aveinvwgt + C3datres*C3aveinvwgtres
            #
            mskdat = datcounts>0
            aveinvwgt[mskdat] = aveinvwgt[mskdat]/datcounts[mskdat]
            aveinvwgt[~mskdat] = 1
            weights = 1/aveinvwgt
            #
            prob, totres = getDeviationProbability(datcounts,modcounts,weights,maxpoissoncount,prob_epsilon,nbinpdf,scaleaxis)
            if np.isnan(prob):
                nbnan = nbnan+1
            if prob<1e-311:
                prob = 1e-311
            ps = -np.log10(prob)
            #
            if ps>psmaxcol:
                psmaxcol = ps
            if totres<0:
                ps = -ps
            totresmap[j,i] = totres
            psmap[j,i] = ps
            if ps<psmin:
                psmin = ps
                imin = i
                jmin = j
            if ps>psmax:
                psmax = ps
                imax = i
                jmax = j
        psmaxcol = np.sqrt(2) * erfcinv(np.power(10., -np.abs(psmaxcol)))
        ipsvalstr = int(psmaxcol)
        if ipsvalstr>35:
            ipsvalstr = 35
        mycom = '%s' % (psvalstr[ipsvalstr])
        if chatter>0:
            sys.stdout.write(mycom)
            sys.stdout.flush()

    if chatter>0:
        print ('')

    if nbnan>0 and chatter>0:
        print ('')
        print ('-------------------------------------> Warning: there are %d pixels with PS=nan.' % (nbnan))
        print ('')

    psmindir = SkyCoord.from_pixel(imin,jmin,wcs=wcs2d,origin=0)
    psmaxdir = SkyCoord.from_pixel(imax,jmax,wcs=wcs2d,origin=0)

    psminsig = np.sign(psmin) * np.sqrt(2) * erfcinv(np.power(10., -np.abs(psmin)))
    psmaxsig = np.sign(psmax) * np.sqrt(2) * erfcinv(np.power(10., -np.abs(psmax)))

    if psmindir.frame.name=='galactic':
        coordx=psmindir.l.to_string(decimal=True)
        coordy=psmindir.b.to_string(decimal=True)
    else:
        coordx=psmindir.ra.to_string(decimal=True)
        coordy=psmindir.dec.to_string(decimal=True)
    if chatter>0:
        print ('Minimum PS: %5.3f (%5.3f sigma) found at the position (%s,%s)=(%s,%s) [fits pixel (%d,%d)]' % (psmin,psminsig,coordname1,coordname2,coordx,coordy,imin+1,jmin+1))
    
    if psmaxdir.frame.name=='galactic':
        coordx=psmaxdir.l.to_string(decimal=True)
        coordy=psmaxdir.b.to_string(decimal=True)
    else:
        coordx=psmaxdir.ra.to_string(decimal=True)
        coordy=psmaxdir.dec.to_string(decimal=True)
    if chatter>0:
        print ('Maximum PS: %5.3f (%5.3f sigma) found at the position (%s,%s)=(%s,%s) [fits pixel (%d,%d)]' % (psmax,psmaxsig,coordname1,coordname2,coordx,coordy,imax+1,jmax+1))

    o['psmax'] = psmax
    o['psmaxsigma'] = psmaxsig
    o['coordname1'] = coordname1
    o['coordname2'] = coordname2
    o['coordx'] = coordx
    o['coordy'] = coordy
    o['ipix'] = imax + 1
    o['jpix'] = jmax + 1

    # project C0 on hlogebin
    C0datres,C0modres,C0aveinvwgtres = rebin_spectrum(C0logselenergycenter,C0dat_summed_map[:,jmax,imax],C0mod_summed_map[:,jmax,imax],C0weight_ave_map[:,jmax,imax],hlogebin)
    datcounts = C0datres
    modcounts = C0modres
    aveinvwgt = C0datres*C0aveinvwgtres # equivalent to dat/(1/invwgt)
    #
    # project C1 on hlogebin
    if optcomponent[1]:
        C1datres,C1modres,C1aveinvwgtres = rebin_spectrum(C1logselenergycenter,C1dat_summed_map[:,jmax,imax],C1mod_summed_map[:,jmax,imax],C1weight_ave_map[:,jmax,imax],hlogebin)
        datcounts = datcounts + C1datres
        modcounts = modcounts + C1modres
        aveinvwgt = aveinvwgt + C1datres*C1aveinvwgtres
        #
    # project C2 on hlogebin
    if optcomponent[2]:
        C2datres,C2modres,C2aveinvwgtres = rebin_spectrum(C2logselenergycenter,C2dat_summed_map[:,jmax,imax],C2mod_summed_map[:,jmax,imax],C2weight_ave_map[:,jmax,imax],hlogebin)
        datcounts = datcounts + C2datres
        modcounts = modcounts + C2modres
        aveinvwgt = aveinvwgt + C2datres*C2aveinvwgtres
        #
    # project C3 on hlogebin
    if optcomponent[3]:
        C3datres,C3modres,C3aveinvwgtres = rebin_spectrum(C3logselenergycenter,C3dat_summed_map[:,jmax,imax],C3mod_summed_map[:,jmax,imax],C3weight_ave_map[:,jmax,imax],hlogebin)
        datcounts = datcounts + C3datres
        modcounts = modcounts + C3modres
        aveinvwgt = aveinvwgt + C3datres*C3aveinvwgtres
    #
    mskdat = datcounts>0
    aveinvwgt[mskdat] = aveinvwgt[mskdat]/datcounts[mskdat]
    aveinvwgt[~mskdat] = 1
    weights = 1/aveinvwgt

    if chatter>1:
        print ('PSF-like integrated count spectra in pixel (%d,%d):' %(imax+1,jmax+1))
        for i in range(len(datcounts)):
            mystr = ('bin %02d [%10.3f,%10.3f] dat %7.0f mod %10.3f wgt %3.2f' % (i,np.power(10,hlogebin[i]),np.power(10,hlogebin[i+1]),datcounts[i],modcounts[i],weights[i]))
            if chatter>2:
                mystr = ('%s   C0 %7.0f %10.3f %3.2f' %(mystr,C0datres[i],C0modres[i],1/C0aveinvwgtres[i]))
                if optcomponent[1]:
                    mystr = ('%s C1 %7.0f %10.3f %3.2f' %(mystr,C1datres[i],C1modres[i],1/C1aveinvwgtres[i]))
                if optcomponent[2]:
                    mystr = ('%s C2 %7.0f %10.3f %3.2f' %(mystr,C2datres[i],C2modres[i],1/C2aveinvwgtres[i]))
                if optcomponent[3]:
                    mystr = ('%s C3 %7.0f %10.3f %3.2f' %(mystr,C3datres[i],C3modres[i],1/C3aveinvwgtres[i]))
            print ('%s' %(mystr))

    psmapsigma = np.sign(psmap) * np.sqrt(2) * erfcinv(np.power(10., -np.abs(psmap)))

    o['wcs2d'] = wcs2d
    o['psmap'] = psmap
    o['psmapsigma'] = psmapsigma
    o['datcounts'] = datcounts
    o['modcounts'] = modcounts
    o['weights'] = weights
    o['hlogebin'] = hlogebin

    return o

def make_psmap_fits(o, out_filename):
    psmap      = o['psmap']
    psmapsigma = o['psmapsigma']
    wcs2d      = o['wcs2d']

    o['file_name'] = out_filename
    hdu_ps_sigma = fits.PrimaryHDU(psmapsigma, wcs2d.to_header())
    hdr = hdu_ps_sigma.header
    hdr['EXTNAME'] = 'PS in sigma'
    hdu_ps = fits.ImageHDU(psmap, wcs2d.to_header())
    hdr = hdu_ps.header
    hdr['EXTNAME'] = 'PS'

    hdul = fits.HDUList([hdu_ps_sigma,hdu_ps])
    hdul.writeto(out_filename, overwrite=True)

if __name__ == "__main__":
    import argparse
    usage = "Usage: %(prog)s  [options] input"

    description = "python script"
    parser = argparse.ArgumentParser(usage=usage, description=description)

    parser.add_argument("-cm", "--cmap", required=True, type=str, help="data 3d map file or list of files (separated by colons)")
    parser.add_argument("-mm", "--mmap", required=True, type=str, help="model 3d map file or list of files (separated by colons)")
    parser.add_argument("-wm", "--wmap", required=False, type=str, default='', help="weights 3d map file or list of files (separated by colons)")
    parser.add_argument("-emin", "--emin", required=True, type=float, default=100, help="PS computation: minimum energy/MeV")
    parser.add_argument("-emax", "--emax", required=True, type=float, default=1e6, help="PS computation: maximum energy/MeV")
    parser.add_argument("-nbinloge", "--nbinloge", required=True, type=int, default=20, help="PS computation: number of log10 energy bins")
    parser.add_argument("-o", "--outfile", required=True, type=str, help="output fits file name")

    parser.add_argument("-fr", "--fixedradius", required=False, type=float, default=-1.0, help="spatial integration: fixed radius (deg)")
    parser.add_argument("-psf0", "--psfpar0", required=False, type=float, default=4.0, help="spatial integration: PSF-like parameter 0 (deg)")
    parser.add_argument("-psf1", "--psfpar1", required=False, type=float, default=100.0, help="spatial integration: PSF-like parameter 1")
    parser.add_argument("-psf2", "--psfpar2", required=False, type=float, default=0.9, help="spatial integration: PSF-like parameter 2")
    parser.add_argument("-psf3", "--psfpar3", required=False, type=float, default=0.1, help="spatial integration: PSF-like parameter 3 (deg)")
    parser.add_argument("-psf0lst", "--psfpar0lst", required=False, type=str, default='', help="spatial integration: list of the PSF-like parameters 0 of all components (separated by colons)")

    parser.add_argument("-c", "--chatter", required=False, type=int, default=1, help="output verbosity")

    parser.add_argument("-ix", "--ipix", required=False, type=int, default=-1, help="PS computation: sub-ROI central pixel i-axis position")
    parser.add_argument("-jx", "--jpix", required=False, type=int, default=-1, help="PS computation: sub-ROI central pixel j-axis position")
    parser.add_argument("-dx", "--dpix", required=False, type=int, default=0, help="PS computation: sub-ROI half-width in pixel (ignoring central pixel; dpix=0 corresponds to the central pixel only)")

    parser.add_argument("-mpc", "--maxpoissoncount", required=False, type=float, default=100, help="LL computation: number of counts up to which Poisson statistics is considered")
    parser.add_argument("-pe", "--prob_epsilon", required=False, type=float, default=1e-7, help="LL computation: precision parameter")
    parser.add_argument("-nb", "--nbinpdf", required=False, type=int, default=50, help="LL computation: number of bins")
    parser.add_argument("-sa", "--scaleaxis", required=False, type=float, default=20, help="LL computation: scale axis")

    args = vars(parser.parse_args())
    o=run(args)

    if len(o)>0 and args['outfile'] !='':
        make_psmap_fits(o,out_filename=args['outfile'])

