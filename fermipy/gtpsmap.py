#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
###################################################################
#
# gtpsmap: check data/model agreement in Fermi-LAT analysis
#
# This python script compares data and model 3D maps by computing a PS map.
# It has been written by Philippe Bruel (LLR, CNRS/IN2P3, IPP Paris, France), based on a new method described in a publication currently in preparation.
# Meanwhile one can look at the corresponding Fermi symposium 2021 presentation:
# https://indico.cern.ch/event/1010947/contributions/4278096/
#
# The main inputs of the script are:
# - the data 3D count map of the data (output of gtbin)
# - the model 3D count map (output of gtmodel or gta.write_model_map() for fermipy users)
# - if you use weighted likelihood to perform the fit, then you also have to provide the 3D weight map
#
# Usage:
# gtpsmap.py --cmap <data 3D map> --mmap <model 3D map> --wmap <weights 3D map (optional)> --outfile <output file>
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
# --emin <minimum energy/MeV; 1> --emax <maximum energy/MeV; 1e9>
#   energy range to compute PS
# --rebin <nb of bins merged when rebinning the energy axis; 1>
# --prob_epsilon <precision parameter; 1e-7>
#   This parameter defines the k-interval over which the Poisson probability of each spectral bin is considered
#   when computing the log-likelihood distribution (the k-interval is such that ΣPoisson(k,model_i) = 1-prob_epsilon).
#   The default value 1e-7 provides a 1% precision on PS up to PS=20. If you want a better precision for larger PS values,
#   you need a smaller prob_epsilon (1e-11 provides a 5% precision up to 250) but that makes the script slower.
# --psfpar0 <PSF-like par0; 4.0 deg> --psfpar1 <PSF-like par1; 100 MeV>  --psfpar2 <PSF-like par2; 0.9>  --psfpar3 <PSF-like par3; 0.1deg>
#   PSF-like selection radius = sqrt(p0^2*pow(E/p1,-2*p2) + p3^2)
# --chatter <output verbosity; 1>
#
###################################################################

import sys, getopt
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
import scipy.signal
from scipy.stats import poisson
from scipy.special import gammainc
from scipy.special import gammaincc
from scipy.special import erfcinv

from gammapy.maps import WcsNDMap, WcsGeom

#import cProfile
#import pstats
#v = dict()

def getDeviationProbability(datcts, modcts, weights, maxpoissoncount, prob_epsilon, nbinpdf, scaleaxis):
    # datcts = array of data counts
    # modcts = array of model counts
    # weights = array of weights
    # for simplicty's sake, LL (LogLikelihood) stands for -LL
    
    datcts=np.rint(datcts)

    # sort model counts in descending order
    sortindex = np.argsort(-modcts)
    smodcts = modcts[sortindex]
    sdatcts = datcts[sortindex]
    sweights = weights[sortindex]

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


def run(args):
    dat_filename    = args['cmap']
    mod_filename    = args['mmap']
    weight_filename = args['wmap']
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
    emin = args['emin']
    emax = args['emax']
    chatter = args['chatter']

    ipix = args['ipix']
    jpix = args['jpix']
    rebin = args['rebin']
    write_fits = args['write_fits']

    o = {} # this sintax is to respect matt syntax where all the output are saved in the dictionary called "o"
    optweight = 1
    if weight_filename == '':
        optweight = 0
    
    dat_hdulist = fits.open(dat_filename)
    dat_hdu = dat_hdulist[0]
    dat_map0 = dat_hdu.data

    ene_hdu = dat_hdulist[1]
    ene_table = ene_hdu.data
    energymin = ene_table['E_MIN']
    energymax = ene_table['E_MAX']
    energycenter = np.sqrt(ene_table['E_MIN']*ene_table['E_MAX'])
    if ene_hdu.header['TUNIT2']=='keV':
        energymin = energymin/1000
        energymax = energymax/1000
        energycenter = energycenter/1000

    mskle = energymax>emin
    mskhe = energymin<emax
    mskerange = mskle & mskhe
    if np.max(mskerange)==False:
        print('The energy range emin=%f emax=%f is empty. Exiting.' % (emin,emax))
        return 0
    selenergymin = energymin[mskerange]
    selenergymax = energymax[mskerange]
    
    mod_hdulist = fits.open(mod_filename)
    mod_hdu = mod_hdulist[0]
    mod_map = mod_hdu.data
    
    # the weights of the integrated count spectra correspond to the Ndata-averaged 1/weights over the PSF-like region
    if optweight==1:
        weight_hdulist = fits.open(weight_filename)
        weight_map = weight_hdulist[0].data
        weight_map = 1/weight_map

    coordname1 = dat_hdu.header['CTYPE1']
    coordname2 = dat_hdu.header['CTYPE2']

    naxis1 = dat_hdu.header['NAXIS1']
    naxis2 = dat_hdu.header['NAXIS2']
    naxis3 = dat_hdu.header['NAXIS3']

    # get 2d part of wcs
    wcs3d = WCS(dat_hdu.header,fix=0)
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

    if fixedradius>0:
        psfpar0 = 0
        psfpar3 = fixedradius
        if chatter>0:
            print ('Computing PSF-like summed maps with fixed radius %f deg' % (fixedradius))
    else:
        if chatter>0:
            print ('Computing PSF-like summed maps with energy dependent parameterization (%f deg, %f MeV, %f, %f deg)' % (psfpar0,psfpar1,psfpar2,psfpar3))

    psflikeroi = np.sqrt(psfpar0*psfpar0*np.power(psfpar1/energymin,2*psfpar2)+psfpar3*psfpar3)

    kernel_map=np.zeros((naxis3,naxis2,naxis1),dtype=np.float64)
    dat_summed_map=np.zeros((naxis3,naxis2,naxis1),dtype=np.float64)
    mod_summed_map=np.zeros((naxis3,naxis2,naxis1),dtype=np.float64)

    dat_map=np.zeros((naxis3,naxis2,naxis1),dtype=np.float64)
    dat_map = dat_map0

    # the weights of the integrated count spectra correspond to the Ndata-averaged 1/weights over the PSF-like region
    if optweight:
        prod_weightdat_map=weight_map*dat_map
        weight_avesummed_map=np.zeros((naxis3,naxis2,naxis1),dtype=np.float64)

    cpix=np.array([icenter,jcenter])
    for k in range(naxis3):
        kernel_map[k,:,:]=dist.degree < psflikeroi[k]
        dat_summed_map[k,:,:]=convolve_map(dat_map[k,:,:],kernel_map[k,:,:],cpix)
        mod_summed_map[k,:,:]=convolve_map(mod_map[k,:,:],kernel_map[k,:,:],cpix)
        if optweight:
            weight_avesummed_map[k,:,:]=convolve_map(prod_weightdat_map[k,:,:],kernel_map[k,:,:],cpix)

    msk = dat_summed_map > 0.1
    dat_summed_map[~msk] = 0

    # the weights of the integrated count spectra correspond to the Ndata-averaged 1/weights over the PSF-like region
    weight_ave_map = np.ones((naxis3,naxis2,naxis1),dtype=np.float64)
    if optweight:
        msk = dat_summed_map>0
        weight_ave_map[msk] = weight_avesummed_map[msk]/dat_summed_map[msk]
        weight_ave_map[msk] = 1/weight_ave_map[msk]

    # select energy range
    mod_summed_map = mod_summed_map[mskerange]
    dat_summed_map = dat_summed_map[mskerange]
    weight_ave_map = weight_ave_map[mskerange]
    print("Selecting the energy bins including emin=%f and emax=%f -> %d bins in [%f,%f] " %(emin,emax,mod_summed_map.shape[0],selenergymin[0],selenergymax[-1]))

    naxis3rebin = mod_summed_map.shape[0]
    if rebin>1:
        r = mod_summed_map.shape[0]%rebin
        nrebin = mod_summed_map.shape[0]//rebin
        naxis3rebin = nrebin
        if r>0:
            mod_summed_map = mod_summed_map[:-r]
            dat_summed_map = dat_summed_map[:-r]
            weight_ave_map = weight_ave_map[:-r]
            selenergymin = selenergymin[:-r]
            selenergymax = selenergymax[:-r]
        print("Rebinning the energy axis with rebin=%d (ignoring %d bins at the end): %d bins in [%f,%f], logE bin size=%f" %(rebin,r,nrebin,selenergymin[0],selenergymax[-1],np.log10(selenergymin[rebin]/selenergymin[0])))
        #
        weight_ave_map = dat_summed_map/weight_ave_map
        mod_summed_map = mod_summed_map.reshape(nrebin,rebin,naxis2,-1).sum(axis=1)
        dat_summed_map = dat_summed_map.reshape(nrebin,rebin,naxis2,-1).sum(axis=1)
        weight_ave_map = weight_ave_map.reshape(nrebin,rebin,naxis2,-1).sum(axis=1)
        mskdat = dat_summed_map>0
        weight_ave_map[mskdat] = weight_ave_map[mskdat]/dat_summed_map[mskdat]
        weight_ave_map[~mskdat] = 1
        weight_ave_map = 1/weight_ave_map
    
    if chatter>0:
        print('Computing the PS map (%d bins in energy range [%3.2f,%3.2f], nbinpdf=%d, scaleaxis=%3.2f, maxpoissoncount=%3.2f, prob_epsilon=%g):\n(for each column of the map, the integer part of the maximum value of abs(PS) is given as one character [1,..,9,A=10,B=11,..,Y=34,Z>=35])' %(naxis3rebin,selenergymin[0],selenergymax[-1],nbinpdf,scaleaxis,maxpoissoncount,prob_epsilon))
        
    psmap      = np.zeros((naxis2,naxis1))

    totresmap  = np.zeros((naxis2,naxis1))

    psmax = -1;
    imax = 0
    jmax = 0
    psvalstr = ['0','1','2','3','4','5','6','7','8','9']
    psvalstr.extend(['A','B','C','D','E','F','G','H','I','J'])
    psvalstr.extend(['K','L','M','N','O','P','Q','R','S','T'])
    psvalstr.extend(['U','V','W','X','Y','Z'])

    nbnan = int(0)
    psmaxcol = 0
    axisrange1 = range(naxis1)
    axisrange2 = range(naxis2)
    # change range if ipix and jpix are provided
    if ipix>0 and jpix>0:
        axisrange1 = range(ipix-1,ipix)
        axisrange2 = range(jpix-1,jpix)

    
    # loop over the pixels and compute PS
    for i in axisrange1:
        psmaxcol = 0;
        for j in axisrange2:
            modcounts = mod_summed_map[:,j,i]
            datcounts = dat_summed_map[:,j,i]
            weights = weight_ave_map[:,j,i]
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
            if ps>psmax:
                psmax = ps
                imax = i
                jmax = j
            totresmap[j,i] = totres
            psmap[j,i] = ps
            if totres<0:
                psmap[j,i] = -ps
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

    maxdir = SkyCoord.from_pixel(imax,jmax,wcs=wcs2d,origin=0)

    if chatter>0:
        if maxdir.frame.name=='galactic':
            coordx=maxdir.l.to_string(decimal=True)
            coordy=maxdir.b.to_string(decimal=True)
        else:
            coordx=maxdir.ra.to_string(decimal=True)
            coordy=maxdir.dec.to_string(decimal=True)
        print ('Maximum PS: %f found at the position (%s,%s)=(%s,%s) [fits pixel (%d,%d)]' % (psmax,coordname1,coordname2,coordx,coordy,imax+1,jmax+1))
    o['psmax'] = psmax
    o['coordname1'] = coordname1
    o['coordname2'] = coordname2
    o['coordx'] = coordx
    o['coordy'] = coordy
    o['ipix'] = imax + 1
    o['jpix'] = jmax + 1


    modcounts = mod_summed_map[:,jmax,imax]
    datcounts = dat_summed_map[:,jmax,imax]
    weights = weight_ave_map[:,jmax,imax]

    if chatter>1:
        print ('PSF-like integrated count spectra in pixel (%d,%d):' %(imax+1,jmax+1))
        for i in range(naxis3rebin):
            print ('bin %02d data %f model %g weight %g' % (i,datcounts[i],modcounts[i],weights[i]))

    psmapsigma = np.sign(psmap) * np.sqrt(2) * erfcinv(np.power(10., -np.abs(psmap)))


    o['wcs2d'] = wcs2d
    o['psmap'] = psmap
    o['psmapsigma'] = psmapsigma
    return o

def make_psmap_fits(o, out_filename):
    psmap      = o['psmap']
    psmapsigma = o['psmapsigma']
    wcs2d      = o['wcs2d']

    o['file_name'] = out_filename
    hdu_ps = fits.PrimaryHDU(psmap, wcs2d.to_header())
    hdu_ps_sigma = fits.ImageHDU(psmapsigma, wcs2d.to_header())
    hdr = hdu_ps_sigma.header
    hdr['EXTNAME'] = 'PS in sigma'

    hdul = fits.HDUList([hdu_ps, hdu_ps_sigma])
    hdul.writeto(out_filename, overwrite=True)

if __name__ == "__main__":
    import argparse
    usage = "Usage: %(prog)s  [options] input"

    description = "python script"
    parser = argparse.ArgumentParser(usage=usage, description=description)

    #parser.add_argument("-d", "--dryrun", action='store_true')
    parser.add_argument("-cm", "--cmap", required=True, type=str, help="data 3d map")

    parser.add_argument("-mm", "--mmap", required=True, type=str, help="model 3d map")

    parser.add_argument("-wm", "--wmap", required=False, type=str, default='', help="weights 3d map")

    parser.add_argument("-o", "--outfile", required=True, type=str, help="Output file")

    parser.add_argument("-fr", "--fixedradius", required=False, type=float, default=-1.0, help="Fixed radius")

    parser.add_argument("-psf0", "--psfpar0", required=False, type=float, default=4.0, help="PSF parameter 0")
    parser.add_argument("-psf1", "--psfpar1", required=False, type=float, default=100.0, help="PSF parameter 1")
    parser.add_argument("-psf2", "--psfpar2", required=False, type=float, default=0.9, help="PSF parameter 2")
    parser.add_argument("-psf3", "--psfpar3", required=False, type=float, default=0.1, help="PSF parameter 3")

    parser.add_argument("-mpc", "--maxpoissoncount", required=False, type=float, default=100, help="Maximum number of counts")
    parser.add_argument("-pe", "--prob_epsilon", required=False, type=float, default=1e-7, help="precision parameter")
    parser.add_argument("-nb", "--nbinpdf", required=False, type=int, default=50, help="Number of bin of the PSF")
    parser.add_argument("-sa", "--scaleaxis", required=False, type=float, default=20, help="Scale axis")
    parser.add_argument("-emin", "--emin", required=False, type=float, default=1.0, help="minimum energy/MeV")
    parser.add_argument("-emax", "--emax", required=False, type=float, default=1e9, help="maximum energy/MeV")
    parser.add_argument("-c", "--chatter", required=False, type=int, default=1, help="output verbosity")

    parser.add_argument("-ix", "--ipix", required=False, type=int, default=-1, help="number of pixel i axis")
    parser.add_argument("-jx", "--jpix", required=False, type=int, default=-1, help="number of pixel j axis")
    parser.add_argument("-r", "--rebin", required=False, type=int, default=1, help="Rebin")


    args = vars(parser.parse_args())
    o=run(args)
    make_psmap_fits(o,out_filename=args['outfile'])

    # define the PSF-like energy dependent cut

    #cProfile.runctx('main(sys.argv[1:])',globals(),locals(),'profout.asc')
    #pstats.Stats('profout.asc').sort_stats('cumtime').print_stats(10)
    #locals().update(v)

