#!/usr/bin/env python
#

# Description
"""
Utilities for plotting SEDs and Castro plots

Many parts of this code are taken from dsphs/like/lnlfn.py by 
  Matthew Wood <mdwood@slac.stanford.edu>
  Alex Drlica-Wagner <kadrlica@slac.stanford.edu>
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

NORM_LABEL = {'NORM':r'Flux Normalization [a.u.]',
              'FLUX':r'E dN/dE [ph $cm^{-2} s^{-1}$]',
              'EFLUX':r'$E^2$ dN/dE [MeV $cm^{-2} s^{-1}]$',
              'NPRED':r'$n_{\rm pred}$ [ph]',
              'DFDE':r'dN/dE [ph $cm^{-2} s^{-1} MeV^{-1}$]',
              'EDFDE':r'E dN/dE [MeV $cm^{-2} s^{-1} MeV^{-1}$]'}
              

def plotNLL_v_Flux(nll,fluxType,nstep=25,xlims=None):
    """ Plot the (negative) log-likelihood as a function of normalization

    nll   : a LnLFN object
    nstep : Number of steps to plot 
    xlims : x-axis limits, if None, take tem from the nll object

    returns fig,ax, which are matplotlib figure and axes objects
    """
    if xlims is None:
        xmin = nll.interp.xmin
        xmax = nll.interp.xmax
    else:
        xmin = xlims[0]
        xmax = xlims[1]
    y1 = nll.interp(xmin)
    y2 = nll.interp(xmax)

    ymin = min(y1,y2,0.0)
    ymax = max(y1,y2,0.5)    

    xvals = np.linspace(xmin,xmax,nstep)
    yvals = nll.interp(xvals)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlim((xmin,xmax))
    ax.set_ylim((ymin,ymax))
    
    ax.set_xlabel(NORM_LABEL[fluxType])
    ax.set_ylabel(r'$\Delta \log\mathcal{L}$')
    ax.plot(xvals,yvals)
    return fig,ax


def plotCastro(castroData,ylims,nstep=25,zlims=None):
    """ Make a color plot (castro plot) of the (negative) log-likelihood as a function of 
    energy and flux normalization

    castroData : A CastroData object, with the log-likelihood v. normalization for each energy bin
    ylims      : y-axis limits
    nstep      : Number of y-axis steps to plot for each energy bin
    zlims      : z-axis limits

    returns fig,ax,im which are matplotlib figure, axes and image objects
    """
    xmin = castroData.specData.log_ebins[0]
    xmax = castroData.specData.log_ebins[-1]
    ymin = ylims[0]
    ymax = ylims[1]
    if zlims is None:
        zmin = 0
        zmax = 10.
    else:
        zmin = zlims[0]
        zmax = zlims[1]
  
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_yscale('log')
    ax.set_xlim((xmin,xmax))
    ax.set_ylim((ymin,ymax))
    
    ax.set_xlabel("log(Energy/GeV)")
    ax.set_ylabel(NORM_LABEL[castroData.norm_type])

    normVals = np.logspace(np.log10(ymin),np.log10(ymax),nstep)
    ztmp = []
    for i in range(castroData.specData.nE):
        ztmp.append(castroData[i].interp(normVals))
        pass
    ztmp = np.asarray(ztmp).T
    ztmp = np.where(ztmp<zmin,np.nan,ztmp)
    im = ax.imshow(ztmp, extent=[xmin,xmax,ymin,ymax],
                   origin='lower', aspect='auto',interpolation='nearest',
                   vmin=zmin, vmax=zmax,cmap=matplotlib.cm.jet_r)
    
    limits = castroData.getLimits(0.05)
    ebin_centers = (castroData.specData.ebins[0:-1]+castroData.specData.ebins[1:])/2.
    ax.errorbar(ebin_centers,limits)

    return fig,ax,im,ztmp



def plotSED(castroData,ylims,TS_thresh=4.0,errSigma=1.0,specVals=[]):
    """ Make a color plot (castro plot) of the (negative) log-likelihood as a function of 
    energy and flux normalization

    castroData : A CastroData object, with the log-likelihood v. normalization for each energy bin
    ylims      : y-axis limits
    TS_thresh  : TS value above with to plot a point, rather than an upper limit
    errSigma   : Number of sigma to use for error bars
    specVals   : List of spectra to add to plot
    returns fig,ax which are matplotlib figure and axes objects
    """
    xmin = castroData.specData.ebins[0]
    xmax = castroData.specData.ebins[-1]
    ymin = ylims[0]
    ymax = ylims[1]
  
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim((xmin,xmax))
    ax.set_ylim((ymin,ymax))
    
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel(NORM_LABEL[castroData.norm_type])

    ts_vals = castroData.ts_vals()
    mles = castroData.mles()

    has_point = ts_vals > TS_thresh
    has_limit = ~has_point
    ul_vals = castroData.getLimits(0.05)
    
    err_pos = castroData.getLimits(0.32) - mles
    err_neg = mles - castroData.getLimits(0.32,upper=False) 

    yerr_points = (err_neg[has_point],err_pos[has_point])
    xerrs = ( castroData.specData.evals - castroData.specData.ebins[0:-1],
              castroData.specData.ebins[1:] - castroData.specData.evals )

    yerr_limits = (0.5*ul_vals[has_limit],np.zeros((has_limit.sum())))

    ax.errorbar(castroData.specData.evals[has_point],mles[has_point],
                yerr=yerr_points,fmt='o')

    ax.errorbar(castroData.specData.evals[has_limit],ul_vals[has_limit], 
                yerr=yerr_limits, lw=1, color='red', ls='none', zorder=1, uplims=True)

    ax.errorbar(castroData.specData.evals[has_limit], ul_vals[has_limit], xerr=(xerrs[0][has_limit],xerrs[1][has_limit]),
                lw=1.35, ls='none', color='red', zorder=2, capsize=0)

    for spec in specVals:
        ax.loglog(castro.specData.evals,spec)
        pass

    return fig,ax


if __name__ == "__main__":

    
    from fermipy import sed
    from fermipy import roi_model
    import sys

    if len(sys.argv) == 1:
        flux_type = "FLUX"
    else:
        flux_type = sys.argv[1]
        

    if flux_type == 'NORM':
        xlims = (0.,1.)
        flux_lims = (1e-5,1e-1)
    elif flux_type == "FLUX":
        xlims = (0.,1.)
        flux_lims = (1e-13,1e-9)
    elif flux_type == "EFLUX":
        xlims = (0.,1.)
        flux_lims = (1e-8,1e-4)
    elif flux_type == "NPRED":
        xlims = (0.,1.)
        flux_lims = (1e-1,1e3)
    elif flux_type == "DFDE":
        xlims = (0.,1.)
        flux_lims = (1e-18,1e-11)
    elif flux_type == "EDFDE":
        xlims = (0.,1.)
        flux_lims = (1e-13,1e-9)
    else:
        print ("Didn't reconginize flux type %s, choose from NORM | FLUX | EFLUX | NPRED | DFDE | EDFDE"%sys.argv[1])
        sys.exit()
        
    tscube = sed.TSCube.create_from_fits("tscube_test.fits",flux_type)
    resultDict = tscube.find_sources(10.0,1.0,use_cumul=False,
                                     output_peaks=True,
                                     output_specInfo=True,
                                     output_srcs=True)

    peaks = resultDict["Peaks"]

    max_ts = tscube.tsmap.counts.max()
    (castro,test_dict) = tscube.test_spectra_of_peak(peaks[0])

    nll = castro[2]
    fig,ax = plotNLL_v_Flux(nll,flux_type)

    fig2,ax2,im2,ztmp2 = plotCastro(castro,ylims=flux_lims,nstep=100)
        
    spec_pl = test_dict["PowerLaw"]["Spectrum"]
    spec_lp = test_dict["LogParabola"]["Spectrum"]
    spec_pc = test_dict["PLExpCutoff"]["Spectrum"]

    fig3,ax3 = plotSED(castro,ylims=flux_lims,TS_thresh=2.0,specVals=[spec_pl])        

    result_pl = test_dict["PowerLaw"]["Result"]
    result_lp = test_dict["LogParabola"]["Result"]
    result_pc = test_dict["PLExpCutoff"]["Result"]
    ts_pl = test_dict["PowerLaw"]["TS"]
    ts_lp = test_dict["LogParabola"]["TS"]
    ts_pc = test_dict["PLExpCutoff"]["TS"]

    print "TS for PL index = 2:  %.1f"%max_ts
    print "Cumulative TS:        %.1f"%castro.ts_vals().sum()
    print "TS for PL index free: %.1f (Index = %.2f)"%(ts_pl,result_pl[1])
    print "TS for LogParabola:   %.1f (Index = %.2f, Beta = %.2f)"%(ts_lp,result_lp[1],result_lp[2])
    print "TS for PLExpCutoff:   %.1f (Index = %.2f, E_c = %.2f)"%(ts_pc,result_pc[1],result_pc[2])

     
