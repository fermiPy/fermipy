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

NORM_LABEL = [r'Flux Normalization [a.u.]',
              r'E dN/dE [ph $cm^{-2} s^{-1}$]',
              r'$E^2$ dN/dE [ph $cm^{-2} s^{-1}$]',
              r'$n_{\rm pred}$ [ph]']

def plotNLL_v_Flux(nll,nstep=25,xlims=None):
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

    ymin = min(y1,y2)
    ymax = max(y1,y2,0.5)    

    xvals = np.linspace(xmin,xmax,nstep)
    yvals = nll.interp(xvals)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlim((xmin,xmax))
    ax.set_ylim((ymin,ymax))
    
    ax.set_xlabel("Flux Normalization [a.u.]")
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
        zmin = -10.
        zmax = 0.
    else:
        zmin = zlims[0]
        zmax = zlims[1]
  
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_yscale('log')
    ax.set_xlim((xmin,xmax))
    ax.set_ylim((ymin,ymax))
    
    ax.set_xlabel("log(Energy/GeV)")
    ax.set_ylabel(NORM_LABEL[castroData.fluxType])

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

    return fig,ax,im



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
    ax.set_ylabel(NORM_LABEL[castroData.fluxType])

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
    import sys

    if len(sys.argv) == 1:
        flux_type = 0
    elif sys.argv[1] == "norm":
        flux_type = 0
    elif sys.argv[1] == "flux":
        flux_type = 1
    elif sys.argv[1] == "eflux":
        flux_type = 2
    elif sys.argv[1] == "npred":
        flux_type = 3
    else:
        print "Didn't reconginize flux type %s, choose from norm | flux | eflux | npred"%sys.argv[1]

    idx_off = 2.
    if flux_type == 0:
        xlims = (0.,1.)
        ylims = (1e-5,1e-1)
        initPars = np.array([1e-3,0.0,0.0])
        initPars_pc = np.array([1e-3,0.0,1000.0])    
    elif flux_type == 1:
        xlims = (0.,1.)
        ylims = (1e-13,1.e-9)
        idx_off = 1.
        initPars = np.array([1e-12,-2.0,0.0])
        initPars_pc = np.array([1e-12,-2.0,1000.0])
    elif flux_type == 2:
        xlims = (0.,1.)
        ylims = (1e-8,1.e-4)
        initPars = np.array([1e-7,-2.0,0.0])
        initPars_pc = np.array([1e-7,-2.0,1000.0])       
    elif flux_type == 3:
        xlims = (0.,1.)
        ylims = (1e-1,1.e3)
        idx_off = 1.
        initPars = np.array([1.0,-2.0,0.0])
        initPars_pc = np.array([1.0,-2.0,1000.0])       
        
    tscube = sed.TSCube.create_from_fits("tscube_test.fits",flux_type)

    ts_map = tscube.tsmap.counts
    max_ts_pix = np.argmax(ts_map)
    max_ts = ts_map.flat[max_ts_pix]
    xpix = max_ts_pix/80
    ypix = max_ts_pix%80
    ipix = 80*ypix + xpix
    
    castro = tscube.castroData_from_ipix(ipix)

    nll = castro[2]
    fig,ax = plotNLL_v_Flux(nll)

    fig2,ax2,im2 = plotCastro(castro,ylims=ylims,nstep=100)
        
    specVals = np.ones((castro.specData.nE))

    result = castro.fitNormalization(specVals,xlims)
    result2 = castro.fitNorm_v2(specVals)

    pl = sed.Powerlaw(castro.specData.evals,1000)
    lp = sed.LogParabola(castro.specData.evals,1000)
    pc = sed.PlExpCutoff(castro.specData.evals,1000)

    result_pl,spec_pl,ts_pl = castro.fit_spectrum(pl,initPars[0:2])    
    result_lp,spec_lp,ts_lp = castro.fit_spectrum(lp,initPars)
    result_pc,spec_pc,ts_pc = castro.fit_spectrum(pc,initPars_pc)
     
    fig3,ax3 = plotSED(castro,ylims=ylims,TS_thresh=4.0,specVals=[spec_pl,spec_lp,spec_pc])

    print "TS for PL index = 2:  %.1f"%max_ts
    print "TS for PL index free: %.1f (Index = %.2f)"%(ts_pl[0],idx_off-result_pl[1])
    print "TS for LogParabola:   %.1f (Index = %.2f, Beta = %.2f)"%(ts_lp[0],idx_off-result_lp[1],result_lp[2])
    print "TS for PLExpCutoff:   %.1f (Index = %.2f, E_c = %.2f)"%(ts_pc[0],idx_off-result_pc[1],result_pc[2])

    fig2.show()
    fig3.show()
