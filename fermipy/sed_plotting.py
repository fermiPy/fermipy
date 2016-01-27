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
    
    ax.set_xlabel("Flux Normalization")
    ax.set_ylabel(r'$Delta \log\mathcal{L}$')
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
    xmin = castroData.ebins[0]
    xmax = castroData.ebins[-1]
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
    
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel("Flux Normalization")

    normVals = np.logspace(np.log10(ymin),np.log10(ymax),nstep)
    ztmp = []
    for i in range(castroData.nE):
        ztmp.append(castroData[i].interp(normVals))
        pass
    ztmp = np.asarray(ztmp).T
    ztmp = np.where(ztmp<zmin,np.nan,ztmp)
    im = ax.imshow(ztmp, extent=[xmin,xmax,ymin,ymax],
                   origin='lower', aspect='auto',interpolation='nearest',
                   vmin=zmin, vmax=zmax,cmap=matplotlib.cm.jet_r)
    
    #limits = castroData.getLimits(0.05)
    #ebin_centers = np.sqrt(castroData.ebins[0:-1]*castroData.ebins[1:])
    #ax.errorbar(ebin_centers,limits)

    return fig,ax,im



if __name__ == "__main__":

    
    from fermipy import sed

    tscube = sed.TSCube.create_from_fits("tscube2_test2.fits")

    ts_map = tscube.tsmap.counts
    max_ts_pix = np.argmax(ts_map)

    castro = tscube.castroData_from_ipix(max_ts_pix)

    nll = castro[2]
    fig,ax = plotNLL_v_Flux(nll)

    fig2,ax2,im2 = plotCastro(castro,ylims=(1e-4,1.),nstep=100)
    
    
        
    
