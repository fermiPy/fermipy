# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities for plotting SEDs and Castro plots

Many parts of this code are taken from dsphs/like/lnlfn.py by 
  Matthew Wood <mdwood@slac.stanford.edu>
  Alex Drlica-Wagner <kadrlica@slac.stanford.edu>
"""
from __future__ import absolute_import, division, print_function
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np

NORM_LABEL = {
    'NORM': r'Flux Normalization [a.u.]',
    'flux': r'$F_{\rm min}^{\rm max} [ph $cm^{-2} s^{-1}$]',
    'eflux': r'$E F_{\rm min}^{\rm max}$ [MeV $cm^{-2} s^{-1}]$',
    'npred': r'$n_{\rm pred}$ [ph]',
    'dfde': r'dN/dE [ph $cm^{-2} s^{-1} MeV^{-1}$]',
    'edfde': r'E dN/dE [MeV $cm^{-2} s^{-1} MeV^{-1}$]',
    'e2dede': r'%E^2% dN/dE [MeV $cm^{-2} s^{-1} MeV^{-1}$]',
    'sigvj': r'$J\langle \sigma v \rangle$ [$GeV^{2} cm^{-2} s^{-1}$]',
    'sigv': r'$\langle \sigma v \rangle$ [$cm^{3} s^{-1}$]',
}


def plotNLL_v_Flux(nll, fluxType, nstep=25, xlims=None):
    """ Plot the (negative) log-likelihood as a function of normalization

    nll   : a LnLFN object
    nstep : Number of steps to plot 
    xlims : x-axis limits, if None, take tem from the nll object

    returns fig,ax, which are matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt

    if xlims is None:
        xmin = nll.interp.xmin
        xmax = nll.interp.xmax
    else:
        xmin = xlims[0]
        xmax = xlims[1]

    y1 = nll.interp(xmin)
    y2 = nll.interp(xmax)

    ymin = min(y1, y2, 0.0)
    ymax = max(y1, y2, 0.5)

    xvals = np.linspace(xmin, xmax, nstep)
    yvals = nll.interp(xvals)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    ax.set_xlabel(NORM_LABEL[fluxType])
    ax.set_ylabel(r'$-\Delta \log\mathcal{L}$')
    ax.plot(xvals, yvals)
    return fig, ax


def make_colorbar(fig, ax, im, zlims):
    """
    """
    pdf_adjust = 0.01  # Dealing with some pdf crap...
    cax = inset_axes(ax, width="3%", height="100%", loc=3,
                     bbox_to_anchor=(1.01, 0.0, 1.05, 1.00),
                     bbox_transform=ax.transAxes,
                     borderpad=0.)
    cbar = fig.colorbar(im, cax, ticks=np.arange(zlims[0], zlims[-1]))
    xy = cbar.outline.xy
    xy[0:, 0] *= 1 - 5 * pdf_adjust
    xy[0:, 1] *= 1 - pdf_adjust
    cbar.outline.set_xy(xy)
    cax.invert_yaxis()
    cax.axis['right'].toggle(ticks=True, ticklabels=True, label=True)
    cax.set_ylabel(r"$\Delta \log \mathcal{L}$")
    return cax, cbar


def plotCastro_base(castroData, xlims, ylims,
                    xlabel, ylabel, nstep=25, zlims=None):
    """ Make a color plot (castro plot) of the 
        log-likelihood as a function of 
        energy and flux normalization

    castroData : A CastroData_Base object, with the 
                 log-likelihood v. normalization for each energy bin
    xlims      : x-axis limits
    ylims      : y-axis limits
    xlabel     : x-axis title
    ylabel     : y-axis title
    nstep      : Number of y-axis steps to plot for each energy bin
    zlims      : z-axis limits

    returns fig,ax,im,ztmp which are matplotlib figure, axes and image objects
    """
    import matplotlib.pyplot as plt

    xmin = xlims[0]
    xmax = xlims[1]
    ymin = ylims[0]
    ymax = ylims[1]
    if zlims is None:
        zmin = -10
        zmax = 0.
    else:
        zmin = zlims[0]
        zmax = zlims[1]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    normVals = np.logspace(np.log10(ymin), np.log10(ymax), nstep)
    ztmp = []
    for i in range(castroData.nx):
        ztmp.append(castroData[i].interp(normVals))

    ztmp = np.asarray(ztmp).T
    ztmp *= -1.
    ztmp = np.where(ztmp < zmin, np.nan, ztmp)
    cmap = plt.get_cmap('jet_r')
    im = ax.imshow(ztmp, extent=[xmin, xmax, ymin, ymax],
                   origin='lower', aspect='auto', interpolation='nearest',
                   vmin=zmin, vmax=zmax, cmap=cmap)

    cax, cbar = make_colorbar(fig, ax, im, (zmin, zmax))
    # cbar = fig.colorbar(im, ticks=np.arange(zmin,zmax),
    #                    fraction=0.10,panchor=(1.05,0.5))
    #cbar.set_label(r'$\Delta \log\mathcal{L}$')
    #cax = None
    return fig, ax, im, ztmp, cax, cbar


def plotCastro(castroData, ylims, nstep=25, zlims=None):
    """ Make a color plot (castro plot) of the 
        delta log-likelihood as a function of 
        energy and flux normalization

    castroData : A CastroData object, with the 
                 log-likelihood v. normalization for each energy bin
    ylims      : y-axis limits
    nstep      : Number of y-axis steps to plot for each energy bin
    zlims      : z-axis limits

    returns fig,ax,im,ztmp which are matplotlib figure, axes and image objects
    """
    xlims = (castroData.refSpec.ebins[0],
             castroData.refSpec.ebins[-1])
    xlabel = "Energy [GeV]"
    ylabel = NORM_LABEL[castroData.norm_type]
    return plotCastro_base(castroData, xlims, ylims,
                           xlabel, ylabel, nstep, zlims)


def plotSED_OnAxis(ax, castroData, TS_thresh=4.0, errSigma=1.0,
                   colorLim='red', colorPoint='blue'):
    """
    """
    ts_vals = castroData.ts_vals()
    mles = castroData.mles()

    has_point = ts_vals > TS_thresh
    has_limit = ~has_point
    ul_vals = castroData.getLimits(0.05)

    err_lims_lo, err_lims_hi = castroData.getIntervals(0.32)

    err_pos = err_lims_hi - mles
    err_neg = mles - err_lims_lo

    yerr_points = (err_neg[has_point], err_pos[has_point])
    xerrs = (castroData.refSpec.eref - castroData.refSpec.ebins[0:-1],
             castroData.refSpec.ebins[1:] - castroData.refSpec.eref)

    yerr_limits = (0.5 * ul_vals[has_limit], np.zeros((has_limit.sum())))

    ax.errorbar(castroData.refSpec.eref[has_point], mles[has_point],
                yerr=yerr_points, fmt='o', color=colorPoint)

    ax.errorbar(castroData.refSpec.eref[has_limit], ul_vals[has_limit],
                yerr=yerr_limits, lw=1, color=colorLim,
                ls='none', zorder=1, uplims=True)

    ax.errorbar(castroData.refSpec.eref[has_limit], ul_vals[has_limit],
                xerr=(xerrs[0][has_limit], xerrs[1][has_limit]),
                lw=1.35, ls='none', color=colorLim, zorder=2, capsize=0)


def plotSED(castroData, ylims, TS_thresh=4.0, errSigma=1.0, specVals=[]):
    """ Make a color plot (castro plot) of the (negative) log-likelihood 
        as a function of energy and flux normalization

    castroData : A CastroData object, with the 
                 log-likelihood v. normalization for each energy bin
    ylims      : y-axis limits
    TS_thresh  : TS value above with to plot a point, 
                 rather than an upper limit
    errSigma   : Number of sigma to use for error bars
    specVals   : List of spectra to add to plot
    returns fig,ax which are matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt

    xmin = castroData.refSpec.ebins[0]
    xmax = castroData.refSpec.ebins[-1]
    ymin = ylims[0]
    ymax = ylims[1]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel(NORM_LABEL[castroData.norm_type])

    plotSED_OnAxis(ax, castroData, TS_thresh, errSigma)

    for spec in specVals:
        ax.loglog(castroData.refSpec.eref, spec)
        pass

    return fig, ax


def compare_SED(castroData1, castroData2, ylims, TS_thresh=4.0,
                errSigma=1.0, specVals=[]):
    """ Compare two SEDs 

    castroData1: A CastroData object, with the 
                 log-likelihood v. normalization for each energy bin
    castroData2: A CastroData object, with the 
                 log-likelihood v. normalization for each energy bin
    ylims      : y-axis limits
    TS_thresh  : TS value above with to plot a point, 
                 rather than an upper limit
    errSigma   : Number of sigma to use for error bars
    specVals   : List of spectra to add to plot
    returns fig,ax which are matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt

    xmin = min(castroData1.refSpec.ebins[0], castroData2.refSpec.ebins[0])
    xmax = max(castroData1.refSpec.ebins[-1], castroData2.refSpec.ebins[-1])
    ymin = ylims[0]
    ymax = ylims[1]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel(NORM_LABEL[castroData1.norm_type])

    plotSED_OnAxis(ax, castroData1, TS_thresh, errSigma,
                   colorLim='blue', colorPoint='blue')
    plotSED_OnAxis(ax, castroData2, TS_thresh, errSigma,
                   colorLim='red', colorPoint='red')

    for spec in specVals:
        ax.loglog(castroData1.refSpec.eref, spec)

    return fig, ax


if __name__ == "__main__":

    from fermipy import castro
    import sys

    if len(sys.argv) == 1:
        flux_type = "FLUX"
    else:
        flux_type = sys.argv[1]

    if flux_type == 'NORM':
        xlims = (0., 1.)
        flux_lims = (1e-5, 1e-1)
    elif flux_type == "FLUX":
        xlims = (0., 1.)
        flux_lims = (1e-13, 1e-9)
    elif flux_type == "EFLUX":
        xlims = (0., 1.)
        flux_lims = (1e-8, 1e-4)
    elif flux_type == "NPRED":
        xlims = (0., 1.)
        flux_lims = (1e-1, 1e3)
    elif flux_type == "DFDE":
        xlims = (0., 1.)
        flux_lims = (1e-18, 1e-11)
    elif flux_type == "EDFDE":
        xlims = (0., 1.)
        flux_lims = (1e-13, 1e-9)
    else:
        print(
            "Didn't reconginize flux type %s, choose from NORM | FLUX | EFLUX | NPRED | DFDE | EDFDE" % sys.argv[1])
        sys.exit()

    tscube = castro.TSCube.create_from_fits("tscube_test.fits", flux_type)
    resultDict = tscube.find_sources(10.0, 1.0, use_cumul=False,
                                     output_peaks=True,
                                     output_specInfo=True,
                                     output_srcs=True)

    peaks = resultDict["Peaks"]

    max_ts = tscube.tsmap.counts.max()
    (castro, test_dict) = tscube.test_spectra_of_peak(peaks[0])

    nll = castro[2]
    fig, ax = plotNLL_v_Flux(nll, flux_type)

    fig2, ax2, im2, ztmp2 = plotCastro(castro, ylims=flux_lims, nstep=100)

    spec_pl = test_dict["PowerLaw"]["Spectrum"]
    spec_lp = test_dict["LogParabola"]["Spectrum"]
    spec_pc = test_dict["PLExpCutoff"]["Spectrum"]

    fig3, ax3 = plotSED(castro, ylims=flux_lims, TS_thresh=2.0,
                        specVals=[spec_pl])

    result_pl = test_dict["PowerLaw"]["Result"]
    result_lp = test_dict["LogParabola"]["Result"]
    result_pc = test_dict["PLExpCutoff"]["Result"]
    ts_pl = test_dict["PowerLaw"]["TS"]
    ts_lp = test_dict["LogParabola"]["TS"]
    ts_pc = test_dict["PLExpCutoff"]["TS"]

    print("TS for PL index = 2:  %.1f" % max_ts)
    print("Cumulative TS:        %.1f" % castro.ts_vals().sum())
    print("TS for PL index free: %.1f (Index = %.2f)" %
          (ts_pl, result_pl[1]))
    print("TS for LogParabola:   %.1f (Index = %.2f, Beta = %.2f)" %
          (ts_lp, result_lp[1], result_lp[2]))
    print("TS for PLExpCutoff:   %.1f (Index = %.2f, E_c = %.2f)" %
          (ts_pc, result_pc[1], result_pc[2]))
