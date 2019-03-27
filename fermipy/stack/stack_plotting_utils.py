#!/usr/bin/env python
#

"""
Utilities to plot dark matter analyses
"""

import numpy as np
from fermipy import sed_plotting



ENERGY_AXIS_LABEL = r'Energy [MeV]'
ENERGY_FLUX_AXIS_LABEL = r'Energy Flux [MeV s$^{-1}$ cm$^{-2}$]'
FLUX_AXIS_LABEL = r'Flux [ph s$^{-1}$ cm$^{-2}$]'
DELTA_LOGLIKE_AXIS_LABEL = r'$\Delta \log\mathcal{L}$'
INDEX_AXIS_LABEL = r'$\Gamma$'
NORM_AXIS_LABEL = r'Norm'
NORM_UNC_AXIS_LABEL = r'\delta Norm'


def plot_stack_spectra_by_spec(stack_spec_table, index=100.,
                               spec_type='eflux', ylims=(1e-12, 1e-8)):
    """ Make a plot of the stacking spectra.

    Parameters
    ----------

    stack_spec_table : `StackSpecTable`
        The object with the spectra

    index : float
        The spectral index

    spec_type : str
        Spectral type, one of 'flux' or 'eflux'

    ylims : tuple
        Y-axis limits for plot


    Returns
    -------

    fig : `matplotlib.Figure`
        The figure

    axis : `matplotlib.Axes`
        The plot axes

    leg : `matplotlib.Legend`
        The figure legend

    """
    import matplotlib.pyplot as plt

    spec_names = stack_spec_table.spectype_names
    spec_ids = stack_spec_table.spectype_map.keys()
    spec_idx_list = stack_spec_table.spectype_map.values()
    energies = stack_spec_table.ebin_refs()

    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_xlim((energies[0], energies[-1]))
    axis.set_ylim(ylims)
    axis.set_xlabel(ENERGY_AXIS_LABEL)
    if spec_type == 'eflux':
        axis.set_ylabel(ENERGY_FLUX_AXIS_LABEL)
    elif spec_type == 'flux':
        axis.set_ylabel(FLUX_AXIS_LABEL)

    for spec, spec_id, idx_list in zip(spec_names, spec_ids, spec_idx_list):
        spec_indices = stack_spec_table.indices(spec_id).data
        index_idx = np.abs(spec_indices - index).argmin()
        table_idx = idx_list[index_idx]
        spectrum = stack_spec_table._s_table[table_idx]["ref_%s" % spec_type]
        axis.plot(energies, spectrum, label=spec)

    leg = axis.legend(loc="best", ncol=2, fontsize=10)
    return fig, axis, leg


def plot_stack_spectra_by_index(stack_spec_table, spec='powerlaw',
                                spec_type='eflux', ylims=(1e-12, 1e-6)):
    """ Make a plot of the stacking spectra.

    Parameters
    ----------

    stack_spec_table : `StackSpecTable`
        The object with the spectra

    spec : str
        The stacking spectrum

    spec_type : str
        Spectral type, one of 'flux' or 'eflux'

    ylims : tuple
        Y-axis limits for plot


    Returns
    -------

    fig : `matplotlib.Figure`
        The figure

    axis : `matplotlib.Axes`
        The plot axes

    leg : `matplotlib.Legend`
        The figure legend

    """
    import matplotlib.pyplot as plt

    spec_id = stack_spec_table.spectype_rev_map[spec]
    spec_idx_list = stack_spec_table.spectype_map[spec_id]
    energies = stack_spec_table.ebin_refs()

    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_xlim((energies[0], energies[-1]))
    axis.set_ylim(ylims)
    axis.set_xlabel(ENERGY_AXIS_LABEL)
    if spec_type == 'eflux':
        axis.set_ylabel(ENERGY_FLUX_AXIS_LABEL)
    else:
        axis.set_ylabel(FLUX_AXIS_LABEL)

    indices = stack_spec_table.indices(spec_id)
    for table_idx, index in zip(spec_idx_list, indices):
        spectrum = stack_spec_table._s_table[table_idx]["ref_%s" % spec_type]
        axis.plot(energies, spectrum, label="%.1F GeV" % index)

    leg = axis.legend(loc="best", ncol=2, fontsize=10)
    return fig, axis, leg


def plot_stack_castro(castro_stack, ylims=None, nstep=100, zlims=None, global_min=False):
    """ Make a color plot (1castro plot) of the delta log-likelihood as a function of
    stacking normalization and index

    Parameters
    ----------

    castro_stack :  `StackCastroData`
        Object with the log-likelihood v. normalization for each index

    ylims      : tuple
        Y-axis limits for the plot

    nstep      : int
        Number of y-axis steps to plot for each energy bin

    zlims      : tuple
        z-axis limits

    global_min : bool
        If True plot likelihood w.r.t. the global minimimum.

    """
    return sed_plotting.plotCastro_base(castro_stack,
                                        ylims=ylims,
                                        xlabel=INDEX_AXIS_LABEL,
                                        ylabel=NORM_AXIS_LABEL,
                                        nstep=nstep,
                                        zlims=zlims,
                                        global_min=global_min)


def plot_castro_nuiscance(xlims, ylims, zvals, zlims=None):
    """ Make a castro plot including the effect of the nuisance parameter
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.set_yscale('log')
    axis.set_xlim(xlims)
    axis.set_ylim(ylims)
    axis.set_xlabel(NORM_UNC_AXIS_LABEL)
    axis.set_ylabel(NORM_AXIS_LABEL)

    if zlims is None:
        zmin = 0
        zmax = 10.
    else:
        zmin = zlims[0]
        zmax = zlims[1]

    image = axis.imshow(zvals, extent=[xlims[0], xlims[-1], ylims[0], ylims[-1]],
                        origin='lower', aspect='auto', interpolation='nearest',
                        vmin=zmin, vmax=zmax, cmap=cm.jet_r)
    return fig, axis, image


def plot_nll(nll_dict, xlims=None, nstep=50, ylims=None):
    """ Plot the -log(L) as a function of sigmav for each object in a dict
    """
    import matplotlib.pyplot as plt

    if xlims is None:
        xmin = 1e-28
        xmax = 1e-24
    else:
        xmin = xlims[0]
        xmax = xlims[1]

    xvals = np.logspace(np.log10(xmin), np.log10(xmax), nstep)
    fig = plt.figure()
    axis = fig.add_subplot(111)

    axis.set_xlim((xmin, xmax))
    if ylims is not None:
        axis.set_ylim((ylims[0], ylims[1]))

    axis.set_xlabel(NORM_AXIS_LABEL)
    axis.set_ylabel(DELTA_LOGLIKE_AXIS_LABEL)
    axis.set_xscale('log')

    for lab, nll in nll_dict.items():
        yvals = nll.interp(xvals)
        yvals -= yvals.min()
        axis.plot(xvals, yvals, label=lab)

    leg = axis.legend(loc="upper left")
    return fig, axis, leg


def plot_comparison(nll, nstep=25, xlims=None):
    """ Plot the comparison between differnt version of the -log(L)
    """
    import matplotlib.pyplot as plt

    if xlims is None:
        xmin = nll._lnlfn.interp.xmin
        xmax = nll._lnlfn.interp.xmax
    else:
        xmin = xlims[0]
        xmax = xlims[1]

    xvals = np.linspace(xmin, xmax, nstep)
    yvals_0 = nll.straight_loglike(xvals)
    yvals_1 = nll.profile_loglike(xvals)
    yvals_2 = nll.marginal_loglike(xvals)

    ymin = min(yvals_0.min(), yvals_1.min(), yvals_2.min(), 0.)
    ymax = max(yvals_0.max(), yvals_1.max(), yvals_2.max(), 0.5)

    fig = plt.figure()
    axis = fig.add_subplot(111)

    axis.set_xlim((xmin, xmax))
    axis.set_ylim((ymin, ymax))

    axis.set_xlabel(NORM_AXIS_LABEL)
    axis.set_ylabel(DELTA_LOGLIKE_AXIS_LABEL)

    axis.plot(xvals, yvals_0, 'r', label=r'Simple $\log\mathcal{L}$')
    axis.plot(xvals, yvals_1, 'g', label=r'Profile $\log\mathcal{L}$')
    #axis.plot(xvals,yvals_2,'b', label=r'Marginal $\log\mathcal{L}$')

    leg = axis.legend(loc="upper left")

    return fig, axis, leg


def plot_stacked(sdict, xlims, ibin=0):
    """ Stack a set of -log(L) curves and plot the stacked curve
    as well as the individual curves
    """
    import matplotlib.pyplot as plt
    ndict = {}

    for key, val in sdict.items():
        ndict[key] = val[ibin]

    #mles = np.array([n.mle() for n in ndict.values()])

    fig = plt.figure()
    axis = fig.add_subplot(111)

    xvals = np.linspace(xlims[0], xlims[-1], 100)

    axis.set_xlim((xvals[0], xvals[-1]))
    axis.set_xlabel(NORM_AXIS_LABEL)
    axis.set_ylabel(DELTA_LOGLIKE_AXIS_LABEL)

    for key, val in ndict.items():
        yvals = val.interp(xvals)
        if key.lower() == "stacked":
            axis.plot(xvals, yvals, lw=3, label=key)
        else:
            yvals -= yvals.min()
            axis.plot(xvals, yvals, label=key)
    leg = axis.legend(loc="upper left", fontsize=10, ncol=2)
    return fig, axis, leg


def plot_limits_from_arrays(ldict, xlims, ylims, bands=None):
    """ Plot the upper limits as a function of stacking normalization and index

    Parameters
    ----------

    ldict      : dict
        A dictionary of strings pointing to pairs of `np.array` objects,
        The keys will be used as labels

    xlims      : tuple
        x-axis limits for the plot

    ylims      : tupel
        y-axis limits for the plot

    bands      : dict
        Dictionary with the expected limit bands


    Returns
    -------

    fig : `matplotlib.Figure`
        The figure

    axis : `matplotlib.Axes`
        The plot axes

    leg : `matplotlib.Legend`
        The figure legend

    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.set_xlabel(INDEX_AXIS_LABEL)
    axis.set_ylabel(NORM_AXIS_LABEL)

    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_xlim((xlims[0], xlims[1]))
    axis.set_ylim((ylims[0], ylims[1]))

    if bands is not None:
        plot_expected_limit_bands(axis, bands)

    for key, val in ldict.items():
        xvals = val[0]
        yvals = val[1]
        if key.lower() == "stacked":
            axis.plot(xvals, yvals, lw=3, label=key)
        else:
            axis.plot(xvals, yvals, label=key)

    leg = axis.legend(loc="upper left")  # ,fontsize=10,ncol=2)
    return fig, axis, leg


def plot_expected_limit_bands(axis, bands):
    """Add the expected limit bands to a plot

    Parameters
    ----------

    axis : `matplotlib.Axes`
        The axes we are adding the bands to

    bands : dict
        Dictionary with the bands

    """
    indices = bands['indices']

    axis.fill_between(indices, bands['q02'], bands['q97'], color='yellow')
    axis.fill_between(indices, bands['q16'], bands['q84'], color='green')
    axis.plot(indices, bands['median'], color='gray')


def plot_mc_truth(axis, mc_model):
    """Add a marker showing the true injected signal to the plot

    Parameters
    ----------

    axis : `matplotlib.Axes`
        The axes we are adding the bands to

    mc_model : dict
        Dictionary with truth

    """
    norm = mc_model['norm']['value']
    index = mc_model['index']['value']
    axis.scatter([index], [norm])


def plot_limits(sdict, xlims, ylims, alpha=0.05):
    """ Plot the upper limits as a function of stacking normalization and index

    Parameters
    ----------

    sdict      : dict
        A dictionary of `StackCastroData` objects with the log-likelihood v. normalization for each energy bin

    xlims      : tuple
        x-axis limits

    ylims      : tuple
        y-axis limits

    alpha      : float
        Confidence level to use in setting limits = 1 - alpha

    """

    ldict = {}
    for key, val in sdict.items():
        ldict[key] = (val.indices, val.getLimits(alpha))
    return plot_limits_from_arrays(ldict, xlims, ylims)


def compare_limits(sdict, xlims, ylims, alpha=0.05):
    """ Plot the upper limits as a functino of stacking normalization and index

    Paramters
    ---------

    sdict      : dict
        Dictionary with limits and keys

    xlims      : tuple
        x-axis limits

    ylims      : tuple
        y-axis limits

    alpha      : float
        Confidence level to use in setting limits = 1 - alpha


    Returns
    -------

    fig : `matplotlib.Figure`
        The figure

    axis : `matplotlib.Axes`
        The plot axes

    leg : `matplotlib.Legend`
        The figure legend

    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    axis = fig.add_subplot(111)

    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_xlim((xlims[0], xlims[1]))
    axis.set_ylim((ylims[0], ylims[1]))
    axis.set_xlabel(INDEX_AXIS_LABEL)
    axis.set_ylabel(NORM_AXIS_LABEL)

    for key, val in sdict.items():
        xvals = val.indices
        yvals = val.getLimits(alpha)
        axis.plot(xvals, yvals, label=key)

    leg = axis.legend(loc="upper left", fontsize=10, ncol=2)
    return fig, axis, leg


def plot_limit(stack_castro_data, ylims, alpha=0.05):
    """Plot the limit curve for a given StackCastroData object

    Parameters
    ----------

    stack_castro_data :  `StackCastroData`
        Object with the log-likelihood v. normalization for each index

    ylims      : tuple
        Y-axis limits for the plot

    alpha      : float
        Confidence level to use in setting limits = 1 - alpha


    Returns
    -------

    fig : `matplotlib.Figure`
        The figure

    axis : `matplotlib.Axes`
        The plot axes

    """
    import matplotlib.pyplot as plt

    xbins = stack_castro_data.indices
    xmin = xbins[0]
    xmax = xbins[-1]

    fig = plt.figure()
    axis = fig.add_subplot(111)

    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_xlabel(INDEX_AXIS_LABEL)
    axis.set_ylabel(NORM_AXIS_LABEL)
    axis.set_xlim((xmin, xmax))

    if ylims is not None:
        axis.set_ylim((ylims[0], ylims[1]))

    yvals = stack_castro_data.getLimits(alpha)
    if yvals.shape[0] == xbins.shape[0]:
        xvals = xbins
    else:
        xvals = np.sqrt(xbins[0:-1] * xbins[1:])
    axis.plot(xvals, yvals)

    return fig, axis
