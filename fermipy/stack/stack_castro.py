#!/usr/bin/env python
#

"""
Interface to Dark Matter spectra
"""
from __future__ import absolute_import, division, print_function

import numpy as np

from astropy.table import Table, Column

from fermipy import castro
from fermipy.stats_utils import create_prior_functor

from fermipy.stack.lnl_norm_prior import LnLFn_norm_prior

REF_ASTRO = 1.0
REF_STACK = 1.0
REF_NORM = 1.0

class StackCastroData(castro.CastroData_Base):
    """ This class wraps the data needed to make a "Castro" plot,
    namely the log-likelihood as a function of the normalizaiton
    and index for a series of spectra
    """

    def __init__(self, norm_vals, nll_vals, nll_offsets, **kwargs):
        """ C'tor

        Parameters
        ----------
        norm_vals : `~numpy.ndarray`
           The normalization values ( n_index X N array, where N is the
           number of sampled values for each bin )
           Note that these should be the true values, with the
           reference J-value (or D-value) included, and _NOT_ the values w.r.t. to the
           reference J-value (or D-value) spectrum.

        nll_vals : `~numpy.ndarray`
           The log-likelihood values ( n_index X N array, where N is
           the number of sampled values for each bin )

        nll_offsets : `~numpy.ndarray`
           The maximum log-likelihood values for each index ( n_index array )
           These can be used to compare the fit-quality between xscan_vals.


        Keyword arguments
        -----------------

        spec_name : str
           The name of the spectral function we are using

        astro_value : float
           The normalization of the predicted flux for this target

        xscan_name : str
           Name of the spectral parameter we scan over

        xscan_vals : list
           Values of the spectral parameter we scan over

        other_pars : dict
           Values of the other spectral parameters

        astro_prior : `dmsky.priors.PriorFunctor`
           The prior on the J-factor (or D-factor)

        prior_applied : bool
           If true, then the prior has already been applied

        ref_astro : float
           Reference value for normalization

        ref_stack : float
           Reference value for the stacking variable

        norm_type : str
           Normalization type: 'stack' or 'norm'

        """
        self._spec_name = kwargs['spec_name']
        self._astro_value = kwargs.get('astro_value', REF_ASTRO)
        self._xscan_name = kwargs.get('xscan_name', None)
        self._xscan_vals = kwargs.get('xscan_vals', [1.0])
        self._other_pars = kwargs.get('other_pars', {})
        self._astro_prior = kwargs.get('astro_prior', None)
        self._prior_applied = kwargs.get('prior_applied', True)
        self._ref_astro = kwargs.get('ref_astro', REF_ASTRO)
        self._ref_stack = kwargs.get('ref_stack', REF_STACK)
        norm_type = kwargs.get('norm_type', 'norm')

        super(StackCastroData, self).__init__(norm_vals, nll_vals,
                                              nll_offsets, norm_type=norm_type)

    @property
    def spec_name(self):
        """ The name of the spectral function """
        return self._spec_name

    @property
    def astro_value(self):
        """ The astrophysical noramlization """
        return self._astro_value

    @property
    def xscan_name(self):
        """ The name of variable we are scanning over """
        return self._xscan_name

    @property
    def n_xscan_vals(self):
        """ The number of xscan_vals tested """
        return len(self._xscan_vals)

    @property
    def xscan_vals(self):
        """ The xscan_vals tested (in GeV) """
        return self._xscan_vals

    @property
    def other_pars(self):
        """ The values of the other parameters (aside from the normalization and the scan variable)"""
        return self._other_pars

    @property
    def astro_prior(self):
        """ The prior on the astrophysical normalization """
        return self._astro_prior

    @property
    def prior_mean(self):
        """ The mean on the prior on the astrophysical normalization

        Note that this is a multiplicative prior, so we except a mean of 1
        """
        if self._astro_prior is None:
            return 1.0
        return self.astro_prior.mean()

    @property
    def prior_sigma(self):
        """ The width of the prior on the astrophysical J or D factor.

        This is actually the width parameter given to the prior function.
        The exact meaning depends on the function type being used.
        """
        if self._astro_prior is None:
            return 0.0
        return self.astro_prior.sigma()

    @property
    def prior_type(self):
        """ The function type of the astrophysical prior

        See '~fermipy.stats_utils.create_prior_functor' for recognized types.
        """
        if self._astro_prior is None:
            return "none"
        return self.astro_prior.funcname

    @property
    def prior_applied(self):
        """ Has the prior already been applied """
        return self._prior_applied

    @property
    def ref_astro(self):
        """ Reference value for normalization """
        return self._ref_astro

    @property
    def ref_stack(self):
        """ Reference value for stacking variable """
        return self._ref_stack


    def x_edges(self):
        """ Make a reasonable set of bin edges for plotting

        To do this we expand relative to the index points by half the bid width in either direction
        """
        xvals = self._xscan_vals.copy()
        half_widths = (xvals[1:] - xvals[0:-1]) /2
        last_val = xvals[-1] + half_widths[-1]
        xvals[0:-1] -= half_widths
        xvals[-1] -= half_widths[-1]
        xvals = np.append(xvals, last_val)
        return last_val


    def build_lnl_fn(self, normv, nllv):
        """ Build a function to return the likelihood value arrays of
        normalization and likelihood values.

        Parameters
        ----------

        normv : `numpy.array`
            Set of test normalization values

        nllv : `numpy.array`
            Corresponding set of negative log-likelihood values

        Returns
        -------

        output : `fermipy.castro.LnLFn` or `LnLFn_norm_prior`
            Object that can compute the negative log-likelihood with
            the prior included (if requested)

        """
        lnlfn = castro.LnLFn(normv, nllv, self._norm_type)
        if self._astro_prior is None:
            return lnlfn
        if self._prior_applied:
            return lnlfn
        return LnLFn_norm_prior(lnlfn, self._astro_prior)


    def build_ref_columns(self):
        """Build the columns with reference data

        Returns
        -------
        collist : list
            The list of newly made columns

        valdict : dcit
            Dictionary with the corresponding values
        """
        col_astro_val = Column(name="astro_value", dtype=float)
        col_ref_astro = Column(name="ref_astro", dtype=float)
        col_ref_stack = Column(name="ref_stack", dtype=float)
        col_xscan_vals = Column(name="scan_%s" % self._xscan_name, dtype=float, shape=self._xscan_vals.shape)
        valdict = {"astro_value": self.astro_value,
                   "ref_astro": self.ref_astro,
                   "ref_stack": self.ref_stack,
                   "scan_%s" % self._xscan_name: self._xscan_vals}
        collist = [col_astro_val, col_ref_astro, col_ref_stack, col_xscan_vals]

        for par_name, par_value in self._other_pars.items():
            col_name = "par_%s" % par_name
            col_par = Column(name=col_name, dtype=float, data=par_value)
            collist.append(col_par)
            valdict[col_name] = par_value

        if self._astro_prior is not None:
            col_prior_type = Column(name="prior_type", dtype="S16")
            col_prior_mean = Column(name="prior_mean", dtype=float)
            col_prior_sigma = Column(name="prior_sigma", dtype=float)
            col_prior_applied = Column(name="prior_applied", dtype=bool)
            collist += [col_prior_type, col_prior_mean,
                        col_prior_sigma, col_prior_applied]
            valdict["prior_type"] = self.prior_type
            valdict["prior_mean"] = self.prior_mean
            valdict["prior_sigma"] = self.prior_sigma
            valdict["prior_applied"] = self.prior_applied

        return collist, valdict



    def build_scandata_table(self, norm_type=None):
        """Build a FITS table with likelihood scan data

        Parameters
        ----------

        norm_type : str or None
            Type of normalization to use.  Valid options are:

            * norm : Self-normalized
            * stack : Reference values of stacking variable

        Returns
        -------

        table : `astropy.table.Table`
            The table has these columns

        astro_value : float
            The astrophysical normalization for this target
        ref_astro : float
            The reference normalization used to build `StackSpecTable`
        ref_stack : float
            The reference stacking variable used to build `StackSpecTable`

        norm_scan : array
            The test values of <sigmav> (or tau)
        dloglike_scan : array
            The corresponding values of the negative log-likelihood

        """
        shape = self._norm_vals.shape
        collist, valdict = self.build_ref_columns()

        col_normv = Column(name="norm_scan", dtype=float, shape=shape)
        col_dll = Column(name="dloglike_scan", dtype=float, shape=shape)
        col_offset = Column(name="dloglike_offset", dtype=float, shape=shape[0])

        collist += [col_normv, col_dll, col_offset]

        if norm_type in ['stack']:
            norm_vals = self._norm_vals / self.ref_stack
        elif norm_type in ['norm']:
            norm_vals = self._norm_vals
        else:
            raise ValueError('Unrecognized normalization type: %s' % norm_type)

        valdict.update({"norm_scan": norm_vals,
                        "dloglike_scan": -1 * self._nll_vals,
                        "dloglike_offset": -1 * self._nll_offsets})

        tab = Table(data=collist)
        tab.add_row(valdict)
        return tab


    def build_limits_table(self, limit_dict):
        """Build a FITS table with limits data

        Paramters
        ---------

        limit_dict : dict
            Dictionary from limit names to values


        Returns
        -------

        table : `astropy.table.Table`
            The table has these columns

        astro_value : float
            The astrophysical J-factor for this target
        ref_astro : float
            The reference J-factor used to build `StackSpecTable`
        ref_stack : float
            The reference <sigmav> used to build `StackSpecTable`
        <LIMIT> : array
            The upper limits

        If a prior was applied these additional colums will be present

        prior_type : str
            Key specifying what kind of prior was applied
        prior_mean : float
            Central value for the prior
        prior_sigma : float
            Width of the prior
        prior_applied : bool
            Flag to indicate that the prior was applied

        """
        collist, valdict = self.build_ref_columns()

        for k, v in limit_dict.items():
            collist.append(Column(name=k, dtype=float, shape=v.shape))
            valdict[k] = v

        tab = Table(data=collist)
        tab.add_row(valdict)
        return tab


    @staticmethod
    def create_prior_from_table(tab_s):
        """
        Parameters
        ----------

        tab_s : `astropy.table.Table`
            Table with likelihood scan data

        Returns
        -------

        prior : `LnLFn`
            Object with prior

        prior_applied : `bool`
            True if the prior has already been applied
        """
        astro_value = np.squeeze(np.array(tab_s['astro_value']))
        try:
            astro_priortype = tab_s['prior_type']
        except KeyError:
            astro_priortype = None

        if astro_priortype is not None:
            prior_mean = np.squeeze(np.array(tab_s['prior_mean']))
            prior_sigma = np.squeeze(np.array(tab_s['prior_sigma']))
            prior_applied = np.squeeze(np.array(tab_s['prior_applied']))
            prior_dict = dict(functype=astro_priortype,
                              mu=prior_mean,
                              sigma=prior_sigma)
            prior_dict['astro_value'] = astro_value
            prior = create_prior_functor(prior_dict)
        else:
            prior = None
            prior_applied = True

        return prior, prior_applied


    @staticmethod
    def extract_context_from_tab(tab_s):
        """
        Parameters
        ----------

        tab_s : `astropy.table.Table`
            Table with likelihood scan data

        Returns
        -------
        odict : `dict`
        """
        other_pars = {}
        for col in tab_s.columns:
            if col.name[0:6] == "scan_":
                xscan_name = col.name[6:]
                xscan_data = np.squeeze(np.array(col.array))
            elif col.name[0:5] == "par_":
                par_name = col.name[4:]
                par_data = np.squeeze(np.array(col.array))
                other_pars[par_name] = par_data

        odict = dict(astro_value=np.squeeze(np.array(tab_s['astro_value'])),
                     ref_astro=np.squeeze(np.array(tab_s['ref_astro'])),
                     ref_stack=np.squeeze(np.array(tab_s['ref_stack'])),
                     xscan_name=xscan_name,
                     xscan_data=xscan_data,
                     other_pars=other_pars)
        return odict


    @classmethod
    def create_from_stack(cls, components, **kwargs):
        """ Create a StackCastroData object by stacking a series of StackCastroData objects

        Parameters
        ----------
        components : list
           List of `StackCastroData` objects we are stacking

        Keyword Arguments
        -----------------
        nystep : int
            Number of steps in <sigmav> to take in sampling.

        ylims : tuple
            Limits of range of <sigmav> to scan

        weights : list or None
            List of weights to apply to components

        ref_astro : float
            Reference J-factor value

        ref_stack : float
            Refernece <sigmav> value

        Returns
        -------

        output : `StackCastroData`
            Object with the stacking variable-space likelihoods

        """
        if not components:
            return None

        ysteps = kwargs.pop('ysteps')
        weights = kwargs.pop('weights', None)
        shape = (components[0].nx, len(ysteps))
        norm_vals, nll_vals, nll_offsets = castro.CastroData_Base.stack_nll(shape, components, ysteps, weights)

        kwcopy = kwargs.copy()
        # copy stuff from component 0
        for attr_name in ['spec_name', 'xscan_name', 'xscan_vals',
                          'other_pars', 'astro_prior', 'ref_astro', 'ref_stack']:
            kwcopy[attr_name] = components[0].__dict__[attr_name]

        kwcopy['astro_value'] = None
        kwcopy['prior_applied'] = True

        return cls(norm_vals, nll_vals, nll_offsets, **kwcopy)



    @classmethod
    def create_from_tables(cls, tab_s, norm_type):
        """ Create a StackCastroData object from likelihood scan tables

        Parameters
        ----------

        tab_s : `astropy.table.Table`
            Table with likelihood scan data

        norm_type : str
            Type of normalization to use.  Valid options are:

            * norm : Self-normalized
            * stack : Reference values of stacking variable

        Returns
        -------

        output : `StackCastroData`
            Object with the stacking-space likelihoods

        """
        context = cls.extract_context_from_tab(tab_s)

        ref_stack = context['ref_stack']

        if norm_type in ['stack']:
            norm_vals = np.squeeze(np.array(tab_s['norm_scan']*ref_stack))
        elif norm_type in ['norm']:
            norm_vals = np.squeeze(np.array(tab_s['norm_scan']))
        else:
            raise ValueError('Unrecognized normalization type: %s' % norm_type)

        nll_vals = -np.squeeze(np.array(tab_s['dloglike_scan']))
        nll_offsets = -np.squeeze(np.array(tab_s['dloglike_offset']))

        prior, prior_applied = cls.create_prior_from_table(tab_s)

        # copy stuff from component 0
        context['astro_prior'] = prior
        context['prior_applied'] = prior_applied
        context['norm_type'] = norm_type
        context['spec_name'] = tab_s.name

        return cls(norm_vals, nll_vals, nll_offsets, **context)


    @classmethod
    def create_from_fitsfile(cls, filepath, spec, norm_type):
        """ Create a StackCastroData object likelihood scan and index tables in FITS file

        Parameters
        ----------

        filepath : str
            Path to likelihood scan data.

        spec : str
            stacking spectra

        norm_type : str
            Type of normalization to use.  Valid options are:

            * norm : Self-normalized
            * stack : Reference values of stacking variable

        Returns
        -------

        output : `StackCastroData`
            Object with the stacking variable-space likelihoods

        """
        tab_s = Table.read(filepath, hdu=spec)
        return cls.create_from_tables(tab_s, norm_type=norm_type)
