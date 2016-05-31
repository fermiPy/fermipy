"""
Utilities for dealing with SEDs

Many parts of this code are taken from dsphs/like/lnlfn.py by 
  Matthew Wood <mdwood@slac.stanford.edu>
  Alex Drlica-Wagner <kadrlica@slac.stanford.edu>
"""

from __future__ import absolute_import, division, print_function, \
    unicode_literals

import copy
import logging
import os

import numpy as np
from scipy.interpolate import UnivariateSpline, splrep, splev
import scipy.optimize as opt
import scipy.special as spf
from scipy.integrate import quad
import scipy

import pyLikelihood as pyLike

import astropy.io.fits as pf
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column

import fermipy.config
import fermipy.defaults as defaults
import fermipy.utils as utils
import fermipy.gtutils as gtutils
import fermipy.roi_model as roi_model
from fermipy.wcs_utils import wcs_add_energy_axis
from fermipy.fits_utils import read_energy_bounds, read_spectral_data
from fermipy.skymap import read_map_from_fits, Map
from fermipy.logger import Logger
from fermipy.logger import logLevel
from fermipy.sourcefind import find_peaks, refine_peak
from fermipy.spectrum import SpectralFunction

from LikelihoodState import LikelihoodState

# Some useful functions

FluxTypes = ['NORM','FLUX','EFLUX','NPRED','DIF_FLUX','DIF_EFLUX']

PAR_NAMES = {"PowerLaw":["Prefactor","Index"],
             "LogParabola":["norm","alpha","beta"],
             "PLExpCutoff":["Prefactor","Index1","Cutoff"]}

class SEDGenerator(object):
    """Mixin class that provides SED functionality to
    `~fermipy.gtanalysis.GTAnalysis`."""
    
    def sed(self, name, profile=True, loge_bins=None, **kwargs):
        """Generate a spectral energy distribution (SED) for a source.  This
        function will fit the normalization of the source in each
        energy bin.  By default the SED will be generated with the
        analysis energy bins but a custom binning can be defined with
        the ``loge_bins`` parameter.  

        Parameters
        ----------
        name : str
            Source name.

        prefix : str
           Optional string that will be prepended to all output files
           (FITS and rendered images).
            
        profile : bool
            Profile the likelihood in each energy bin.

        loge_bins : `~numpy.ndarray`
            Sequence of energies in log10(E/MeV) defining the edges of
            the energy bins.  If this argument is None then the
            analysis energy bins will be used.  The energies in this
            sequence must align with the bin edges of the underyling
            analysis instance.

        bin_index : float
            Spectral index that will be use when fitting the energy
            distribution within an energy bin.

        use_local_index : bool
            Use a power-law approximation to the shape of the global
            spectrum in each bin.  If this is false then a constant
            index set to `bin_index` will be used.

        fix_background : bool
            Fix background components when fitting the flux
            normalization in each energy bin.  If fix_background=False
            then all background parameters that are currently free in
            the fit will be profiled.  By default fix_background=True.

        ul_confidence : float
            Set the confidence level that will be used for the
            calculation of flux upper limits in each energy bin.

        cov_scale : float
            Scaling factor that will be applied when setting the
            gaussian prior on the normalization of free background
            sources.  If this parameter is None then no gaussian prior
            will be applied.

        write_fits : bool

        write_npy : bool
            
        Returns
        -------
        sed : dict 
           Dictionary containing output of the SED analysis.  This
           dictionary is also saved to the 'sed' dictionary of the
           `~fermipy.roi_model.Source` instance.

        """

        name = self.roi.get_source_by_name(name).name

        prefix = kwargs.get('prefix','')
        write_fits = kwargs.get('write_fits',True)
        
        self.logger.info('Computing SED for %s' % name)
        
        o = self._make_sed(name,profile,loge_bins,**kwargs)
        
        self._plotter.make_sed_plot(self, name, **kwargs)

        if write_fits:
            self._make_sed_fits(o,name,**kwargs)
        self.logger.info('Finished SED')
        
        return o

    def _make_sed_fits(self,sed,name,**kwargs):

        prefix = kwargs.get('prefix',None)

        name = name.lower().replace(' ', '_')
        
        # Write a FITS file
        cols = [Column(name='E_MIN',dtype='f8',data=sed['emin'],unit='MeV'),
                Column(name='E_REF',dtype='f8',data=sed['ectr'],unit='MeV'),
                Column(name='E_MAX',dtype='f8',data=sed['emax'],unit='MeV'),
                Column(name='REF_DFDE_E_MIN',dtype='f8',
                       data=sed['ref_dfde_emin'],unit='ph / (MeV cm2 s)'),
                Column(name='REF_DFDE_E_MAX',dtype='f8',
                       data=sed['ref_dfde_emax'],unit='ph / (MeV cm2 s)'),
                Column(name='REF_DFDE',dtype='f8',
                       data=sed['ref_dfde'],unit='ph / (MeV cm2 s)'),
                Column(name='REF_FLUX',dtype='f8',
                       data=sed['ref_flux'],unit='ph / (cm2 s)'),
                Column(name='REF_EFLUX',dtype='f8',
                       data=sed['ref_eflux'],unit='MeV / (cm2 s)'),
                Column(name='REF_NPRED',dtype='f8',data=sed['ref_npred']),
                Column(name='NORM',dtype='f8',data=sed['norm']),
                Column(name='NORM_ERR',dtype='f8',data=sed['norm_err']),
                Column(name='NORM_ERRP',dtype='f8',data=sed['norm_err_hi']),
                Column(name='NORM_ERRN',dtype='f8',data=sed['norm_err_lo']),
                Column(name='NORM_UL',dtype='f8',data=sed['norm_ul95']),
                Column(name='TS',dtype='f8',data=sed['ts']),
                Column(name='LOGLIKE',dtype='f8', data=sed['loglike']),
                Column(name='NORM_SCAN',dtype='f8',data=sed['norm_scan']),
                Column(name='DLOGLIKE_SCAN',dtype='f8',
                       data=sed['dloglike_scan']),
                
                ]
                
        tab = Table(cols)
        filename = utils.format_filename(self.config['fileio']['workdir'],
                                         'sed', prefix=[prefix,name],
                                         extension='.fits')

        sed['file'] = os.path.basename(filename)
        
        tab.write(filename,format='fits',overwrite=True)
        
    def _make_sed(self, name, profile=True, loge_bins=None, **kwargs):

        # Extract options from kwargs
        config = copy.deepcopy(self.config['sed'])
        config.update(kwargs)

        bin_index = config['bin_index']
        use_local_index = config['use_local_index']
        fix_background = config['fix_background']
        ul_confidence = config['ul_confidence']
        cov_scale = config['cov_scale']        

        if loge_bins is None:
            loge_bins = self.log_energies
        else:
            loge_bins = np.array(loge_bins)

        nbins = len(loge_bins) - 1
        max_index = 5.0
        min_flux = 1E-30
        npts = self.config['gtlike']['llscan_npts']
        loge_bounds = self.loge_bounds
        
        # Output Dictionary
        o = {'logemin': loge_bins[:-1],
             'logemax': loge_bins[1:],
             'logectr': 0.5 * (loge_bins[:-1] + loge_bins[1:]),
             'emin' : 10 ** loge_bins[:-1],
             'emax' : 10 ** loge_bins[1:],
             'ectr' : 10 ** (0.5 * (loge_bins[:-1] + loge_bins[1:])),
             'ref_flux': np.zeros(nbins),
             'ref_eflux': np.zeros(nbins),
             'ref_dfde': np.zeros(nbins),
             'ref_dfde_emin': np.zeros(nbins),
             'ref_dfde_emax': np.zeros(nbins),
             'ref_e2dfde': np.zeros(nbins),
             'ref_npred': np.zeros(nbins),
             'norm': np.zeros(nbins),
             'flux': np.zeros(nbins),
             'eflux': np.zeros(nbins),
             'dfde': np.zeros(nbins),
             'e2dfde': np.zeros(nbins),
             'index': np.zeros(nbins),
             'npred': np.zeros(nbins),
             'ts': np.zeros(nbins),
             'loglike' : np.zeros(nbins),
             'norm_scan' : np.zeros((nbins,npts)),
             'dloglike_scan' : np.zeros((nbins,npts)),
             'loglike_scan' : np.zeros((nbins,npts)),
             'fit_quality': np.zeros(nbins),
             'lnlprofile': [],
             'correlation' : {},
             'model_flux' : {},
             'params' : {},
             'config': config
             }

        for t in ['norm','flux','eflux','dfde','e2dfde']:
            o['%s_err'%t] = np.zeros(nbins)*np.nan
            o['%s_err_hi'%t] = np.zeros(nbins)*np.nan
            o['%s_err_lo'%t] = np.zeros(nbins)*np.nan
            o['%s_ul95'%t] = np.zeros(nbins)*np.nan
            o['%s_ul'%t] = np.zeros(nbins)*np.nan
        
        saved_state = LikelihoodState(self.like)
        source = self.components[0].like.logLike.getSource(str(name))
        
        # Perform global spectral fit
        self._latch_free_params()
        self.free_sources(False,pars='shape')
        self.free_source(name)
        self.fit(loglevel=logging.DEBUG,update=False,
                 min_fit_quality=2)
        o['model_flux'] = self.bowtie(name)
        spectral_pars = gtutils.get_pars_dict_from_source(source)
        o['params'] = roi_model.get_params_dict(spectral_pars)
        
        self._restore_free_params()

        # Setup background parameters for SED
        self.free_sources(False,pars='shape')
        self.free_norm(name)

        if fix_background:
            self.free_sources(free=False)
        elif cov_scale is not None:
            self._latch_free_params()
            self.zero_source(name)
            self.fit(loglevel=logging.DEBUG,update=False)            
            srcNames = list(self.like.sourceNames())
            srcNames.remove(name)
            self.constrain_norms(srcNames, cov_scale)
            self.unzero_source(name)
            self._restore_free_params()
                                    
        # Precompute fluxes in each bin from global fit
        gf_bin_flux = []
        gf_bin_index = []
        for i, (logemin, logemax) in enumerate(zip(loge_bins[:-1],
                                                   loge_bins[1:])):

            emin = 10 ** logemin
            emax = 10 ** logemax
            delta = 1E-5
            f = self.like[name].flux(emin, emax)
            f0 = self.like[name].flux(emin * (1 - delta), emin * (1 + delta))
            f1 = self.like[name].flux(emax * (1 - delta), emax * (1 + delta))

            if f0 > min_flux:
                g = 1 - np.log10(f0 / f1) / np.log10(emin / emax)
                gf_bin_index += [g]
                gf_bin_flux += [f]
            else:
                gf_bin_index += [max_index]
                gf_bin_flux += [min_flux]

        old_spectrum = source.spectrum()
        self.like.setSpectrum(str(name), str('PowerLaw'))
        self.free_parameter(name, 'Index', False)
        self.set_parameter(name, 'Prefactor', 1.0, scale=1E-13,
                          true_value=False,
                          bounds=[1E-10, 1E10],
                          update_source=False)
        self.set_parameter(name, 'Scale', 1E3, scale=1.0,
                           bounds=[1, 1E6], update_source=False)
        
        src_norm_idx = -1        
        free_params = self.get_params(True)
        for j, p in enumerate(free_params):
            if not p['is_norm']:
                continue
            if p['is_norm'] and p['src_name'] == name:
                src_norm_idx = j

            o['correlation'][p['src_name']] =  np.zeros(nbins) * np.nan
        
        for i, (logemin, logemax) in enumerate(zip(loge_bins[:-1],
                                                   loge_bins[1:])):

            logectr = 0.5 * (logemin + logemax)
            emin = 10 ** logemin
            emax = 10 ** logemax
            ectr = 10 ** logectr
            ectr2 = ectr**2
            
            self.set_parameter(name, 'Scale', ectr, scale=1.0,
                               bounds=[1, 1E6], update_source=False)

            if use_local_index:
                o['index'][i] = -min(gf_bin_index[i], max_index)
            else:
                o['index'][i] = -bin_index

            self.set_norm(name, 1.0, update_source=False)
            self.set_parameter(name, 'Index', o['index'][i], scale=1.0,
                               update_source=False)
            self.like.syncSrcParams(str(name))

            ref_flux = self.like[name].flux(emin, emax)
            
            o['ref_flux'][i] = self.like[name].flux(emin, emax)
            o['ref_eflux'][i] = self.like[name].energyFlux(emin, emax)
            o['ref_dfde'][i] = self.like[name].spectrum()(pyLike.dArg(ectr))
            o['ref_dfde_emin'][i] = self.like[name].spectrum()(pyLike.dArg(emin))
            o['ref_dfde_emax'][i] = self.like[name].spectrum()(pyLike.dArg(emax))
            o['ref_e2dfde'][i] = o['ref_dfde'][i]*ectr2
            cs = self.model_counts_spectrum(name, logemin, logemax, summed=True)
            o['ref_npred'][i] = np.sum(cs)
                                   
            normVal = self.like.normPar(name).getValue()
            flux_ratio = gf_bin_flux[i] / ref_flux
            newVal = max(normVal * flux_ratio, 1E-10)
            self.set_norm(name, newVal, update_source=False)
            
            self.like.syncSrcParams(str(name))
            self.free_norm(name)
            self.logger.debug('Fitting %s SED from %.0f MeV to %.0f MeV' %
                              (name, emin, emax))
            self.set_energy_range(logemin, logemax)

            fit_output = self.fit(loglevel=logging.DEBUG,update=False)
            free_params = self.get_params(True)
            for j, p in enumerate(free_params):
                
                if not p['is_norm']:
                    continue
                
                o['correlation'][p['src_name']][i] = \
                    fit_output['correlation'][src_norm_idx,j]
            
            o['fit_quality'][i] = fit_output['fit_quality']

            prefactor = self.like[self.like.par_index(name, 'Prefactor')]

            flux = self.like[name].flux(emin, emax)
            eflux = self.like[name].energyFlux(emin, emax)
            dfde = self.like[name].spectrum()(pyLike.dArg(ectr))
            #prefactor.getTrueValue()

            o['norm'][i] = flux/o['ref_flux'][i]
            o['flux'][i] = flux
            o['eflux'][i] = eflux
            o['dfde'][i] = dfde
            o['e2dfde'][i] = dfde * ectr2

            cs = self.model_counts_spectrum(name, logemin,
                                            logemax, summed=True)
            o['npred'][i] = np.sum(cs)
            o['ts'][i] = max(self.like.Ts2(name, reoptimize=False), 0.0)
            o['loglike'][i] = -self.like()
            
            lnlp = self.profile_norm(name, logemin=logemin, logemax=logemax,
                                    savestate=False, reoptimize=True,
                                    npts=npts)

            o['loglike_scan'][i] = lnlp['loglike']
            o['dloglike_scan'][i] = lnlp['dloglike']
            o['norm_scan'][i] = lnlp['flux']/ref_flux            
            o['lnlprofile'] += [lnlp]

            ul_data = utils.get_parameter_limits(lnlp['flux'], lnlp['dloglike'])

            o['norm_err_hi'][i] = ul_data['err_hi']/ref_flux
            o['norm_err_lo'][i] = ul_data['err_lo']/ref_flux
            
            if np.isfinite(ul_data['err_lo']):
                o['norm_err'][i] = 0.5*(ul_data['err_lo'] +
                                        ul_data['err_hi'])/ref_flux
            else:
                o['norm_err'][i] = ul_data['err_hi']/ref_flux

            o['norm_ul95'][i] = ul_data['ul']/ref_flux

            ul_data = utils.get_parameter_limits(lnlp['flux'],
                                                 lnlp['dloglike'], 
                                                 ul_confidence=ul_confidence)
            o['norm_ul'][i] = ul_data['ul']/ref_flux

        for t in ['flux','eflux','dfde','e2dfde']:
            
            o['%s_err'%t] = o['norm_err']*o['ref_%s'%t]
            o['%s_err_hi'%t] = o['norm_err_hi']*o['ref_%s'%t]
            o['%s_err_lo'%t] = o['norm_err_lo']*o['ref_%s'%t]
            o['%s_ul95'%t] = o['norm_ul95']*o['ref_%s'%t]
            o['%s_ul'%t] = o['norm_ul']*o['ref_%s'%t]
        
        self.set_energy_range(loge_bounds[0], loge_bounds[1])
        self.like.setSpectrum(str(name), old_spectrum)
        saved_state.restore()
        self._sync_params(name)

        if cov_scale is not None:
            self.remove_priors()
        
        src = self.roi.get_source_by_name(name)
        src.update_data({'sed': copy.deepcopy(o)})
        
        return o
        


if __name__ == "__main__":

    pass
