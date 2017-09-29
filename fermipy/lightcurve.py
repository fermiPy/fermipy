# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import os
import copy
import shutil
import logging
import yaml
import json
from collections import OrderedDict
from multiprocessing import Pool
from functools import partial
import sys

import numpy as np

import fermipy.config as config
import fermipy.utils as utils
import fermipy.gtutils as gtutils
import fermipy.roi_model as roi_model
import fermipy.gtanalysis
from fermipy import defaults
from fermipy import fits_utils
from fermipy.config import ConfigSchema
from fermipy.gtutils import FreeParameterState

import pyLikelihood as pyLike
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table, Column

import pyLikelihood as pyLike

def _fit_lc(gta, name, **kwargs):

    # lightcurve fitting routine-
    # 1.) start by freeing target and provided list of
    # sources, fix all else- if fit fails, fix all pars
    # except norm and try again
    # 2.) if that fails to converge then try fixing low TS
    #  (<4) sources and then refit
    # 3.) if that fails to converge then try fixing low-moderate TS (<9) sources and then refit
    # 4.) if that fails then fix sources out to 1dg away from center of ROI
    # 5.) if that fails set values to 0 in output and print warning message

    free_sources = kwargs.get('free_sources', [])
    free_background = kwargs.get('free_background', False)
    free_params = kwargs.get('free_params', None)

    if name in free_sources:
        free_sources.remove(name)

    free_state = FreeParameterState(gta)
    gta.free_sources(free=False)

    for niter in range(5):

        if free_background:
            free_state.restore()
        gta.free_source(name, pars=free_params)

        if niter == 0:
            gta.free_sources_by_name(free_sources)
        elif niter == 1:
            gta.logger.info('Fit Failed. Retrying with free '
                             'normalizations.')
            gta.free_sources_by_name(free_sources, False)
            gta.free_sources_by_name(free_sources, pars='norm')
        elif niter == 2:
            gta.logger.info('Fit Failed with User Supplied List of '
                             'Free/Fixed Sources.....Lets try '
                             'fixing TS<4 sources')
            gta.free_sources_by_name(free_sources, False)
            gta.free_sources_by_name(free_sources, pars='norm')
            gta.free_sources(minmax_ts=[0, 4], free=False, exclude=[name])
        elif niter == 3:
            gta.logger.info('Fit Failed with User Supplied List of '
                             'Free/Fixed Sources.....Lets try '
                             'fixing TS<9 sources')
            gta.free_sources_by_name(free_sources, False)
            gta.free_sources_by_name(free_sources, pars='norm')
            gta.free_sources(minmax_ts=[0, 9], free=False, exclude=[name])
        elif niter == 4:
            gta.logger.info('Fit still did not converge, lets try fixing the '
                             'sources up to 1dg out from ROI')
            gta.free_sources_by_name(free_sources, False)
            for s in free_sources:
                src = gta.roi.get_source_by_name(s)
                if src['offset'] < 1.0:
                    gta.free_source(s, pars='norm')
            gta.free_sources(minmax_ts=[0, 9], free=False, exclude=[name])
        else:
            gta.logger.error('Fit still didnt converge.....please examine this data '
                              'point, setting output to 0')
            break

        fit_results = gta.fit()
        if fit_results['fit_success'] is True:
            break

    return fit_results

def _process_lc_bin(itime, name, config, basedir, workdir, diff_sources, const_spectrum, **kwargs):
    i, time = itime

    config = copy.deepcopy(config)
    config['selection']['tmin'] = time[0]
    config['selection']['tmax'] = time[1]
    config['ltcube']['use_local_ltcube'] = kwargs['use_local_ltcube']
    config['model']['diffuse_dir'] = [workdir]

    if config['components'] is None:
        config['components'] = []

    # create output directories labeled in MET vals
    outdir = basedir + 'lightcurve_%.0f_%.0f' % (time[0], time[1])
    config['fileio']['outdir'] = os.path.join(workdir, outdir)
    utils.mkdir(config['fileio']['outdir'])

    yaml.dump(utils.tolist(config),
              open(os.path.join(config['fileio']['outdir'],
                                'config.yaml'), 'w'))

    xmlfile = os.path.join(config['fileio']['outdir'], 'base.xml')

    # Make a copy of the source maps. TODO: Implement a
    # correction to account for the difference in exposure for
    # each time bin.
    #     for c in self.components:
    #        shutil.copy(c._files['srcmap'],config['fileio']['outdir'])

    try:
        from fermipy.gtanalysis import GTAnalysis
        gta = GTAnalysis(config)
        gta.load_roi(workdir+'/_lc_%s.npy'%name)
        gta.logger.info('Fitting time range %i %i' % (time[0], time[1]))
        for j, c in enumerate(gta.components):
            if len(config['components']) <= j:
                config['components'] += [{}]

            data_cfg = {'evfile': c.files['ft1'],
                        'scfile': c.data_files['scfile'],
                        'ltcube': None}

            config['components'][j] = \
                utils.merge_dict(config['components'][j],
                                 {'data': data_cfg},
                                 add_new_keys=True)

        gta.setup()
    except:
        print('Analysis failed in time range %i %i'%
                            (time[0], time[1]))
        print(sys.exc_info()[0])
        raise
        return {}

    # Write the current model
    gta.write_xml(xmlfile)

    # Optimize the model
    gta.optimize()

    fit_results = _fit_lc(gta, name, **kwargs)
    gta.write_xml('fit_model_final.xml')
    srcmodel = copy.deepcopy(gta.get_src_model(name))

    # rerun fit using params from full time (constant) fit using same 
    # param vector as the successful fit to get loglike
    specname, spectrum = const_spectrum
    gta.set_source_spectrum(name, spectrum_type=specname, 
            spectrum_pars=spectrum)
    gta.free_source(name, free=False)
    const_fit_results = gta.fit()
    const_srcmodel = gta.get_src_model(name)

    o = {'flux_const': const_srcmodel['flux'],
            'loglike_flux_const': const_fit_results['loglike']}

    if fit_results['fit_success'] == 1:
        for k in defaults.source_flux_output.keys():
            if not k in srcmodel:
                continue
            """if (isinstance(srcmodel[k], np.ndarray) and
                    o[k][i].shape != output[k].shape):
                gta.logger.warning(
                    'Incompatible shape for column %s', k)
                continue"""
            o[k] = srcmodel[k]
    
    gta.logger.info('Finished time range %i %i' % (time[0], time[1]))

    return o

def calcTS_var(loglike, loglike_flux_const, flux_err, flux_const, systematic):
    # calculates variability according to Eq. 4 in 2FGL 
    # including correction using non-numbered Eq. following Eq. 4
    v_sqs = [loglike[i] - loglike_flux_const[i] for i in xrange(len(loglike))]
    factors = [flux_err[i]**2 / (flux_err[i]**2 + systematic**2 * flux_const**2) for i in xrange(len(flux_err))]
    return 2.*np.sum([a*b for a,b in zip(factors, v_sqs)])

class LightCurve(object):

    def lightcurve(self, name, **kwargs):
        """Generate a lightcurve for the named source. The function will
        complete the basic analysis steps for each bin and perform a
        likelihood fit for each bin. Extracted values (along with
        errors) are Integral Flux, spectral model, Spectral index, TS
        value, pred. # of photons. Note: successful calculation of 
        TS:subscript:`var` requires at least one free background 
        parameter and a previously optimized ROI model.

        Parameters
        ---------
        name: str
            source name

        {options}

        Returns
        ---------
        LightCurve : dict
           Dictionary containing output of the LC analysis

        """

        name = self.roi.get_source_by_name(name).name

        # Create schema for method configuration
        schema = ConfigSchema(self.defaults['lightcurve'],
                              optimizer=self.defaults['optimizer'])
        schema.add_option('prefix', '')
        config = utils.create_dict(self.config['lightcurve'],
                                   optimizer=self.config['optimizer'])
        config = schema.create_config(config, **kwargs)

        self.logger.info('Computing Lightcurve for %s' % name)

        o = self._make_lc(name, **config)
        filename = utils.format_filename(self.workdir, 'lightcurve',
                                         prefix=[config['prefix'],
                                                 name.lower().replace(' ', '_')])

        o['file'] = None
        if config['write_fits']:
            o['file'] = os.path.basename(filename) + '.fits'
            self._make_lc_fits(o, filename + '.fits', **config)

        if config['write_npy']:
            np.save(filename + '.npy', o)

        self.logger.info('Finished Lightcurve')

        return o

    def _make_lc_fits(self, lc, filename, **kwargs):

        # produce columns in fits file
        cols = OrderedDict()
        cols['tmin'] = Column(name='tmin', dtype='f8',
                              data=lc['tmin'], unit='s')
        cols['tmax'] = Column(name='tmax', dtype='f8',
                              data=lc['tmax'], unit='s')
        cols['tmin_mjd'] = Column(name='tmin_mjd', dtype='f8',
                                  data=lc['tmin_mjd'], unit='day')
        cols['tmax_mjd'] = Column(name='tmax_mjd', dtype='f8',
                                  data=lc['tmax_mjd'], unit='day')

        # add in columns for model parameters
        for k, v in lc.items():
            if k in cols:
                continue
            if isinstance(v, np.ndarray):
                cols[k] = Column(name=k, data=v, dtype=v.dtype)

        tab = Table(cols.values())
        hdu_lc = fits.table_to_hdu(tab)
        hdu_lc.name = 'LIGHTCURVE'
        hdus = [fits.PrimaryHDU(), hdu_lc]
        keywords = {'SRCNAME': lc['name'],
                    'CONFIG': json.dumps(lc['config'])}
        fits_utils.write_hdus(hdus, filename, keywords=keywords)

    def _create_lc_dict(self, name, times):

        # Output Dictionary
        o = {}
        o['name'] = name
        o['tmin'] = times[:-1]
        o['tmax'] = times[1:]
        o['tmin_mjd'] = utils.met_to_mjd(o['tmin'])
        o['tmax_mjd'] = utils.met_to_mjd(o['tmax'])

        for k, v in defaults.source_flux_output.items():

            if not k in self.roi[name]:
                continue

            v = self.roi[name][k]

            if isinstance(v, np.ndarray) and v.dtype.kind in ['S', 'U']:
                o[k] = np.zeros(o['tmin'].shape + v.shape, dtype=v.dtype)
            elif isinstance(v, np.ndarray):
                o[k] = np.nan * np.ones(o['tmin'].shape + v.shape)
            elif isinstance(v, np.float):
                o[k] = np.nan * np.ones(o['tmin'].shape)

        return o

    def _make_lc(self, name, **kwargs):

        # make array of time values in MET
        if kwargs['time_bins']:
            times = np.array(kwargs['time_bins'])
        elif kwargs['nbins']:
            times = np.linspace(self.tmin, self.tmax,
                                kwargs['nbins'] + 1)
        else:
            times = np.arange(self.tmin, self.tmax,
                              kwargs['binsz'])

        o = self._create_lc_dict(name, times)
        o['config'] = kwargs

        diff_sources = [s.name for s in self.roi.sources if s.diffuse]
        skydir = self.roi[name].skydir

        if kwargs.get('free_radius', None) is not None:
            kwargs['free_sources'] += [s.name for s in
                                       self.roi.get_sources(skydir=skydir,
                                                            distance=kwargs[
                                                                'free_radius'],
                                                            exclude=diff_sources)]

        self.write_roi('_lc_%s'%name, make_plots=False, save_model_map=False)
        free_param_vector = self.get_free_param_vector()
        # save params from full time fit
        spectrum = self.like[name].src.spectrum()
        specname = spectrum.genericName()
        const_spectrum = (specname, gtutils.get_function_pars_dict(spectrum))

        directory = kwargs.get('directory', None)
        basedir = directory + '/' if directory is not None else ''
        mt = kwargs.get('multithread', False)
        wrap = partial(_process_lc_bin, name=name, config=self.config,
                basedir=basedir, workdir=self.workdir, diff_sources=diff_sources, 
                const_spectrum=const_spectrum, **kwargs)
        itimes = enumerate(zip(times[:-1], times[1:]))
        if mt:
            p = Pool()
            mapo = p.map(wrap, itimes)
        else:
            mapo = map(wrap, itimes)

        for k in mapo[0].keys():
            if k in o:
                o.pop(k, None)

        files = os.listdir('./')
        for file in files:
            if file.startswith('_lc_%s'%name):
                os.remove(file)

        merged = utils.merge_list_of_dicts(mapo)
        o = utils.merge_dict(o, merged, add_new_keys=True)

        systematic = kwargs.get('systematic', 0.02)

        o['ts_var'] = calcTS_var(loglike=o['loglike'], 
                loglike_flux_const=o['loglike_flux_const'],
                flux_err=o['flux_err'],
                flux_const=o['flux_const'][0],
                systematic=systematic)

        return o

