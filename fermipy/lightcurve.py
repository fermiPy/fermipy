# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import os
import copy
import shutil
import logging
import yaml
import json
from collections import OrderedDict

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


class LightCurve(object):

    def lightcurve(self, name, **kwargs):
        """Generate a lightcurve for the named source. The function will
        complete the basic analysis steps for each bin and perform a
        likelihood fit for each bin. Extracted values (along with
        errors) are Integral Flux, spectral model, Spectral index, TS
        value, pred. # of photons.

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

        for i, time in enumerate(zip(times[:-1], times[1:])):

            self.logger.info('Fitting time range %i %i', time[0], time[1])

            config = copy.deepcopy(self.config)
            config['selection']['tmin'] = time[0]
            config['selection']['tmax'] = time[1]
            config['ltcube']['use_local_ltcube'] = kwargs['use_local_ltcube']
            config['model']['diffuse_dir'] = [self.workdir]

            if config['components'] is None:
                config['components'] = []

            for j, c in enumerate(self.components):
                if len(config['components']) <= j:
                    config['components'] += [{}]

                data_cfg = {'evfile': c.files['ft1'],
                            'scfile': c.data_files['scfile'],
                            'ltcube': None}

                config['components'][j] = \
                    utils.merge_dict(config['components'][j],
                                     {'data': data_cfg},
                                     add_new_keys=True)

            # create output directories labeled in MET vals
            outdir = 'lightcurve_%.0f_%.0f' % (time[0], time[1])
            config['fileio']['outdir'] = os.path.join(self.workdir, outdir)
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
                gta = self.clone(config, loglevel=logging.DEBUG)
                gta.setup()
            except:
                self.logger.warning('Analysis failed in time range %i %i',
                                    time[0], time[1])
                continue

            # Write the current model
            gta.write_xml(xmlfile)

            # Optimize the model (skip diffuse?)
            gta.optimize(skip=diff_sources)

            fit_results = self._fit_lc(gta, name, **kwargs)
            gta.write_xml('fit_model_final.xml')
            output = gta.get_src_model(name)

            if fit_results['fit_success'] == 1:            
                for k in defaults.source_flux_output.keys():
                    if not k in output:
                        continue
                    if (isinstance(output[k],np.ndarray) and
                         o[k][i].shape != output[k].shape):
                        self.logger.warning('Incompatible shape for column %s',k)
                        continue                        
                    o[k][i] = output[k]
                        
            self.logger.info('Finished time range %i %i', time[0], time[1])

        src = self.roi.get_source_by_name(name)
        return o

    def _fit_lc(self, gta, name, **kwargs):

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
                self.logger.info('Fit Failed. Retrying with free '
                                 'normalizations.')
                gta.free_sources_by_name(free_sources, False)
                gta.free_sources_by_name(free_sources, pars='norm')
            elif niter == 2:
                self.logger.info('Fit Failed with User Supplied List of '
                                 'Free/Fixed Sources.....Lets try '
                                 'fixing TS<4 sources')
                gta.free_sources_by_name(free_sources, False)
                gta.free_sources_by_name(free_sources, pars='norm')
                gta.free_sources(minmax_ts=[0, 4], free=False, exclude=[name])
            elif niter == 3:
                self.logger.info('Fit Failed with User Supplied List of '
                                 'Free/Fixed Sources.....Lets try '
                                 'fixing TS<9 sources')
                gta.free_sources_by_name(free_sources, False)
                gta.free_sources_by_name(free_sources, pars='norm')
                gta.free_sources(minmax_ts=[0, 9], free=False, exclude=[name])
            elif niter == 4:
                self.logger.info('Fit still did not converge, lets try fixing the '
                                 'sources up to 1dg out from ROI')
                gta.free_sources_by_name(free_sources, False)
                for s in free_sources:
                    src = self.roi.get_source_by_name(s)
                    if src['offset'] < 1.0:
                        gta.free_source(s, pars='norm')
                gta.free_sources(minmax_ts=[0, 9], free=False, exclude=[name])
            else:
                self.logger.error('Fit still didnt converge.....please examine this data '
                                  'point, setting output to 0')
                break

            fit_results = gta.fit()
            if fit_results['fit_success'] is True:
                break

        return fit_results
