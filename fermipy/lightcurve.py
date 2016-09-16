# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import os
import copy
import shutil
import subprocess

import numpy as np

import fermipy.config as config
import fermipy.utils as utils
import fermipy.gtutils as gtutils
import fermipy.roi_model as roi_model
import fermipy.gtanalysis

import pyLikelihood as pyLike
from astropy.io import fits

import GtApp
import BinnedAnalysis as ba
import UnbinnedAnalysis as uba
import pyLikelihood as pyLike
from UpperLimits import UpperLimits


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

        prefix : str
            Optional string that will be prepended to all output files

        calc_ul: bool
             specifies whether or not to calculate upper limits for
             flux points below a threshold TS

        thresh_TS: float
            threshold values of TS below which triggers UL calculation
            (if calc_ul is true)

        binning: float
            user provided time binning in seconds

        unbinned_analysis : bool
           if true, perform an unbinned analysis


        Returns
        ---------
        LightCurve : dict
           Dictionary containing output of the LC analysis

        """

        name = self.roi.get_source_by_name(name).name

        # extract options from kwargs
        config = copy.deepcopy(self.config['lightcurve'])
        config.setdefault('prefix', '')
        config.setdefault('write_fits', False)
        config.setdefault('write_npy', True)
        fermipy.config.validate_config(kwargs, config)
        config = utils.merge_dict(config, kwargs)

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

    def _make_lc_fits(self, LightCurve, filename, **kwargs):

        # produce columns in fits file
        cols = [Column(name='MJD', dtype='f8', data=LightCurve['plottimes'], unit='MJD Days'),
                Column(name='IntFlux', dtype='f8', data=LightCurve['IntFlux'], unit='ph cm^-2 s^-1'),
                Column(name='IntFluxErr', dtype='f8', data=LightCurve['IntFluxErr'], unit='ph cm^-2 s^-1'),
                Column(name='Model', dtype=np.str, data=LightCurve['model'], unit='')]

        # add in columns for model parameters
        for fields in LightCurve:
            if (str(fields[:3]) == 'par'):
                cols.append(Column(name=fields, dtype='f8', data=LightCurve[str(fields)], unit=''))
                
        cols.append(Column(name='TS Value', dtype='f8', data=LightCurve['TS'], unit=''))
        cols.append(Column(name='NPred', dtype='f8', data=LightCurve['npred'],unit='# of predicted photons'))
        cols.append(Column(name='retCode', dtype='int32', data=LightCurve['retCode'], unit=''))

        tab = Table(cols)

        tab.write(filename, format='fits', overwrite=True)

        hdulist = fits.open(filename)
        hdulist[1].name = 'LightCurve'
        hdulist = fits.HDUList([hdulist[0], hdulist[1]])

        for h in hdulist:
            h.header['SRCNAME'] = LightCurve['name']
            h.header['CREATOR'] = 'fermipy ' + fermipy.__version__
            
        hdulist.writeto(filename, clobber=True)
    
    def _make_lc(self, name, **config):

        calc_ul = config['calc_ul']
        thresh_TS = config['thresh_TS']
        binning = config['binning']
        #unbinned_analysis = config['unbinned_analysis']

        # make array of time values in MET
        times = np.arange(self.config['selection']['tmin'],
                          self.config['selection']['tmax'], binning)

        # Output Dictionary

        o = {'name': name,
             # time array in MJD,
             'plottimes':  np.zeros_like(times[:-1]),
             'model': str,
             'IntFlux': np.zeros_like(times[:-1]),
             'IntFluxErr': np.zeros_like(times[:-1]),
             'pars': dict(),
             'TS':  np.zeros_like(times[:-1]),
             'retCode':  np.zeros_like(times[:-1]),
             'npred':  np.zeros_like(times[:-1])
             }

        for i, time in enumerate(zip(times[:-1], times[1:])):

            config = copy.deepcopy(self.config)
            config['selection']['tmin'] = time[0]
            config['selection']['tmax'] = time[1]            
            config['fileio']['outdir'] = os.path.join(self.workdir,'%i_%i'%(time[0],time[1]))
            
            utils.mkdir(config['fileio']['outdir'])
            
            xmlfile = os.path.join(config['fileio']['outdir'],'base.xml')

            # Make a copy of the source maps. TODO: Implement a
            # correction to account for the difference in exposure for
            # each time bin.
            for c in self.components:            
                shutil.copy(c._files['srcmap'],config['fileio']['outdir'])
            
            # Write the current model
            self.write_xml(xmlfile)

            gta = fermipy.gtanalysis.GTAnalysis(config)
            gta.setup()

            # Load the baseline model
            gta.load_xml('base.xml')

            # Optimize the model
            gta.optimize()
            
            # Try Fitting with everything in the source model fixed:

            # Delete low TS sources?

            # Okay now free the normalization parameter and try the fit

            #srcnormpar.setFree(1)
            #binnedA.syncSrcParams(self.config['selection']['target'])

            self.logger.info('Fitting with normalization free')

            # Okay now free everything and try the fit

            self.logger.info('Fitting with everything free')


            # Retrieve the flux and parameter errors and values
            
            #print(binnedA.flux(str(self.config['selection']['target']),
            #                   emin=self.config['selection']['emin'],
            #                   emax=self.config['selection']['emax']))
                

            #o['plottimes'][i] = 54682.65 + (times[i]-239557414.0)/(86400.) + (binning/2.)/86400.
                


        src = self.roi.get_source_by_name(name)
        src.update_data({'LightCurve': copy.deepcopy(o)})

        return o
