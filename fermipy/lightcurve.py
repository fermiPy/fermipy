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
             'MJD':  np.zeros_like(times[:-1]),
             'flux': np.zeros_like(times[:-1]),
             'flux_err': np.zeros_like(times[:-1]),
             'flux100': np.zeros_like(times[:-1]),
             'flux100_err': np.zeros_like(times[:-1]),
             'flux1000': np.zeros_like(times[:-1]),
             'flux1000_err': np.zeros_like(times[:-1]),
             'flux10000': np.zeros_like(times[:-1]),
             'flux10000_err': np.zeros_like(times[:-1]),
             'eflux':np.zeros_like(times[:-1]),
             'eflux_err':np.zeros_like(times[:-1]),
             'eflux100':np.zeros_like(times[:-1]),
             'eflux100_err':np.zeros_like(times[:-1]),
             'eflux1000':np.zeros_like(times[:-1]),
             'eflux1000_err':np.zeros_like(times[:-1]),
             'eflux10000':np.zeros_like(times[:-1]),
             'eflux10000_err':np.zeros_like(times[:-1]),
             'dfde':np.zeros_like(times[:-1]),
             'dfde_err':np.zeros_like(times[:-1]),
             'dfde100':np.zeros_like(times[:-1]),
             'dfde100_err':np.zeros_like(times[:-1]),
             'dfde1000':np.zeros_like(times[:-1]),
             'dfde1000_err':np.zeros_like(times[:-1]),
             'dfde10000':np.zeros_like(times[:-1]),
             'dfde10000_eff':np.zeros_like(times[:-1]),
             'dfde_index':np.zeros_like(times[:-1]),
             'dfde_index_err':np.zeros_like(times[:-1]),
             'dfde100_index':np.zeros_like(times[:-1]),
             'dfde100_index_err':np.zeros_like(times[:-1]),
             'dfde1000_index':np.zeros_like(times[:-1]),    
             'dfde1000_index_err':np.zeros_like(times[:-1]),
             'dfde10000_index':np.zeros_like(times[:-1]),
             'dfde10000_index_err':np.zeros_like(times[:-1]),
             'flux_ul95':  np.zeros_like(times[:-1]),
             'flux100_ul95':np.zeros_like(times[:-1]),
             'flux1000_ul95':np.zeros_like(times[:-1]),
             'flux10000_ul95':np.zeros_like(times[:-1]),
             'eflux_ul95':    np.zeros_like(times[:-1]),
             'eflux100_ul95':np.zeros_like(times[:-1]),
             'eflux1000_ul95':np.zeros_like(times[:-1]),
             'eflux10000_ul95':np.zeros_like(times[:-1]),
             'pivot_energy': np.zeros_like(times[:-1]),
             'ts': np.zeros_like(times[:-1]),
             'loglike': np.zeros_like(times[:-1]),
             'npred': np.zeros_like(times[:-1])

             }





        print(times[:-1])
        for i, time in enumerate(zip(times[:-1], times[1:])):

            config = copy.deepcopy(self.config)
            config['selection']['tmin'] = time[0]
            config['selection']['tmax'] = time[1]            
            #create out directories labeled in MJD vals
            config['fileio']['outdir'] = os.path.join(self.workdir,'%i_%i'%(54682.65 + (time[0]-239557414.0)/(86400.) + (binning/2.)/86400.,54682.65 + (time[1]-239557414.0)/(86400.) + (binning/2.)/86400))
            
            utils.mkdir(config['fileio']['outdir'])
            
            xmlfile = os.path.join(config['fileio']['outdir'],'base.xml')
            
            # Make a copy of the source maps. TODO: Implement a
            # correction to account for the difference in exposure for
            # each time bin.
       #     for c in self.components:            
        #        shutil.copy(c._files['srcmap'],config['fileio']['outdir'])
            
            
            

            gta = fermipy.gtanalysis.GTAnalysis(config)
            gta.setup()

            # Write the current model 
            gta.write_xml(xmlfile)

            # Load the baseline model
            gta.load_xml(xmlfile)

            # Optimize the model
            gta.optimize()
            
            # Start by freeing normalization for everything w/in 3dg of center of ROI:
            self.logger.info('Fitting with normalization free')

            gta.free_sources(distance=3.0,pars='norm')
            
            fit_results = gta.fit()
            #gta.write_xml('fit_model_iter1.xml')

            if(fit_results['fit_success'] != 1):
                
                print('Fit Failed with all Source Parameters Fixed......Lets try getting rid of some low TS sources')
                
                gta.delete_sources(minmax_ts=[0,1])
                
                fit_results = gta.fit()
                
                if(fit_results['fit_success'] != 1):
                    gta.delete_sources(minmax_ts=[0,2])

                    fit_results = gta.fit()
                    
                    if(fit_results['fit_success'] != 1):
                        print('Fit still did not converge, lets try fixing the sources up to 1dg out from ROI')

                        gta.free_sources(free=0)
                        gta.free_sources(distance=1.0,pars='norm')

                    if(fit_results['fit_success'] != 1):
                        print('Fit still didnt converge.....please examine this data point')

            gta.write_xml('fit_model_pass1.xml')
#now fix the values, but free up params for source and diffuse comps
            
            self.logger.info('Fitting with all params free for source and diffuse, all else fixed')
            gta.free_sources(free=0)
            gta.free_sources(distance=0.1)
            gta.fit()

            if(fit_results['fit_success'] != 1):

                print('Fit Failed with all Source Parameters Fixed......Lets try getting rid of some low TS sources')

                gta.delete_sources(minmax_ts=[0,1])

                fit_results = gta.fit()

                if(fit_results['fit_success'] != 1):
                        gta.delete_sources(minmax_ts=[0,2])

                        fit_results = gta.fit()

                        if(fit_results['fit_success'] != 1):
                            print('Fit still did not converge, lets try fixing the sources up to 1dg out from ROI')

                            gta.free_sources(free=0)
                            gta.free_sources(distance=1.0,pars='norm')

                            if(fit_results['fit_success'] != 1):
                                print('Fit still didnt converge.....please examine this data point')

            gta.write_xml('fit_model_final.xml')
            output = gta.get_src_model(name)



            o['MJD'][i]=54682.65 + (times[i]-239557414.0)/(86400.) + (binning/2.)/86400.       
            o['flux'][i]                   = output['flux'][0]          
            o['flux_err'][i]               = output['flux'][1]
            o['flux100'][i]                = output['flux100'][0]
            o['flux100_err'][i]            = output['flux100'][1]
            o['flux1000'][i]               = output['flux1000'][0]
            o['flux1000_err'][i]           = output['flux1000'][1]
            o['flux10000'][i]              = output['flux10000'][0]
            o['flux10000_err'][i]          = output['flux10000'][1]
            o['eflux'][i]                  = output['eflux'][0]
            o['eflux_err'][i]              = output['eflux'][1]
            o['eflux100'][i]               = output['eflux100'][0]
            o['eflux100_err'][i]           = output['eflux100'][1]
            o['eflux1000'][i]              = output['eflux1000'][0]
            o['eflux1000_err'][i]          = output['eflux1000'][1]
            o['eflux10000'][i]             = output['eflux10000'][0]
            o['eflux10000_err'][i]         = output['eflux10000'][1]
            o['dfde'][i]                   = output['dfde'][0]
            o['dfde_err'][i]               = output['dfde'][1]
            o['dfde100'][i]                = output['dfde100'][0]
            o['dfde100_err'][i]            = output['dfde100'][1]
            o['dfde1000'][i]               = output['dfde1000'][0]
            o['dfde1000_err'][i]           = output['dfde1000'][1]
            o['dfde10000][i']              = output['dfde10000'][0]
            o['dfde10000_eff'][i]          = output['dfde10000'][1]
            o['dfde_index'][i]             = output['dfde_index'][0]
            o['dfde_index_err'][i]         = output['dfde_index'][1]
            o['dfde100_index'][i]          = output['dfde100_index'][0]
            o['dfde100_index_err'][i]      = output['dfde100_index'][1]
            o['dfde1000_index'][i]         = output['dfde1000_index'][0]
            o['dfde1000_index_err'][i]     = output['dfde1000_index'][1]
            o['dfde10000_index'][i]        = output['dfde10000_index'][0]
            o['dfde10000_index_err'][i]    = output['dfde10000_index'][1]
            o['flux_ul95'][i]              = output['flux_ul95']
            o['flux100_ul95'][i]           = output['flux100_ul95']
            o['flux1000_ul95'][i]          = output['flux1000_ul95']
            o['flux10000_ul95'][i]         = output['flux10000_ul95']
            o['eflux_ul95'][i]             = output['eflux_ul95']
            o['eflux100_ul95'][i]          = output['eflux100_ul95']
            o['eflux1000_ul95'][i]         = output['eflux1000_ul95']
            o['eflux10000_ul95'][i]        = output['eflux10000_ul95']
            o['pivot_energy'][i]           = output['pivot_energy']
            o['ts'][i]                     = output['ts']
            o['loglike'][i]                = output['loglike']
            o['npred'][i]                  = output['npred']
        

          

        src = self.roi.get_source_by_name(name)
        src.update_data({'LightCurve': copy.deepcopy(o)})

        return o
