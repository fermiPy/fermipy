# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import os
import copy
import shutil
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

        prefix : str
            Optional string that will be prepended to all output files

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

        # Create schema for method configuration
        schema = ConfigSchema(self.defaults['lightcurve'],
                              optimizer=self.defaults['optimizer'])
        schema.add_option('prefix', '')
        schema.add_option('write_fits', False)
        schema.add_option('write_npy', True)
        schema.add_option('binning', 86400.)
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
        cols['tmin'] = Column(name='tmin', dtype='f8', data=lc['tmin'], unit='s')
        cols['tmax'] = Column(name='tmax', dtype='f8', data=lc['tmax'], unit='s')
        cols['tmin_mjd'] = Column(name='tmin_mjd', dtype='f8', data=lc['tmin_mjd'], unit='day')
        cols['tmax_mjd'] = Column(name='tmax_mjd', dtype='f8', data=lc['tmax_mjd'], unit='day')
                
        # add in columns for model parameters
        for k,v in lc.items():

            if k in cols:
                continue
            
            if isinstance(v,np.ndarray):
                cols[k] = Column(name=k, data=v, dtype='f8')
        
        #for fields in lc:
        #    if (str(fields[:3]) == 'par'):
        #        cols.append(Column(name=fields, dtype='f8', data=lc[str(fields)], unit=''))
                
        tab = Table(cols.values())
        tab.write(filename, format='fits', overwrite=True)

        hdulist = fits.open(filename)
        hdulist[1].name = 'LIGHTCURVE'
        hdulist = fits.HDUList([hdulist[0], hdulist[1]])
        fits_utils.write_fits(hdulist,filename,{'SRCNAME' : lc['name']})
    
    def _make_lc(self, name, **config):

        # make array of time values in MET
        if config['time_bins']:
            times = config['time_bins']
        elif config['nbins']:
            times = np.linspace(self.config['selection']['tmin'],
                                self.config['selection']['tmax'],
                                config['nbins']+1)
        else:
            times = np.arange(self.config['selection']['tmin'],
                              self.config['selection']['tmax'],
                              config['binsz'])

        # Output Dictionary
        o = {}
        o['name'] = name
        o['tmin'] = times[:-1]
        o['tmax'] = times[1:]
        o['tmin_mjd'] = utils.met_to_mjd(o['tmin'])
        o['tmax_mjd'] = utils.met_to_mjd(o['tmax'])
        o['config'] = config
        
        for k, v in defaults.source_flux_output.items():

            if not k in self.roi[name]:
                continue
            
            v = self.roi[name][k]
            
            if isinstance(v,np.ndarray):
                o[k] = np.zeros(times[:-1].shape + v.shape)
            elif isinstance(v,np.float):
                o[k] = np.zeros(times[:-1].shape)
                
        for i, time in enumerate(zip(times[:-1], times[1:])):

            self.logger.info('Fitting time range %i %i',time[0],time[1])
            
            config = copy.deepcopy(self.config)
            config['selection']['tmin'] = time[0]
            config['selection']['tmax'] = time[1]            
            #create out directories labeled in MJD vals
            outdir = '%.3f_%.3f'%(utils.met_to_mjd(time[0]),
                                  utils.met_to_mjd(time[1]))
            config['fileio']['outdir'] = os.path.join(self.workdir,outdir)
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

            # Optimize the model (skip diffuse?)
            gta.optimize(skip=['galdiff','isodiff'])
            
            # Start by fixing all paramters for sources with Variability Index <50
            
            gta.free_sources(free=False, minmax_var=[0,50], exclude_diffuse=True )
                            
            gta.free_source(name)
            fit_results = gta.fit()
         



#If fit fails, try altering model
         
            if(fit_results['fit_success'] != 1):
              
                print('Fit Failed with all Source Parameters Fixed......Lets try '
                    'getting rid of some low TS sources')                
            
                for s in gta.roi.get_sources(minmax_ts=[0,10]):                                    
                    if s != name:                                                         
                        gta.delete_source(s.name,build_fixed_wts=False)                                          
                for c in gta.components:                                              
                    c.like.logLike.buildFixedModelWts()                               
                                                                                      
                gta._update_roi()                                                                    
         
                fit_results = gta.fit()
              
                if(fit_results['fit_success'] != 1):
                
                    for s in gta.roi.get_sources(minmax_ts=[0,10]):
                        if s != name:
                            gta.delete_source(s.name,build_fixed_wts=False)
                    for c in gta.components:
                        c.like.logLike.buildFixedModelWts()

                        gta._update_roi()

                    fit_results = gta.fit()

                    if(fit_results['fit_success'] != 1):
                      print('Fit still did not converge, lets try fixing the sources up to 1dg out from ROI')
         
                      gta.free_sources(free=0)
                      gta.free_sources(distance=1.0,pars='norm')
                      fit_results = gta.fit()
                      if(fit_results['fit_success'] != 1):
                          print('Fit still didnt converge.....please examine this data point')
         



#            gta.write_xml('fit_model_pass1.xml')
            


#now fix the values, but free up params for source and diffuse comps
#            
#            self.logger.info('Fitting with all params free for source and diffuse, all else fixed')
#            gta.free_sources(free=False)
#            gta.free_source(name)
#            #gta.free_sources(distance=0.1)
#            gta.fit()
#
#            if(fit_results['fit_success'] != 1):
#
#                print('Fit Failed with all Source Parameters Fixed......Lets try '
#                      'getting rid of some low TS sources')
#
#                gta.delete_sources(minmax_ts=[0,1])
#
#                fit_results = gta.fit()
#
#                if(fit_results['fit_success'] != 1):
#                    gta.delete_sources(minmax_ts=[0,2])
#
#                    fit_results = gta.fit()
#
#                    if(fit_results['fit_success'] != 1):
#                        print('Fit still did not converge, lets try fixing the sources up to 1dg out from ROI')
#
#                        gta.free_sources(free=0)
#                        gta.free_sources(distance=1.0,pars='norm')
#
#                        if(fit_results['fit_success'] != 1):
#                            print('Fit still didnt converge.....please examine this data point')

            gta.write_xml('fit_model_final.xml')
            output = gta.get_src_model(name)

            for k in defaults.source_flux_output.keys():
                if not k in output:
                    continue                
                o[k] = output[k]

        src = self.roi.get_source_by_name(name)
        src.update_data({'lightcurve': copy.deepcopy(o)})

        return o
