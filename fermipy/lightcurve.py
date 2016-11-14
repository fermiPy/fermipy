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

    def lightcurve(self, name, freesources, **kwargs):
        """Generate a lightcurve for the named source. The function will
        complete the basic analysis steps for each bin and perform a
        likelihood fit for each bin. Extracted values (along with
        errors) are Integral Flux, spectral model, Spectral index, TS
        value, pred. # of photons.

        Parameters
        ---------
        name: str
            source name

        fitsources: dict
            dict of sources to be left free in the fitting routine


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

        o = self._make_lc(name, freesources, **config)
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
    
    def _make_lc(self, name, freesources, **config):

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
                              config['binning'])

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

            #fitting routine-
            #  1.)   start by freeing target and provided list of sources, fix all else- if fit fails, fix all pars except norm and try again
            #    2.)    if that fails to converge then try fixing low TS (<4) sources and then refit
            #      3.)    if that fails to converge then try fixing low-moderate TS (<9) sources and then refit   
            #        4.)    if that fails then fix sources out to 1dg away from center of ROI
            #         5.)    if that fails set values to 0 in output and print warning message
            #  
            #-----------11111111----------#

            gta.free_sources(free=False, exclude_diffuse=True )

            for names in freesources:
                gta.free_source(names)
            gta.free_source(name)

            fit_results = gta.fit()
    
         
            if(fit_results['fit_success'] != 1):

                 for names in freesources:
                     gta.free_source(names, free=False)
                     gta.free_source(names, pars='norm')
                 gta.free_source(name)
                 fit_results = gta.fit()

             #-----------22222222---------#

                 if(fit_results['fit_success'] != 1):
                     print('Fit Failed with User Supplied List of Free/Fixed Sources.....Lets try '
                               'fixing TS<4 sources')               
                     
                     gta.free_sources(free=False, exclude_diffuse=True )

                     for names in freesources:
                         gta.free_source(names)

                     gta.free_sources(minmax_ts=[0,4],free=False) 
                     gta.free_source(name)

                     fit_results = gta.fit()
                
               #----------333333333---------#    
                                                                                     
                     if(fit_results['fit_success'] != 1):
                         print('Fit Failed with User Supplied List of Free/Fixed Sources.....Lets try '
                                  'fixing TS<9 sources')
                   
                         gta.free_sources(free=False, exclude_diffuse=True )

                         for names in freesources:
                             gta.free_source(names)

                         gta.free_sources(minmax_ts=[0,9],free=False)
                         gta.free_source(name)
                         fit_results = gta.fit()
               
                   #-----------4444444444---------#  

                         if(fit_results['fit_success'] != 1):
                             print('Fit still did not converge, lets try fixing the sources up to 1dg out from ROI')
        
                     
                             gta.free_sources(free=False, exclude_diffuse=True)
                                  
                             for s in freesources:
                                 src = self.roi.get_source_by_name(s)
                                 if src['offset'] < 1.0:
                                     gta.free_source(s)
                             gta.free_sources(minmax_ts=[0,9], free=False)
                                  
                             gta.free_source(name)
                     
                             fit_results = gta.fit()
                     

                      #-----------55555555555---------#      
                             if(fit_results['fit_success'] != 1):
                                 print('Fit still didnt converge.....please examine this data point, setting output to 0')
                         

            
            gta.write_xml('fit_model_final.xml')
            output = gta.get_src_model(name)

            for k in defaults.source_flux_output.keys():
                if not k in output:
                    continue                
                if (fit_results['fit_success'] == 1):
                    o[k] = output[k]
                else:
                    o[k]=0.0

        src = self.roi.get_source_by_name(name)
        src.update_data({'lightcurve': copy.deepcopy(o)})

        return o
