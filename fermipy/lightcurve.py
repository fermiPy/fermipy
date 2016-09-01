# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import copy
import numpy as np
from os import path
import subprocess

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
        unbinned_analysis = config['unbinned_analysis']

        # make array of time values in MET
        self.times = np.arange(self.config['selection']['tmin'], self.config[
                               'selection']['tmax'], binning)

        # Output Dictionary

        o = {'name': name,
             # time array in MJD,
             'plottimes':  np.zeros_like(self.times[:-1]),
             'model': str,
             'IntFlux': np.zeros_like(self.times[:-1]),
             'IntFluxErr': np.zeros_like(self.times[:-1]),
             'pars': dict(),
             'TS':  np.zeros_like(self.times[:-1]),
             'retCode':  np.zeros_like(self.times[:-1]),
             'npred':  np.zeros_like(self.times[:-1])
             }

        for i, time in enumerate(zip(self.times[:-1], self.times[1:])):

            ######
            # Branch of analysis for binned
            # prepare counts cube for binned analysis

            if(unbinned_analysis == False):

                # Produce necessary files for each bin
                self.logger.info("**Working on range (%f,%f)", time[0], time[1])
# run gtselect
                kw_gtselect = dict(infile=self.config['data']['evfile'],
                                   outfile='lc_filtered_bin' +
                                   str(i) + '.fits',
                                   ra='INDEF',  # self.roi.skydir.ra.deg,
                                   dec='INDEF',  # self.roi.skydir.dec.deg,
                                   # self.config['selection']['radius'],
                                   rad='INDEF',
                                   convtype=self.config[
                                       'selection']['convtype'],
                                   evtype=self.config['selection']['evtype'],
                                   evclass=self.config['selection']['evclass'],
                                   tmin=time[0],
                                   tmax=time[1],
                                   emin=self.config['selection']['emin'],
                                   emax=self.config['selection']['emax'],
                                   zmax=self.config['selection']['zmax'],
                                   chatter=self.config['logging']['chatter'])

                if(not path.exists('lc_filtered_bin' + str(i) + '.fits')):
                    fermipy.gtanalysis.run_gtapp(
                        'gtselect', self.logger, kw_gtselect)
                else:
                    print("File exists.  Won't execute.")
# run gtmktime
                kw_gtmktime = dict(evfile='lc_filtered_bin' + str(i) + '.fits',
                                   outfile='lc_filtered_gti_bin' +
                                   str(i) + '.fits',
                                   scfile=self.config['data']['scfile'],
                                   roicut=self.config['selection']['roicut'],
                                   filter=self.config['selection']['filter'])
                if(not path.exists('lc_filtered_gti_bin' + str(i) + '.fits')):
                    fermipy.gtanalysis.run_gtapp(
                        'gtmktime', self.logger, kw_gtmktime)
                else:
                    print("File exists.  Won't execute.")


# run gtltcube

                kw_gtltcube = dict(evfile='lc_filtered_gti_bin' + str(i) + '.fits',
                                   scfile=self.config['data']['scfile'],
                                   outfile='lc_ltcube_bin' + str(i) + '.fits',
                                   dcostheta=0.025,
                                   binsz=1.0)

                if (not path.exists('lc_ltcube_bin' + str(i) + '.fits')):
                    print("Calculating ltcube for bin " + str(i))

                    fermipy.gtanalysis.run_gtapp(
                        'gtltcube', self.logger, kw_gtltcube)

                else:
                    print("LTcube for this bins already exists......")


# run gtbin

                kw_gtccube = dict(algorithm='CCUBE',
                                  evfile='lc_filtered_gti_bin' +
                                  str(i) + '.fits',
                                  outfile='lc_ccube_binned' + str(i) + '.fits',
                                  scfile=self.config['data']['scfile'],
                                  nxpix=np.int(
                                      (np.float(self.config['binning']['roiwidth']) * np.sqrt(2) / 0.2)),
                                  nypix=np.int(
                                      (np.float(self.config['binning']['roiwidth']) * np.sqrt(2) / 0.2)),
                                  binsz=0.2,
                                  coordsys='CEL',
                                  xref=self.roi.skydir.ra.deg,
                                  yref=self.roi.skydir.dec.deg,
                                  axisrot=0.0,
                                  proj='AIT',
                                  ebinalg='LOG',
                                  emin=self.config['selection']['emin'],
                                  emax=self.config['selection']['emax'],
                                  enumbins=37
                                  )
                if (not path.exists('lc_ccube_binned' + str(i) + '.fits')):
                    print("Calculating counts cube for bin" + str(i))
                    fermipy.gtanalysis.run_gtapp(
                        'gtbin', self.logger, kw_gtccube)
                else:
                    print("CCUBE exists already- lucky you.")


# run gtexpcube2
                kw_expcube2 = dict(infile='lc_ltcube_bin' + str(i) + '.fits',
                                   cmap='none',
                                   outfile='lc_expcube_bin' + str(i) + '.fits',
                                   irfs='P8R2_SOURCE_V6',
                                   nxpix=np.int(
                                       ((np.float(self.config['binning']['roiwidth']) + 10.0) * np.sqrt(2) / 0.2)),
                                   nypix=np.int(
                                       ((np.float(self.config['binning']['roiwidth']) + 10.0) * np.sqrt(2) / 0.2)),
                                   binsz=0.2,
                                   coordsys='CEL',
                                   xref=self.roi.skydir.ra.deg,
                                   yref=self.roi.skydir.dec.deg,
                                   proj='AIT',
                                   ebinalg='LOG',
                                   emin=self.config['selection']['emin'],
                                   emax=self.config['selection']['emax'],
                                   evtype=self.config['selection']['evtype'],
                                   chatter='1',
                                   enumbins=37)

                if (not path.exists('lc_expcube_bin' + str(i) + '.fits')):
                    print("Calculating the Expcube for bin" + str(i))

                    fermipy.gtanalysis.run_gtapp(
                        'gtexpcube2', self.logger, kw_expcube2)
                else:
                    print("ExpCube exists already- lucky you.")

# run gtsrcmaps
                kw_srcmap = dict(scfile=self.config['data']['scfile'],
                                 cmap='lc_ccube_binned' + str(i) + '.fits',
                                 expcube='lc_ltcube_bin' + str(i) + '.fits',
                                 srcmdl='fit_model_00.xml',
                                 bexpmap='lc_expcube_bin' + str(i) + '.fits',
                                 outfile='lc_srcmap_bin' + str(i) + '.fits',
                                 irfs='CALDB',
                                 emapbnds='no')

                if (not path.exists('lc_srcmap_bin' + str(i) + '.fits')):
                    print("Calculating the Source Map for bin" + str(i))

                    fermipy.gtanalysis.run_gtapp(
                        'gtsrcmaps', self.logger, kw_srcmap)
                else:
                    print("Source Map exists already- lucky you.")


# create BinnedAnalysis object, fit it like a boss, then push out the vals

                kwBO = dict(irfs='CALDB',
                            srcMaps='lc_srcmap_bin' + str(i) + '.fits',
                            expCube='lc_ltcube_bin' + str(i) + '.fits',
                            binnedExpMap='lc_expcube_bin' + str(i) + '.fits'
                            )

                bo = ba.BinnedObs(**utils.unicode_to_str(kwBO))

                kwBA = dict(srcModel='fit_model_00.xml',
                            optimizer='MINUIT')

                binnedA = ba.BinnedAnalysis(bo, **utils.unicode_to_str(kwBA))
                likeObj = pyLike.Minuit(binnedA.logLike)

                # Try Fitting with everything in the source model fixed:
#
#
                srcfreepar = binnedA.freePars(
                    str(self.config['selection']['target']))
                srcnormpar = binnedA.normPar(
                    self.config['selection']['target'])

                if len(srcfreepar) > 0:
                    binnedA.setFreeFlag(
                        str(self.config['selection']['target']), srcfreepar, 0)
                    binnedA.syncSrcParams(
                        str(self.config['selection']['target']))

                self.logger.info(
                    'Computing LL for all source parameters fixed')

                try:
                    binnedA.fit(verbosity=0, covar=True, optObject=likeObj)
                except:
                    pass

                if (likeObj.getRetCode() != 0):
                    self.logger.info(
                        'Looks like the fit didnt converge, lets try deleting some low (<1) TS sources')
                    deletesrc = []
                    for s in like.sourceNames():
                        freepars = binnedA.freePars(s)
                        if(s != str(self.config['selection']['target']) and binnedA[s].src.getType() == 'Point' and len(freepars) > 0):
                            ts = binnedA.Ts(s)
                            if ts < 1.0:
                                deletesrc.append(s)
                                self.logger.info('-- {} (TS={})'.format(s, ts))
                    if deletesrc:
                        for s in deletesrc:
                            binnedA.deleteSource(s)

                    self.logger.info('Trying the fit again.....')

                    try:
                        binnedA.fit(verbosity=0, covar=True, optObject=likeObj)
                    except:
                        pass

                if (likeObj.getRetCode() != 0):
                    self.logger.info(
                        'Looks like the fit still didnt converge, lets try deleting some more low (<2) TS sources')
                    deletesrc = []
                    for s in like.sourceNames():
                        freepars = binnedA.freePars(s)
                        if(s != str(self.config['selection']['target']) and binnedA[s].src.getType() == 'Point' and len(freepars) > 0):
                            ts = binnedA.Ts(s)
                            if ts < 2.0:
                                deletesrc.append(s)
                                self.logger.info('-- {} (TS={})'.format(s, ts))
                    if deletesrc:
                        for s in deletesrc:
                            binnedA.deleteSource(s)

                    self.logger.info(
                        'Last time trying the fit....if it still doesnt converge you might have to tinker with your model more for this bin....')

                    try:
                        binnedA.fit(verbosity=0, covar=True, optObject=likeObj)
                    except:
                        pass

                if (likeObj.getRetCode() != 0):
                    self.logger.info(
                        'Warning- the fit still didnt converge....')

 # print binnedA.flux(str(self.config['selection']['target']),
 # emin=self.config['selection']['emin'],
 # emax=self.config['selection']['emax'])

                # Okay now free the normalization parameter and try the fit

                srcnormpar.setFree(1)
                binnedA.syncSrcParams(self.config['selection']['target'])

                self.logger.info('Fitting with normalization free')

                try:
                    binnedA.fit(verbosity=0, covar=True, optObject=likeObj)
                except:
                    pass

# print binnedA.flux(str(self.config['selection']['target']),
# emin=self.config['selection']['emin'],
# emax=self.config['selection']['emax'])

                # Okay now free everything and try the fit

                self.logger.info('Fitting with everything free')

                binnedA.setFreeFlag(
                    str(self.config['selection']['target']), srcfreepar, 1)
                binnedA.syncSrcParams(str(self.config['selection']['target']))

                try:
                    binnedA.fit(verbosity=0,covar=True,optObject=likeObj)
                except:
                    pass
                print(binnedA.flux(str(self.config['selection']['target']),
                                   emin=self.config['selection']['emin'],
                                   emax=self.config['selection']['emax']))
                
                binnedA.flux(str(self.config['selection']['target']),  emin=self.config['selection']['emin'], emax=self.config['selection']['emax'])
                pars = dict()
                for pn in binnedA[str(self.config['selection']['target'])].funcs['Spectrum'].paramNames:
                    p = binnedA[str(self.config['selection']['target'])].funcs['Spectrum'].getParam(pn)
                    pars[p.getName()] = dict(name      = p.getName(),
                                         value     = p.getTrueValue(),
                                         error     = p.error()*p.getScale(),
                                         free      = p.isFree())
                    o['pars'][i] = pars

                o['model']=self.roi[self.config['selection']['target']]['SpectrumType']    
                o['IntFlux'][i] = binnedA.flux(str(self.config['selection']['target']),
                                               emin=self.config['selection']['emin'],
                                               emax=self.config['selection']['emax'])    
                o['IntFluxErr'][i] = binnedA.fluxError(str(self.config['selection']['target']),
                                                       emin=self.config['selection']['emin'],
                                                       emax=self.config['selection']['emax'])
                o['TS'][i] = binnedA.Ts(str(self.config['selection']['target']))
                o['retCode'][i] = likeObj.getRetCode()
                o['npred'][i] = binnedA.NpredValue(str(self.config['selection']['target']))
                o['plottimes'][i] = 54682.65 + (self.times[i]-239557414.0)/(86400.) + (binning/2.)/86400.
                
                if(calc_ul == True and (o['TS'][i] < thresh_TS) ):
                    self.logger.info('TS is less than critical values, calculating upper limit')
                    ul=UpperLimits(binnedA)
                    ulvals=ul[str(self.config['selection']['target'])].compute()
                    o['IntFlux'][i] = ulvals[0]
                    o['IntFluxErr'][i] = 0.0

######
#Branch of anlysis for Unbinned***** CURRENTLY NOT WORKING

            if(unbinned_analysis == True):
#####Branch to deal with UnBinned Analysis

                self.logger.info("**Working on range (%f,%f)",time[0],time[1])
#run gtselect                                                                                                                                                                                             
                kw_gtselect = dict(infile=self.config['data']['evfile'],
                           outfile='lc_filtered_bin'+str(i)+'.fits',
                           ra=self.roi.skydir.ra.deg,                                                                                                                                            
                           dec=self.roi.skydir.dec.deg,                                                                                                                                          
                           rad=self.config['binning']['roiwidth'],                                                                                                                               
                           convtype=self.config['selection']['convtype'],
                           evtype=self.config['selection']['evtype'],
                           evclass=self.config['selection']['evclass'],
                           tmin=time[0],
                           tmax=time[1],
                           emin=self.config['selection']['emin'],
                           emax=self.config['selection']['emax'],
                           zmax=self.config['selection']['zmax'],
                           chatter=self.config['logging']['chatter'])

                if(not path.exists('lc_filtered_bin'+str(i)+'.fits')):
                    fermipy.gtanalysis.run_gtapp('gtselect', self.logger, kw_gtselect)
                else:
                    print("File exists.  Won't execute.")
#run gtmktime                                                                                                                                                                                             
                kw_gtmktime = dict(evfile='lc_filtered_bin'+str(i)+'.fits',
                           outfile='lc_filtered_gti_bin'+str(i)+'.fits',
                           scfile=self.config['data']['scfile'],
                           roicut=self.config['selection']['roicut'],
                           filter=self.config['selection']['filter'])
                if(not path.exists('lc_filtered_gti_rsp_bin'+str(i)+'.fits')):
                    fermipy.gtanalysis.run_gtapp('gtmktime', self.logger, kw_gtmktime)
                else:
                    print("File exists.  Won't execute.")


#run gtltcube                                                                                                                                                                                             

                kw_gtltcube = dict(evfile='lc_filtered_gti_bin'+str(i)+'.fits',
                                   scfile=self.config['data']['scfile'],
                                   outfile='lc_ltcube_bin'+str(i)+'.fits',
                                   dcostheta=0.025,
                                   binsz=1.0)

                if (not path.exists('lc_ltcube_bin'+str(i)+'.fits')):
                    print("Calculating ltcube for bin "+str(i))

                    fermipy.gtanalysis.run_gtapp('gtltcube', self.logger, kw_gtltcube)

                else:
                    print("LTcube for this bins already exists......")
                                        
                
#run gtexpmap

                kw_expmap = dict( evfile='lc_filtered_gti_bin'+str(i)+'.fits',
                                  outfile='lc_expmap_bin'+str(i)+'.fits',
                                  scfile=self.config['data']['scfile'],
                                  expcube='lc_ltcube_bin'+str(i)+'.fits',
                                  irfs='CALDB',
                                  srcrad=np.int(np.float(self.config['binning']['roiwidth'])+10.0),
                                  nlat=np.int((np.float(self.config['binning']['roiwidth'])+10.0)*4.0),
                                  nlong=np.int((np.float(self.config['binning']['roiwidth'])+10.0)*4.0),
                                  nenergies=20)


                
                if (not path.exists('lc_expmap_bin'+str(i)+'.fits')):
                    print("Calculating expmap for bin "+str(i))

                    fermipy.gtanalysis.run_gtapp('gtexpmap', self.logger, kw_expmap)

                else:
                    print("ExpMap for this bin already exists......")

#run gtdiffrsp

                kw_diffrsp = dict(

                        srcmdl='fit_model_00.xml',
                        evfile='lc_filtered_gti_bin'+str(i)+'.fits',
                        irfs='CALDB',
                        scfile=self.config['data']['scfile'])


                if (not path.exists('lc_filtered_gti_rsp_bin'+str(i)+'.fits')):
                    print("Calculating diffuse responses for bin "+str(i))

                    fermipy.gtanalysis.run_gtapp('gtdiffrsp', self.logger, kw_diffrsp)
                    oldfile = 'lc_filtered_gti_bin'+str(i)+'.fits'     
                    newfile = 'lc_filtered_gti_rsp_bin'+str(i)+'.fits'  
                    subprocess.call(["mv",oldfile, newfile])      
                        
                else:
                    print("Diffuse Responses for this bin already exists......")


                kwUBO = dict(irfs='CALDB',
                               expCube='lc_ltcube_bin'+str(i)+'.fits',
                               expMap='lc_expmap_bin'+str(i)+'.fits',
                               scFile=self.config['data']['scfile']

                               )

                ubo = uba.UnbinnedObs('lc_filtered_gti_rsp_bin'+str(i)+'.fits',**utils.unicode_to_str(kwUBO))

                kwUBA = dict(srcModel='fit_model_00.xml',
                            optimizer='MINUIT')

                UnbinnedA = uba.UnbinnedAnalysis(ubo, **utils.unicode_to_str(kwUBA))
                likeObj = pyLike.Minuit(UnbinnedA.logLike)

                try:
                    UnbinnedA.fit(covar=True,optObject=likeObj)
                except:
                    pass

                self.IntFlux[i] = UnbinnedA.flux(str(self.config['selection']['target']),
                                                 emin=self.config['selection']['emin'], emax=self.config['selection']['emax'])
                self.IntFluxErr[i] = UnbinnedA.fluxError(str(self.config['selection']['target']),
                                                         emin=self.config['selection']['emin'], emax=self.config['selection']['emax'])
                self.TS[i] = UnbinnedA.Ts(str(self.config['selection']['target']))
                self.retCode[i] = likeObj.getRetCode()
                self.npred[i] = UnbinnedA.NpredValue(str(self.config['selection']['target']))
                self.plottimes[i] = 54682.65 + (self.times[i]-239557414.0)/(86400.) + (binning/2.)/86400.
                print(self.plottimes[i], self.IntFlux[i], self.IntFluxErr[i],  self.TS[i],  self.retCode[i], self.npred[i])
#calculate UL for points under TS = 5 if option is chosen                                                                                    
                if(calc_ul == True):                                                           
                    if(self.TS[i] < thresh_TS):
                        ul=UpperLimits(UnbinnedA)
                        ulvals=ul[str(self.config['selection']['target'])].compute()
                        o['IntFlux'][i] = ulvals[0]
                        o['IntFluxErr'][i] = 0.0
                        



        src = self.roi.get_source_by_name(name)
        src.update_data({'LightCurve': copy.deepcopy(o)})

        return o
