# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import numpy as np

from astropy.coordinates import SkyCoord

from fermipy import utils

def fill_evclass_hist(evts, axes):

    
    pass


class Accumulator(object):

    defaults = {
        'scfile' : (None,'',str)
        }
    
    def __init__(self):

        self._energy_bins = 10**np.linspace(1.0,6.0,41)
        self._ctheta_bins = np.linspace(0.0,1.0,11)
        self._dtheta_bins = np.linspace(0.0,1.0,101)**2
        self._evclass_bins = np.linspace(0.0,16.0,17)
        self._evtype_bins = np.linspace(0.0,16.0,17)

        egy = np.sqrt(self._energy_bins[1:]*self._energy_bins[:-1])        
        scale = np.sqrt( (20.*(egy/100.)**-0.8)**2 + 2.**2)
        scale[scale > 30.] = 30.
        self._psf_scale = scale
        
    def create_hist(self, tab, skydir=None):
        """Load events from a table into a table."""
        
        nevt = len(tab)        
        evclass = tab['EVENT_CLASS'][:,::-1]
        evtype = tab['EVENT_TYPE'][:,::-1]

        ebin = utils.val_to_bin(self._energy_bins,tab['ENERGY'])
        scale = self._psf_scale[ebin]
        
        vals = [tab['ENERGY'],
                np.cos(np.radians(tab['THETA']))]

        bins = [self._evclass_bins, self._evtype_bins, self._energy_bins, self._ctheta_bins]
        
        if skydir is not None:
            c = SkyCoord(tab['RA'],tab['DEC'],unit='deg')
            vals += [c.separation(skydir)/scale]
            bins += [self._dtheta_bins]
            
        shape = [len(b)-1 for b in bins]        
        h = np.zeros(shape)

        for i in range(16):
            for j in range(16):

                m = (evclass[:,i] == True) & (evtype[:,j] == True)          
                z = np.vstack([i*np.ones(nevt),j*np.ones(nevt)] + vals)
                z = z[:,m]
                h += np.histogramdd(z.T,bins=bins)[0]
            
        return h

class AGNAccumulator(Accumulator):

    def __init__(self):
        pass
    
class PulsarAccumulator(Accumulator):

    def __init__(self):
        pass
    

    def load(self, tab):

        mon = tab['PULSE_PHASE']

#        off_phase: '0.7/1.0'
#        on_phase: '0.5/0.65,0.1/0.2'
        
        hon = self.load_table(tab)
    

        
