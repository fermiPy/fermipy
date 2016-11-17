# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import itertools
import gzip

import numpy as np

from astropy.coordinates import SkyCoord
from astropy.table import Table, Column
from astropy.io import fits

from fermipy import utils
from fermipy import catalog

agn_src_list = ['3FGL J1104.4+3812', '3FGL J2158.8-3013', '3FGL J1555.7+1111',
                '3FGL J0538.8-4405', '3FGL J1427.0+2347', '3FGL J0222.6+4301',
                '3FGL J1653.9+3945', '3FGL J0721.9+7120', '3FGL J0449.4-4350',
                '3FGL J2254.0+1608', '3FGL J0428.6-3756', '3FGL J0136.5+3905',
                '3FGL J1512.8-0906', '3FGL J1015.0+4925', '3FGL J2000.0+6509',
                '3FGL J1224.9+2122', '3FGL J0957.6+5523', '3FGL J1221.3+3010',
                '3FGL J2202.7+4217', '3FGL J2243.9+2021', '3FGL J0508.0+6736',
                '3FGL J0809.8+5218', '3FGL J0303.4-2407', '3FGL J1256.1-0547',
                '3FGL J0630.9-2406', '3FGL J0738.1+1741', '3FGL J1427.9-4206',
                '3FGL J2009.3-4849', '3FGL J1504.4+1029', '3FGL J0112.1+2245',
                '3FGL J1725.0+1152', '3FGL J0509.4+0541', '3FGL J1217.8+3007',
                '3FGL J2345.2-1554', '3FGL J0033.6-1921', '3FGL J1444.0-3907',
                '3FGL J0612.8+4122', '3FGL J1248.2+5820', '3FGL J1522.1+3144',
                '3FGL J1418.4-0233', '3FGL J1058.6+5627', '3FGL J1150.5+4155',
                '3FGL J0120.4-2700', '3FGL J0238.6+1636', '3FGL J0532.0-4827',
                '3FGL J1037.5+5711', '3FGL J1903.2+5541', '3FGL J0808.2-0751',
                '3FGL J0457.0-2324', '3FGL J1754.1+3212']


def fill_evclass_hist(evts, axes):

    
    pass


def calc_eff(ns0,nb0,ns1,nb1,alpha, sum_axes=None):

    if sum_axes:
        ns0 = np.apply_over_axes(np.sum,ns0,axes=sum_axes)
        nb0 = np.apply_over_axes(np.sum,nb0,axes=sum_axes)
        ns1 = np.apply_over_axes(np.sum,ns1,axes=sum_axes)
        nb1 = np.apply_over_axes(np.sum,nb1,axes=sum_axes)
        
    shape = np.broadcast(ns0,nb0,ns1,nb1).shape
    eff = np.zeros(shape)
    eff_var = np.zeros(shape)

    s0 = ns0-alpha*nb0
    s1 = ns1-alpha*nb1
    mask = (s0*np.ones(shape) > 0)
    mask &= (s1*np.ones(shape) > 0)

    s0[s0<=0] = 1.0
    eff = s1/s0
    eff_var = (((ns0-ns1+alpha**2*(nb0-nb1))*eff**2 +
                (ns1+alpha**2*nb1)*(1-eff)**2)/s0**2)
    
    eff[~mask] = 0.0
    eff_var[~mask] = 0.0
    return eff, eff_var


class Accumulator(object):

    defaults = {
        'scfile' : (None,'',str)
        }
    
    def __init__(self):

        self._energy_bins = 10**np.linspace(1.0,6.0,41)
        self._ctheta_bins = np.linspace(0.0,1.0,11)
        self._xsep_bins = np.linspace(0.0,1.0,101)**2
        self._evclass_bins = np.linspace(0.0,16.0,17)
        self._evtype_bins = np.linspace(0.0,16.0,17)

        egy = np.sqrt(self._energy_bins[1:]*self._energy_bins[:-1])        
        scale = np.sqrt( (20.*(egy/100.)**-0.8)**2 + 2.**2)
        scale[scale > 30.] = 30.
        self._psf_scale = scale
        self._sep_bins = self._xsep_bins[None,:]*self._psf_scale[:,None]
        self._domega = np.pi*(self._sep_bins[:,1:]**2 - self._sep_bins[:,:-1]**2)
        self._hists = {}
        self.init()

    @property
    def hists(self):
        return self._hists
        
    def init(self):

        evclass_shape = [16,40,10]
        evtype_shape = [16,16,40,10]
        evclass_psf_shape = [16,40,10,100]
        evtype_psf_shape = [16,16,40,10,100]
        
        self._hists = dict(evclass_on = np.zeros(evclass_shape),
                           evclass_off = np.zeros(evclass_shape),
                           evclass_alpha = np.zeros([1,40,1]),
                           evtype_on = np.zeros(evtype_shape),
                           evtype_off = np.zeros(evtype_shape),
                           evtype_alpha = np.zeros([1,1,40,1]),
                           evclass_psf_on = np.zeros(evclass_psf_shape),
                           evclass_psf_off = np.zeros(evclass_psf_shape),
                           evtype_psf_on = np.zeros(evtype_psf_shape),
                           evtype_psf_off = np.zeros(evtype_psf_shape),
                           )
                           
                           
    def process(self,filename):

        tab = Table.read(filename,'EVENTS')
        self.load_events(tab)
        
        # Loop over sources
        #for skydir in self._skydirs:
        

    def load_hists(self,filename):

        tab = Table.read(filename,'DATA')

        for k in self.hists.keys():

            if k in ['evclass_on','evclass_off']:
                self.hists[k] += np.array(tab[k.upper()][0])
            elif k in ['evclass_alpha']:
                self.hists[k] = np.array(tab[k.upper()][0])
            
        
    def write(self, outfile, compress=True):

        cols = [Column(name='E_MIN', dtype='f8', data=self._energy_bins[None,:-1], unit='MeV'),
                Column(name='E_MAX', dtype='f8', data=self._energy_bins[None,1:], unit='MeV'),
                Column(name='COSTHETA_MIN', dtype='f8', data=self._ctheta_bins[None,:-1]),
                Column(name='COSTHETA_MAX', dtype='f8', data=self._ctheta_bins[None,1:]),
                Column(name='SEP_MIN', dtype='f8', data=self._sep_bins[None,:,:-1], unit='deg'),
                Column(name='SEP_MAX', dtype='f8', data=self._sep_bins[None,:,1:], unit='deg')]
    
        tab0 = Table(cols)

        cols = []
        for k, v in self.hists.items():
            cols += [Column(name=k, dtype='f8',data=v[None,...])]

        tab1 = Table(cols)
        
        hdulist = fits.HDUList([fits.PrimaryHDU(), fits.table_to_hdu(tab1), fits.table_to_hdu(tab0)])

        hdulist[1].name='DATA'
        hdulist[2].name='AXES'
        hdulist[0].header['DATATYPE'] = self._type

        if compress:
            fp = gzip.GzipFile(outfile + '.gz', 'wb')        
            hdulist.writeto(fp,clobber=True)
        else:
            hdulist.writeto(outfile,clobber=True)

    def calc_sep(self, tab, src_list):

        print('calculating separations')
        
        src_tab = catalog.Catalog3FGL().table
        m = utils.find_rows_by_string(src_tab,src_list,['Source_Name','ASSOC1','ASSOC2'])
        rows = src_tab[m]
        src_skydir = SkyCoord(rows['RAJ2000'],rows['DEJ2000'],unit='deg')
        evt_skydir = SkyCoord(tab['RA'],tab['DEC'],unit='deg')
        lat_skydir = SkyCoord(tab['PtRaz'],tab['PtDecz'],unit='deg')
        evt_sep = evt_skydir.separation(src_skydir[:,None]).deg        
        evt_ebin = utils.val_to_bin(self._energy_bins,tab['ENERGY'])
        evt_xsep = evt_sep/self._psf_scale[evt_ebin][None,:]
        evt_ctheta = np.cos(lat_skydir.separation(src_skydir[:,None]).rad)

        return evt_sep, evt_xsep, evt_ctheta

    def load_events(self, tab):

        hists = self._hists
        evt_sep, evt_xsep, evt_ctheta = self.calc_sep(tab,self._src_list)

        self.fill_alpha()
        
        for sep, xsep, cth in zip(evt_sep, evt_xsep, evt_ctheta):

            print('loading source')
            
            tab['SEP'] = sep
            tab['XSEP'] = xsep
            tab['CTHETA'] = cth
            
            tab_on, tab_off = self.create_onoff(tab)

            print('create hists')
            
            hists['evclass_psf_on'] += self.create_hist(tab_on,fill_sep=True)
            hists['evclass_psf_off'] += self.create_hist(tab_off,fill_sep=True)
            hists['evtype_psf_on'] += self.create_hist(tab_on,fill_sep=True, fill_evtype=True)            
            hists['evtype_psf_off'] += self.create_hist(tab_off,fill_sep=True, fill_evtype=True)
            hists['evclass_on'] += self.create_hist(tab_on)
            hists['evclass_off'] += self.create_hist(tab_off)
            hists['evtype_on'] += self.create_hist(tab_on,fill_evtype=True)
            hists['evtype_off'] += self.create_hist(tab_off,fill_evtype=True)
    
    def create_hist(self, tab, fill_sep=False, fill_evtype=False):
        """Load events from a table into a histogram."""
        
        nevt = len(tab)        
        evclass = tab['EVENT_CLASS'][:,::-1]
        evtype = tab['EVENT_TYPE'][:,::-1]
        xsep = tab['XSEP']
        
        ebin = utils.val_to_bin(self._energy_bins,tab['ENERGY'])
        scale = self._psf_scale[ebin]
        
        vals = [tab['ENERGY'],tab['CTHETA']]
        bins = [self._energy_bins, self._ctheta_bins]

        if fill_sep:
            vals += [xsep]
            bins += [self._xsep_bins]
        
        if fill_evtype:
            loopv = [self._evclass_bins[:-1], self._evtype_bins[:-1]]
            shape = [16,16] + [len(b)-1 for b in bins]
        else:
            loopv = [self._evclass_bins[:-1]]
            shape = [16] + [len(b)-1 for b in bins]            

        h = np.zeros(shape)
        for t in itertools.product(*loopv):
            
            m = (evclass[:,int(t[0])] == True)
            if fill_evtype:
                m &= (evtype[:,int(t[1])] == True)                
            if not np.sum(m):
                continue
                
            z = np.vstack(vals)
            z = z[:,m]

            if fill_evtype:
                h[int(t[0]),int(t[1])] += np.histogramdd(z.T,bins=bins)[0]
            else:
                h[int(t[0])] += np.histogramdd(z.T,bins=bins)[0]
            
        return h

    def calc_eff(self):
        """Calculate the efficiency."""
        
        hists = self.hists

        cth_axis_idx = dict(evclass=2,evtype=3)
        for k in ['evclass','evtype']:

            if k == 'evclass':
                ns0 = hists['evclass_on'][4][None,...]
                nb0 = hists['evclass_off'][4][None,...]
            else:
                ns0 = hists['evclass_on'][4][None,None,...]
                nb0 = hists['evclass_off'][4][None,None,...]

            eff, eff_var = calc_eff(ns0, nb0,
                                    hists['%s_on'%k], hists['%s_off'%k],
                                    hists['%s_alpha'%k])
            hists['%s_cth_eff'%k] = eff
            hists['%s_cth_eff_var'%k] = eff_var

            eff, eff_var = calc_eff(ns0, nb0,
                                    hists['%s_on'%k], hists['%s_off'%k],
                                    hists['%s_alpha'%k],
                                    sum_axes=[cth_axis_idx[k]])
            hists['%s_eff'%k] = eff
            hists['%s_eff_var'%k] = eff_var

        
        
        

class GRAccumulator(Accumulator):

    def __init__(self):
        super(GRAccumulator, self).__init__()
        self._type = 'ridge'

    def create_onoff(self, tab):

        mon = np.abs(tab['GLAT']) < 5.0
        moff = np.abs(tab['GLAT']) > 60.0
        return tab[mon], tab[moff]
    
        
class AGNAccumulator(Accumulator):

    def __init__(self):
        super(AGNAccumulator, self).__init__()
        self._type = 'agn'
        self._src_list = agn_src_list

    def create_onoff(self, tab):

        m = tab['SEP'] < 4.0
        src_tab = tab[m]
        mon = (src_tab['SEP'] < 2.0)
        moff = (src_tab['SEP'] > 2.0) & (src_tab['SEP'] < 4.0)
        return src_tab[mon], src_tab[moff]

    def fill_alpha(self):

        self._hists['evclass_alpha'][...] = 2.0**2/(4.0**2-2.0**2)
        self._hists['evtype_alpha'][...] = 2.0**2/(4.0**2-2.0**2)


        
        
class PSRAccumulator(Accumulator):

    def __init__(self):
        super(PSRAccumulator, self).__init__()
        self._type = 'psr'
        self._src_list = ['Vela']

    def create_onoff(self, tab):

        m = tab['XSEP'] < 1.0
        src_tab = tab[m]
        mon = (src_tab['PULSE_PHASE'] > 0.5) & (src_tab['PULSE_PHASE'] < 0.65)
        mon |= (src_tab['PULSE_PHASE'] > 0.1) & (src_tab['PULSE_PHASE'] < 0.2)
        moff = (src_tab['PULSE_PHASE'] > 0.7) & (src_tab['PULSE_PHASE'] < 1.0)
        return src_tab[mon], src_tab[moff]

    def fill_alpha(self):

        self._hists['evclass_alpha'][...] = (0.15+0.1)/0.3
        self._hists['evtype_alpha'][...] = (0.15+0.1)/0.3
    

            
#        off_phase: '0.7/1.0'
#        on_phase: '0.5/0.65,0.1/0.2'
        
