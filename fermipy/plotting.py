import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy import wcs
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import pywcsgrid2
import numpy as np
from numpy import ma
import matplotlib.cbook as cbook
from matplotlib.colors import NoNorm, LogNorm, Normalize

from fermipy.utils import merge_dict, AnalysisBase, wcs_to_axes
from fermipy.utils import edge_to_center, edge_to_width, valToEdge

def make_counts_spectrum_plot(o,energies,imfile):

    fig = plt.figure()

    gs = gridspec.GridSpec(2, 1, height_ratios = [1.4,1])
    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[1,0],sharex=ax0)
    
#    axes = axes_grid.Grid(fig,111,
#                          nrows_ncols=(2,1),
#                          axes_pad=0.05,
#                          add_all=True)
#    ax = axes[0]
    
    x = 0.5*(energies[1:] + energies[:-1])
    xerr = 0.5*(energies[1:] - energies[:-1])
    y = o['roi']['counts']
    ym = o['roi']['model_counts']
    

    ax0.errorbar(x,y,yerr=np.sqrt(y),xerr=xerr,color='k',
                 linestyle='None',marker='s',
                 label='Data')

    ax0.errorbar(x,ym,color='k',linestyle='-',marker='None',
                       label='Total')

    src_dict = o['sources']
    
    for v in sorted(src_dict.values(),
                    key=lambda t: t['Npred'],reverse=True)[:6]:
        ax0.errorbar(x,v['model_counts'],linestyle='-',marker='None',
                     label=v['name'])


    for v in sorted(src_dict.values(),
                    key=lambda t: t['Npred'],reverse=True)[6:]:
        ax0.errorbar(x,v['model_counts'],color='gray',
                     linestyle='-',marker='None',
                     label='__nolabel__')

    ax0.set_yscale('log')
    ax0.set_ylim(1,None)
    ax0.set_xlim(energies[0],energies[-1])
    ax0.legend(frameon=False,loc='best',prop={'size' : 8},ncol=2)

    ax1.errorbar(x,(y-ym)/ym,xerr=xerr,yerr=np.sqrt(y)/ym,
                 color='k',linestyle='None',marker='s',
                 label='Data')

    ax1.set_xlabel('Energy [log$_{10}$(E/MeV)]')
    ax1.set_ylabel('Fractional Residual')
    ax0.set_ylabel('Counts')
    
    ax1.set_ylim(-0.4,0.4)
    ax1.axhline(0.0,color='k')
    
    plt.savefig(imfile)
    plt.close(fig)


def load_ds9_cmap():
    # http://tdc-www.harvard.edu/software/saoimage/saoimage.color.html
    ds9_b = {
        'red'   : [[0.0 , 0.0 , 0.0], 
                   [0.25, 0.0 , 0.0], 
                   [0.50, 1.0 , 1.0], 
                   [0.75, 1.0 , 1.0], 
                   [1.0 , 1.0 , 1.0]],
        'green' : [[0.0 , 0.0 , 0.0], 
                   [0.25, 0.0 , 0.0], 
                   [0.50, 0.0 , 0.0], 
                   [0.75, 1.0 , 1.0], 
                   [1.0 , 1.0 , 1.0]],
        'blue'  : [[0.0 , 0.0 , 0.0], 
                   [0.25, 1.0 , 1.0], 
                   [0.50, 0.0 , 0.0], 
                   [0.75, 0.0 , 0.0], 
                   [1.0 , 1.0 , 1.0]]
        }
     
    plt.register_cmap(name='ds9_b', data=ds9_b) 
    plt.cm.ds9_b = plt.cm.get_cmap('ds9_b')
    return plt.cm.ds9_b

class PowerNorm(mpl.colors.Normalize):
    """
    Normalize a given value to the ``[0, 1]`` interval with a power-law
    scaling. This will clip any negative data points to 0.
    """
    def __init__(self, gamma, vmin=None, vmax=None, clip=True):
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)
        self.gamma = gamma

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        gamma = self.gamma
        vmin, vmax = self.vmin, self.vmax
        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin == vmax:
            result.fill(0)
        else:
            if clip:
                mask = ma.getmask(result)
                val = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                mask=mask)
            resdat = result.data
            resdat -= vmin
            np.power(resdat, gamma, resdat)
            resdat /= (vmax - vmin) ** gamma
            result = np.ma.array(resdat, mask=result.mask, copy=False)
            result[(value < 0)&~result.mask] = 0
        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        gamma = self.gamma
        vmin, vmax = self.vmin, self.vmax

        if cbook.iterable(value):
            val = ma.asarray(value)
            return ma.power(value, 1. / gamma) * (vmax - vmin) + vmin
        else:
            return pow(value, 1. / gamma) * (vmax - vmin) + vmin

    def autoscale(self, A):
        """
        Set *vmin*, *vmax* to min, max of *A*.
        """
        self.vmin = ma.min(A)
        if self.vmin < 0:
            self.vmin = 0
            warnings.warn("Power-law scaling on negative values is "
                          "ill-defined, clamping to 0.")

        self.vmax = ma.max(A)

    def autoscale_None(self, A):
        """ autoscale only None-valued vmin or vmax"""
        if self.vmin is None and np.size(A) > 0:
            self.vmin = ma.min(A)
            if self.vmin < 0:
                self.vmin = 0
                warnings.warn("Power-law scaling on negative values is "
                              "ill-defined, clamping to 0.")

        if self.vmax is None and np.size(A) > 0:
            self.vmax = ma.max(A)

class ImagePlotter(object):

    def __init__(self,data,wcs):

        if data.ndim == 3:
            data = np.sum(copy.deepcopy(data),axis=0)
            wcs = pywcs.WCS(wcs.to_header(),naxis=[1,2])
        else:
            data = copy.deepcopy(data)
            
        self._data = data
        self._wcs = wcs
            
    def plot(self,subplot=111,catalog=None,cmap='jet',**kwargs):

        

        kwargs_contour = { 'levels' : None, 'colors' : ['k'],
                           'linewidths' : None,
                           'origin' : 'lower' }
        
        kwargs_imshow = { 'interpolation' : 'nearest',
                          'origin' : 'lower','norm' : None,
                          'vmin' : None, 'vmax' : None }

        zscale = kwargs.get('zscale','lin')
        gamma = kwargs.get('gamma',0.5)
        beam_size = kwargs.get('beam_size',None)
        
        if zscale == 'pow':
            kwargs_imshow['norm'] = PowerNorm(gamma=gamma)
        elif zscale == 'sqrt': 
            kwargs_imshow['norm'] = PowerNorm(gamma=0.5)
        elif zscale == 'log': kwargs_imshow['norm'] = LogNorm()
        elif zscale == 'lin': kwargs_imshow['norm'] = Normalize()
        else: kwargs_imshow['norm'] = Normalize()
        
        ax = pywcsgrid2.subplot(subplot, header=self._wcs.to_header())
#        ax = pywcsgrid2.axes(header=self._wcs.to_header())

        load_ds9_cmap()
        colormap = mpl.cm.get_cmap(cmap)
        colormap.set_under('white')

        data = copy.copy(self._data)
        kwargs_imshow = merge_dict(kwargs_imshow,kwargs)
        kwargs_contour = merge_dict(kwargs_contour,kwargs)
        
        im = ax.imshow(data,**kwargs_imshow)
        im.set_cmap(colormap)

        if kwargs_contour['levels']:        
            cs = ax.contour(data,**kwargs_contour)
        #        plt.clabel(cs, fontsize=5, inline=0)
        
#        im.set_clim(vmin=np.min(self._counts[~self._roi_msk]),
#                    vmax=np.max(self._counts[~self._roi_msk]))
        
        ax.set_ticklabel_type("d", "d")

#        if self._axes[0]._coordsys == 'gal':
#            ax.set_xlabel('GLON')
#            ax.set_ylabel('GLAT')
#        else:        
        ax.set_xlabel('RA')
        ax.set_ylabel('DEC')

        xlabel = kwargs.get('xlabel',None)
        ylabel = kwargs.get('ylabel',None)
        if xlabel is not None: ax.set_xlabel(xlabel)
        if ylabel is not None: ax.set_ylabel(ylabel)

#        plt.colorbar(im,orientation='horizontal',shrink=0.7,pad=0.15,
#                     fraction=0.05)
        ax.grid()
        
#        ax.add_compass(loc=1)
#        ax.set_display_coord_system("gal")       
 #       ax.locator_params(axis="x", nbins=12)

#        ax.add_size_bar(1./self._axes[0]._delta, # 30' in in pixel
#                        r"$1^{\circ}$",loc=3,color='w')
            
        if beam_size is not None:
            ax.add_beam_size(2.0*beam_size[0]/self._axes[0]._delta,
                             2.0*beam_size[1]/self._axes[1]._delta,
                             beam_size[2],beam_size[3],
                             patch_props={'fc' : "none", 'ec' : "w"})
            
#        self._ax = ax
        
        return im, ax

def get_image_wcs(header):

    if header['NAXIS'] == 3:
        wcs = pywcs.WCS(header,naxis=[1,2])
        data = copy.deepcopy(np.sum(hdulist[0].data,axis=0))
    else:
        wcs = pywcs.WCS(header)
        data = copy.deepcopy(hdulist[0].data)
    
    
class ROIPlotter(AnalysisBase):

    defaults = {
        'marker_threshold' : (10,''),
        'source_color'     : ('w','')
        }
    
    def __init__(self,data,wcs,roi,**kwargs):
        AnalysisBase.__init__(self,None,**kwargs)
        self._implot = ImagePlotter(data,wcs)            
        self._roi = roi
        self._data = data.T
        self._wcs = wcs

    @staticmethod
    def create_from_fits(fitsfile,roi,**kwargs):

        hdulist = pyfits.open(fitsfile)        
        header = hdulist[0].header
        header = pyfits.Header.fromstring(header.tostring())
        
#        if header['NAXIS'] == 3:
#            wcs = pywcs.WCS(header,naxis=[1,2])
#            data = copy.deepcopy(np.sum(hdulist[0].data,axis=0))
#        else:
        wcs = pywcs.WCS(header)
        data = copy.deepcopy(hdulist[0].data)

        
        
        return ROIPlotter(data,wcs,roi,**kwargs)

    def plot_projection(self,iaxis,**kwargs):

        data = kwargs.pop('data',self._data)
        noerror = kwargs.pop('noerror',False)
        
        axes = wcs_to_axes(self._wcs,self._data.shape[::-1])
        x = edge_to_center(axes[iaxis])
        w = edge_to_width(axes[iaxis])
        
        c = self.get_data_projection(data,axes,iaxis)
        
        if noerror:
            plt.errorbar(x,c,**kwargs)
        else:
            plt.errorbar(x,c,yerr=c**0.5,xerr=w/2.,**kwargs)

    @staticmethod
    def get_data_projection(data,axes,iaxis,xmin=-1,xmax=1):

        s0 = slice(None,None)
        s1 = slice(None,None)
        s2 = slice(None,None)
        
        if iaxis == 0:
            i0 = valToEdge(axes[iaxis],xmin)
            i1 = valToEdge(axes[iaxis],xmax)
            s1 = slice(i0,i1)
            saxes = [1,2]
        else:
            i0 = valToEdge(axes[iaxis],xmin)
            i1 = valToEdge(axes[iaxis],xmax)
            s0 = slice(i0,i1)
            saxes = [0,2]
            
        c = np.apply_over_axes(np.sum,data[s0,s1,s2],axes=saxes)
        c = np.squeeze(c)

        return c
        
        
    def plot(self,**kwargs):
        
        marker_threshold = 10
        label_threshold = 10
        src_color='w'
        fontweight = 'normal'
        
        im_kwargs = dict(cmap='ds9_b',vmin=None,vmax=None,levels=None,
                         zscale='lin',subplot=111)
        
        plot_kwargs = dict(linestyle='None',marker='+',
                           markerfacecolor = 'None',
                           markeredgecolor=src_color,clip_on=True)
        
        text_kwargs = dict(color=src_color,size=8,clip_on=True,
                           fontweight='normal')

        cb_kwargs = dict(orientation='vertical',shrink=1.0,pad=0.1,
                         fraction=0.1,cb_label=None)
        
        
        im_kwargs = merge_dict(im_kwargs,kwargs,add_new_keys=True)
        plot_kwargs = merge_dict(plot_kwargs,kwargs)
        text_kwargs = merge_dict(text_kwargs,kwargs)
        cb_kwargs = merge_dict(cb_kwargs,kwargs)
                
        im, ax = self._implot.plot(**im_kwargs)
        pixcrd = self._implot._wcs.wcs_world2pix(self._roi._src_skydir.ra.deg,
                                                 self._roi._src_skydir.dec.deg,0)
        
        for i, s in enumerate(self._roi.sources):

            label = s.name
            ax.text(pixcrd[0][i]+2.0,pixcrd[1][i]+2.0,label,
                    **text_kwargs)

    #        if marker_threshold is not None and s['Signif_Avg'] > marker_threshold:      
            ax.plot(pixcrd[0][i],pixcrd[1][i],**plot_kwargs)

        extent = im.get_extent()
        ax.set_xlim(extent[0],extent[1])
        ax.set_ylim(extent[2],extent[3])

        cb_label = cb_kwargs.pop('cb_label',None)
        cb = plt.colorbar(im,**cb_kwargs)
        if cb_label: cb.set_label(cb_label)
        
