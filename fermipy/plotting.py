from __future__ import absolute_import, division, print_function, \
    unicode_literals

import copy
import os
import matplotlib

try:
    os.environ['DISPLAY']
except KeyError:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
from matplotlib.patches import Circle, Ellipse, Rectangle
from matplotlib.colors import LogNorm, Normalize, PowerNorm

import astropy.io.fits as pyfits
import astropy.wcs as pywcs
from astropy.coordinates import SkyCoord
import wcsaxes
import numpy as np

import fermipy
import fermipy.config
import fermipy.utils as utils
import fermipy.wcs_utils as wcs_utils
import fermipy.fits_utils as fits_utils
import fermipy.defaults as defaults
import fermipy.roi_model as roi_model
import fermipy.catalog as catalog
from fermipy.utils import merge_dict
from fermipy.skymap import Map, HpxMap
from fermipy.logger import Logger
from fermipy.logger import logLevel


def draw_arrows(x, y, color='k'):
    for t, z in zip(x, y):
        plt.arrow(t, z, 0.0, -z * 0.2, fc=color, ec=color,
                  head_width=t * 0.1, head_length=z * 0.05)


def get_xerr(sed):
    delo = 10 ** sed['ecenter'] - 10 ** sed['emin']
    dehi = 10 ** sed['emax'] - 10 ** sed['ecenter']
    xerr = np.vstack((delo, dehi))
    return xerr


def make_counts_spectrum_plot(o, roi, energies, imfile):
    fig = plt.figure()

    gs = gridspec.GridSpec(2, 1, height_ratios=[1.4, 1])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)

    #    axes = axes_grid.Grid(fig,111,
    #                          nrows_ncols=(2,1),
    #                          axes_pad=0.05,
    #                          add_all=True)
    #    ax = axes[0]

    x = 0.5 * (energies[1:] + energies[:-1])
    xerr = 0.5 * (energies[1:] - energies[:-1])
    y = o['counts']
    ym = o['model_counts']

    ax0.errorbar(x, y, yerr=np.sqrt(y), xerr=xerr, color='k',
                 linestyle='None', marker='s',
                 label='Data')

    ax0.errorbar(x, ym, color='k', linestyle='-', marker='None',
                 label='Total')

    for s in sorted(roi.sources,
                    key=lambda t: t['npred'], reverse=True)[:6]:
        ax0.errorbar(x, s['model_counts'], linestyle='-', marker='None',
                     label=s['name'])

    for s in sorted(roi.sources,
                    key=lambda t: t['npred'], reverse=True)[6:]:
        ax0.errorbar(x, s['model_counts'], color='gray',
                     linestyle='-', marker='None',
                     label='__nolabel__')

    ax0.set_yscale('log')
    ax0.set_ylim(0.1, None)
    ax0.set_xlim(energies[0], energies[-1])
    ax0.legend(frameon=False, loc='best', prop={'size': 8}, ncol=2)

    ax1.errorbar(x, (y - ym) / ym, xerr=xerr, yerr=np.sqrt(y) / ym,
                 color='k', linestyle='None', marker='s',
                 label='Data')

    ax1.set_xlabel('Energy [log$_{10}$(E/MeV)]')
    ax1.set_ylabel('Fractional Residual')
    ax0.set_ylabel('Counts')

    ax1.set_ylim(-0.4, 0.4)
    ax1.axhline(0.0, color='k')

    plt.savefig(imfile)
    plt.close(fig)




def load_ds9_cmap():
    # http://tdc-www.harvard.edu/software/saoimage/saoimage.color.html
    ds9_b = {
        'red': [[0.0, 0.0, 0.0],
                [0.25, 0.0, 0.0],
                [0.50, 1.0, 1.0],
                [0.75, 1.0, 1.0],
                [1.0, 1.0, 1.0]],
        'green': [[0.0, 0.0, 0.0],
                  [0.25, 0.0, 0.0],
                  [0.50, 0.0, 0.0],
                  [0.75, 1.0, 1.0],
                  [1.0, 1.0, 1.0]],
        'blue': [[0.0, 0.0, 0.0],
                 [0.25, 1.0, 1.0],
                 [0.50, 0.0, 0.0],
                 [0.75, 0.0, 0.0],
                 [1.0, 1.0, 1.0]]
    }

    plt.register_cmap(name='ds9_b', data=ds9_b)
    plt.cm.ds9_b = plt.cm.get_cmap('ds9_b')
    return plt.cm.ds9_b

def load_bluered_cmap():
    bluered = {'red':   ((0.0, 0.0, 0.0),
                        (0.5, 0.0, 0.0),
                        (1.0, 1.0, 1.0)),

               'green': ((0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0)),

               'blue':  ((0.0, 0.0, 1.0),
                        (0.5, 0.0, 0.0),
                        (1.0, 0.0, 0.0))
    }

    plt.register_cmap(name='bluered', data=bluered)
    plt.cm.bluered = plt.cm.get_cmap('bluered')
    return plt.cm.bluered


def annotate(**kwargs):
    ax = kwargs.pop('ax', plt.gca())
    erange = kwargs.pop('erange', None)
    src = kwargs.pop('src', None)

    text = []

    if src:
        
        if 'ASSOC1' in src['assoc'] and src['assoc']['ASSOC1']:
            text += ['%s (%s)' % (src['name'], src['assoc']['ASSOC1'])]
        else:
            text += [src['name']]

    if erange:
        text += ['E = %.3f - %.3f GeV' % (10 ** erange[0] / 1E3,
                                          10 ** erange[1] / 1E3)]

    if not text: return

    ax.annotate('\n'.join(text),
                xy=(0.05, 0.93),
                xycoords='axes fraction', fontsize=12,
                xytext=(-5, 5), textcoords='offset points',
                ha='left', va='top')


class ImagePlotter(object):

    def __init__(self, data, proj):

        if isinstance(proj,pywcs.WCS):
            self._projtype = 'WCS'
            if data.ndim == 3:
                data = np.sum(copy.deepcopy(data), axis=2)
                proj = pywcs.WCS(proj.to_header(), naxis=[1, 2])
            else:
                data = copy.deepcopy(data)        
            self._proj = proj
            self._wcs = proj            
        elif isinstance(proj,HPX):
            self._projtype = 'HPX'
            self._proj = proj
            self._wcs,data = make_wcs_from_hpx(proj,data)
        else:
            raise Exception("Can't co-add map of unknown type %s"%type(proj))
                
        self._data = data


    @property
    def projtype(self):
        return self._projtype


    def plot(self, subplot=111, cmap='jet', **kwargs):

        kwargs_contour = {'levels': None, 'colors': ['k'],
                          'linewidths': None}

        kwargs_imshow = {'interpolation': 'nearest',
                         'origin': 'lower', 'norm': None,
                         'vmin': None, 'vmax': None}

        zscale = kwargs.get('zscale', 'lin')
        gamma = kwargs.get('gamma', 0.5)

        if zscale == 'pow':
            kwargs_imshow['norm'] = PowerNorm(gamma=gamma)
        elif zscale == 'sqrt':
            kwargs_imshow['norm'] = PowerNorm(gamma=0.5)
        elif zscale == 'log':
            kwargs_imshow['norm'] = LogNorm()
        elif zscale == 'lin':
            kwargs_imshow['norm'] = Normalize()
        else:
            kwargs_imshow['norm'] = Normalize()

        fig = plt.gcf()
        
        ax = fig.add_subplot(subplot,
                             projection=wcsaxes.WCS(self._wcs.to_header()))

        load_ds9_cmap()
        colormap = matplotlib.cm.get_cmap(cmap)
        colormap.set_under(colormap(0))

        data = copy.copy(self._data)
        kwargs_imshow = merge_dict(kwargs_imshow, kwargs)
        kwargs_contour = merge_dict(kwargs_contour, kwargs)

        im = ax.imshow(data.T, **kwargs_imshow)
        im.set_cmap(colormap)

        if kwargs_contour['levels']:
            cs = ax.contour(data.T, **kwargs_contour)
            cs.levels = ['%.0f'%val for val in cs.levels]
            plt.clabel(cs,inline=1,fontsize=8)
            
        coordsys = wcs_utils.get_coordsys(self._proj)

        if coordsys == 'CEL':
            ax.set_xlabel('RA')
            ax.set_ylabel('DEC')
        elif coordsys == 'GAL':
            ax.set_xlabel('GLON')
            ax.set_ylabel('GLAT')

        xlabel = kwargs.get('xlabel', None)
        ylabel = kwargs.get('ylabel', None)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        #        plt.colorbar(im,orientation='horizontal',shrink=0.7,pad=0.15,
        #                     fraction=0.05)
        ax.coords.grid(color='white')  # , alpha=0.5)
        #       ax.locator_params(axis="x", nbins=12)

        return im, ax


class ROIPlotter(fermipy.config.Configurable):
    defaults = {
        'erange': (None, '', list),
        'catalogs': (None, '', list),
        'graticule_radii': (None, '', list),
        'label_ts_threshold': (0.0, '', float),
        'cmap': ('ds9_b', '', str),
    }

    def __init__(self, data_map, **kwargs):
        self._roi = kwargs.pop('roi',None)
        super(ROIPlotter,self).__init__(None,**kwargs)        
        #fermipy.config.Configurable.__init__(self, None, **kwargs)

        self._data_map = data_map
        self._catalogs = []
        for c in self.config['catalogs']:
            if isinstance(c,str):            
                self._catalogs += [catalog.Catalog.create(c)]
            else:
                self._catalogs += [c]
                
        if isinstance(data_map,Map):
            self._projtype = 'WCS'
            self._data = data_map.counts.T
            self._proj = data_map.wcs
            self._wcs = self._proj
        elif isinstance(data_map,HpxMap):
            self._projtype = 'HPX'
            self._proj = data_map.hpx
            self._wcs,dataT = data_map.make_wcs_from_hpx(sum_ebins=False,
                                                    proj='CAR',
                                                    oversample=2)
            self._data = dataT.T
        else:
            raise Exception("Can't make ROIPlotter of unknown projection type %s"%type(data_map))
    
        self._erange = self.config['erange']

        if self._erange:
            axes = wcs_utils.wcs_to_axes(self._wcs, self._data.shape[::-1])
            i0 = utils.val_to_edge(axes[2], self._erange[0])
            i1 = utils.val_to_edge(axes[2], self._erange[1])
            imdata = self._data[:, :, i0:i1]
        else:
            imdata = self._data
        
        self._implot = ImagePlotter(imdata, self._wcs)

    @property
    def data(self):
        return self._data

    @property
    def cmap(self):
        return self._data_map

    @property
    def projtype(self):
        return self._projtype   

    @property
    def proj(self):
        return self._proj

    @staticmethod
    def create_from_fits(fitsfile, roi, **kwargs):

        hdulist = pyfits.open(fitsfile)
        try:
            if hdulist[1].name == "SKYMAP":
                projtype = "HPX"
            else:
                projtype = "WCS"
        except:
            projtype = "WCS"

        if projtype == "WCS":
            header = hdulist[0].header
            header = pyfits.Header.fromstring(header.tostring())
            wcs = pywcs.WCS(header)
            data = copy.deepcopy(hdulist[0].data)
            themap = Map(data, wcs)
        elif projtype == "HPX":
            themap = HpxMap.create_from_hdulist(hdulist,ebounds="EBOUNDS")            
        else:
            raise Exception("Unknown projection type %s"%projtype)        
        
        return ROIPlotter(themap, roi=roi, **kwargs)

    def plot_projection(self, iaxis, **kwargs):

        data = kwargs.pop('data', self._data)
        noerror = kwargs.pop('noerror', False)

        axes = wcs_utils.wcs_to_axes(self._wcs, self._data.shape[::-1])
        x = utils.edge_to_center(axes[iaxis])
        w = utils.edge_to_width(axes[iaxis])

        c = self.get_data_projection(data, axes, iaxis, erange=self._erange)

        if noerror:
            plt.errorbar(x, c, **kwargs)
        else:
            plt.errorbar(x, c, yerr=c ** 0.5, xerr=w / 2., **kwargs)

    @staticmethod
    def get_data_projection(data, axes, iaxis, xmin=-1, xmax=1, erange=None):

        s0 = slice(None, None)
        s1 = slice(None, None)
        s2 = slice(None, None)

        if iaxis == 0:
            i0 = utils.val_to_edge(axes[iaxis], xmin)
            i1 = utils.val_to_edge(axes[iaxis], xmax)
            s1 = slice(i0, i1)
            saxes = [1, 2]
        else:
            i0 = utils.val_to_edge(axes[iaxis], xmin)
            i1 = utils.val_to_edge(axes[iaxis], xmax)
            s0 = slice(i0, i1)
            saxes = [0, 2]

        if erange is not None:
            j0 = utils.val_to_edge(axes[2], erange[0])
            j1 = utils.val_to_edge(axes[2], erange[1])
            s2 = slice(j0, j1)
            
        c = np.apply_over_axes(np.sum, data[s0, s1, s2], axes=saxes)
        c = np.squeeze(c)
        return c

    @staticmethod
    def setup_projection_axis(iaxis, erange=None):

        plt.gca().legend(frameon=False, prop={'size': 10})
        plt.gca().set_ylabel('Counts')
        if iaxis == 0:
            plt.gca().set_xlabel('LON Offset [deg]')
        else:
            plt.gca().set_xlabel('LAT Offset [deg]')

    def plot_sources(self,skydir,labels,
                     plot_kwargs,text_kwargs, **kwargs):

        ax = plt.gca()

        nolabels = kwargs.get('nolabels',False)
        label_mask = kwargs.get('label_mask',
                                np.ones(len(labels),dtype=bool))
        if nolabels:
            label_mask.fill(False)
        
        pixcrd = wcs_utils.skydir_to_pix(skydir, self._implot._wcs)
        
        for i, (x,y,label,show_label) in enumerate(zip(pixcrd[0],pixcrd[1],labels,label_mask)):

            if show_label:
                t = ax.annotate(label,xy=(x,y),
                                xytext=(5.0, 5.0), textcoords='offset points',
                                **text_kwargs)            
                plt.setp(t, path_effects=[PathEffects.withStroke(linewidth=2.0, foreground="black")])
            
            t = ax.plot(x, y, **plot_kwargs)
            plt.setp(t, path_effects=[PathEffects.withStroke(linewidth=2.0, foreground="black")])
        
    def plot_roi(self, roi, **kwargs):

        src_color = 'w'
        
        label_ts_threshold = kwargs.get('label_ts_threshold',0.0)
        plot_kwargs = dict(linestyle='None', marker='+',
                           markerfacecolor='None',mew=0.66,ms=8,
#                           markersize=8,
                           markeredgecolor=src_color, clip_on=True)

        text_kwargs = dict(color=src_color, size=8, clip_on=True,
                           fontweight='normal')

        ts = np.array([s['ts'] for s in roi.point_sources])

        if label_ts_threshold is None:
            m = np.zeros(len(ts),dtype=bool)            
        elif label_ts_threshold <= 0:
            m = np.ones(len(ts),dtype=bool)            
        else:
            m = ts > label_ts_threshold
            
        skydir = roi._src_skydir
        labels = [s.name for s in roi.point_sources]        
        self.plot_sources(skydir,labels,plot_kwargs,text_kwargs,
                          label_mask=m, **kwargs)

    def plot_catalog(self, catalog):

        color = 'lime'
        
        plot_kwargs = dict(linestyle='None', marker='x',
                           markerfacecolor='None',
                           markeredgecolor=color, clip_on=True)

        text_kwargs = dict(color=color, size=8, clip_on=True,
                           fontweight='normal')
                
        skydir = catalog.skydir

        if 'NickName' in catalog.table.columns:
            labels = catalog.table['NickName']
        else:
            labels = catalog.table['Source_Name']
            
        separation = skydir.separation(self.cmap.skydir).deg
        m = separation < max(self.cmap.width)

        self.plot_sources(skydir[m],labels[m],plot_kwargs,text_kwargs,
                          nolabels=True)

            
    def plot(self, **kwargs):
        
        zoom = kwargs.get('zoom',None)
        graticule_radii = kwargs.get('graticule_radii',
                                     self.config['graticule_radii'])
        label_ts_threshold = kwargs.get('label_ts_threshold',
                                       self.config['label_ts_threshold'])
        cmap = kwargs.setdefault('cmap',self.config['cmap'])
        
        im_kwargs = dict(cmap='ds9_b',
                         interpolation='nearest',
                         vmin=None, vmax=None, levels=None,
                         zscale='lin', subplot=111)

        cb_kwargs = dict(orientation='vertical', shrink=1.0, pad=0.1,
                         fraction=0.1, cb_label=None)

        im_kwargs = merge_dict(im_kwargs, kwargs)
        cb_kwargs = merge_dict(cb_kwargs, kwargs)
        
        im, ax = self._implot.plot(**im_kwargs)

        self._ax = ax
        
        for c in self._catalogs:
            self.plot_catalog(c)
        
        if self._roi is not None:
            self.plot_roi(self._roi,
                          label_ts_threshold=label_ts_threshold)        
            
        self._extent = im.get_extent()
        ax.set_xlim(self._extent[0], self._extent[1])
        ax.set_ylim(self._extent[2], self._extent[3])
        
        self.zoom(zoom)
            
        cb_label = cb_kwargs.pop('cb_label', None)
        cb = plt.colorbar(im, **cb_kwargs)
        if cb_label:
            cb.set_label(cb_label)

        for r in graticule_radii:            
            self.draw_circle(self.cmap.skydir,r)
        

    def draw_circle(self,skydir,radius):

        coordsys = wcs_utils.get_coordsys(self.proj)
        #if coordsys == 'GAL':            
        #    c = Circle((skydir.galactic.l.deg,skydir.galactic.b.deg),
        #               radius,facecolor='none',edgecolor='w',linestyle='--',
        #               transform=self._ax.get_transform('galactic'))
        #elif coordsys == 'CEL':            
        #    c = Circle((skydir.fk5.l.deg,skydir.fk5.b.deg),
        #               radius,facecolor='none',edgecolor='w',linestyle='--',
        #               transform=self._ax.get_transform('fk5'))
        
        c = Circle(self.cmap.pix_center,radius/max(self.cmap.pix_size),
                   facecolor='none',edgecolor='w',linestyle='--',linewidth=0.5)
        
        self._ax.add_patch(c)

    def zoom(self,zoom):

        if zoom is None:
            return

        extent = self._extent
        
        xw = extent[1]-extent[0]
        x0 = 0.5*(extent[0]+extent[1])
        yw = extent[1]-extent[0]
        y0 = 0.5*(extent[0]+extent[1])
                                
        xlim = [x0 - 0.5*xw/zoom,x0 + 0.5*xw/zoom]
        ylim = [y0 - 0.5*yw/zoom,y0 + 0.5*yw/zoom]
        
        self._ax.set_xlim(xlim[0],xlim[1])
        self._ax.set_ylim(ylim[0],ylim[1])

class SEDPlotter(object):
    def __init__(self, src):

        self._src = copy.deepcopy(src)
        self._sed = copy.deepcopy(self._src['sed'])

    @staticmethod
    def plot_lnlscan(sed, **kwargs):

        ax = kwargs.pop('ax', plt.gca())
        llhCut = kwargs.pop('llhCut', -2.70)
        cmap = kwargs.pop('cmap', 'BuGn')

        lhProf = sed['lnlprofile']

        fmin = min(-8, np.log10(np.min(sed['e2dfde_ul95'])) - 0.5)
        fmax = max(-5, np.log10(np.max(sed['e2dfde_ul95'])) + 0.5)

        fluxM = np.arange(fmin, fmax, 0.01)
        fbins = len(fluxM)
        llhMatrix = np.zeros((len(sed['ecenter']), fbins))

        # loop over energy bins
        for i in range(len(lhProf)):
            m = lhProf[i]['dfde'] > 0
            flux = np.log10(
                lhProf[i]['dfde'][m] * (10 ** sed['ecenter'][i]) ** 2)
            logl = lhProf[i]['dloglike'][m]
            logli = np.interp(fluxM, flux, logl)
            logli[fluxM > flux[-1]] = logl[-1]
            logli[fluxM < flux[0]] = logl[0]
            llhMatrix[i, :] = logli

        xedge = np.logspace(sed['emin'][0], sed['emax'][-1],
                            len(sed['ecenter']) + 1)
        yedge = np.logspace(fmin, fmax, fbins)
        xedge, yedge = np.meshgrid(xedge, yedge)
        im = ax.pcolormesh(xedge, yedge, llhMatrix.T,
                           vmin=llhCut, vmax=0, cmap=cmap)
        cb = plt.colorbar(im)
        cb.set_label('Delta LogLikelihood')

        plt.gca().set_ylim(10 ** fmin, 10 ** fmax)
        plt.gca().set_yscale('log')
        plt.gca().set_xscale('log')
        plt.gca().set_xlim(10 ** sed['emin'][0], 10 ** sed['emax'][-1])

    @staticmethod
    def plot_sed(sed, **kwargs):

        ts_thresh = kwargs.pop('ts_thresh', 4)
        kwargs.setdefault('marker', 'o')
        kwargs.setdefault('linestyle', 'None')
        kwargs.setdefault('color', 'k')
        color = kwargs.get('color', 'k')

        m = sed['ts'] < ts_thresh

        x = 10 ** sed['ecenter']
        y = sed['e2dfde']
        yerr = sed['e2dfde_err']
        yerr_lo = sed['e2dfde_err_lo']
        yerr_hi = sed['e2dfde_err_hi']
        yul = sed['e2dfde_ul95']

        y[m] = yul[m]
        yerr[m] = 0
        yerr_lo[m] = 0
        yerr_hi[m] = 0

        delo = 10 ** sed['ecenter'] - 10 ** sed['emin']
        dehi = 10 ** sed['emax'] - 10 ** sed['ecenter']
        xerr0 = np.vstack((delo[m], dehi[m]))
        xerr1 = np.vstack((delo[~m], dehi[~m]))

        plt.errorbar(x[~m], y[~m], xerr=xerr1, yerr=(yerr_lo[~m], yerr_hi[~m]), **kwargs)
        plt.errorbar(x[m], yul[m], xerr=xerr0, yerr=yul[m]*0.2, uplims=True, **kwargs)

        plt.gca().set_yscale('log')
        plt.gca().set_xscale('log')
        plt.gca().set_xlim(10 ** sed['emin'][0], 10 ** sed['emax'][-1])
        plt.gca().set_ylim(min(1E-8,np.min(y)*0.5),max(1E-5,np.max(y)*1.5))
        

    @staticmethod
    def plot_sed_resid(src, model_flux, **kwargs):

        sed = src['sed']

        m = sed['ts'] < 4

        x = 10 ** sed['ecenter']
        y = sed['e2dfde']
        yerr = sed['e2dfde_err']
        yul = sed['e2dfde_ul95']

        y[m] = yul[m]
        yerr[m] = 0

        delo = 10 ** sed['ecenter'] - 10 ** sed['emin']
        dehi = 10 ** sed['emax'] - 10 ** sed['ecenter']
        xerr = np.vstack((delo, dehi))

        ym = np.interp(sed['ecenter'],
                       model_flux['ecenter'],
                       10 ** (2 * model_flux['ecenter']) * model_flux['dfde'])

        plt.errorbar(x, (y - ym) / ym, xerr=xerr, yerr=yerr / ym, **kwargs)

    @staticmethod
    def plot_model(model_flux, **kwargs):

        ax = plt.gca()
        color = kwargs.pop('color', 'k')
        noband = kwargs.pop('noband', False)

        e2 = 10 ** (2 * model_flux['ecenter'])

        ax.plot(10 ** model_flux['ecenter'],
                model_flux['dfde'] * e2, color=color, **kwargs)

        ax.plot(10 ** model_flux['ecenter'],
                model_flux['dfde_lo'] * e2, color=color,
                linestyle='--', **kwargs)
        ax.plot(10 ** model_flux['ecenter'],
                model_flux['dfde_hi'] * e2, color=color,
                linestyle='--', **kwargs)

        if not noband:
            ax.fill_between(10 ** model_flux['ecenter'],
                            model_flux['dfde_lo'] * e2,
                            model_flux['dfde_hi'] * e2,
                            alpha=0.5, color=color, zorder=-1)

    @staticmethod
    def annotate(src, xy=(0.05, 0.93)):

        ax = plt.gca()

        name = src['name']

        if src['assoc']:
            name += ' (%s)' % src['assoc']

        ax.annotate(name,
                    xy=xy,
                    xycoords='axes fraction', fontsize=12,
                    xytext=(-5, 5), textcoords='offset points',
                    ha='left', va='center')

    def plot(self, showlnl=False, **kwargs):

        sed = self._sed
        src = self._src
        ax = plt.gca()
        name = src['name']
        cmap = kwargs.get('cmap', 'BuGn')

        annotate(src=src, ax=ax)

        SEDPlotter.plot_sed(sed)

        if src['ts'] > 9.:
        
            if 'model_flux' in sed:
                SEDPlotter.plot_model(sed['model_flux'], noband=showlnl)        
            elif 'model_flux' in src:
                SEDPlotter.plot_model(src, noband=showlnl)

        if showlnl:
            SEDPlotter.plot_lnlscan(sed, cmap=cmap)

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('Energy [MeV]')
        ax.set_ylabel('E$^{2}$dF/dE [MeV cm$^{-2}$ s$^{-1}$]')


class ExtensionPlotter(object):
    def __init__(self, src, roi, suffix, workdir, erange=None):

        self._src = copy.deepcopy(src)

        name = src['name'].lower().replace(' ', '_')

        self._file0 = os.path.join(workdir,
                                   'mcube_%s_noext%s.fits' % (name, suffix))
        self._file1 = os.path.join(workdir,
                                   'mcube_%s_ext_bkg%s.fits' % (name, suffix))
        self._file2 = os.path.join(workdir, 'ccube%s.fits' % suffix)

        self._files = []
        self._width = src['extension']['width']
        for i, w in enumerate(src['extension']['width']):
            self._files += [os.path.join(workdir, 'mcube_%s_ext%02i%s.fits' % (
                name, i, suffix))]
        self._roi = roi
        self._erange = erange

    def plot(self, iaxis):

        p0 = ROIPlotter.create_from_fits(self._file2, roi=self._roi,
                                         erange=self._erange)
        p1 = ROIPlotter.create_from_fits(self._file1, roi=self._roi,
                                         erange=self._erange)
        p0.plot_projection(iaxis, color='k', label='Data', marker='s',
                           linestyle='None')
        p1.plot_projection(iaxis, color='b', noerror=True, label='Background')

        n = len(self._width)
        step = max(1, int(n / 5.))

        fw = zip(self._files, self._width)[::step]

        for i, (f, w) in enumerate(fw):
            cf = float(i) / float(len(fw) - 1.0)
            cf = 0.2 + cf * 0.8

            p = ROIPlotter.create_from_fits(f, roi=self._roi, erange=self._erange)
            p._data += p1.data
            p.plot_projection(iaxis, color=matplotlib.cm.Reds(cf),
                              noerror=True, label='%.4f$^\circ$' % w)


class AnalysisPlotter(fermipy.config.Configurable):
    defaults = dict(defaults.plotting.items(),
                    fileio=defaults.fileio,
                    logging=defaults.logging)

    def __init__(self, config, **kwargs):
        fermipy.config.Configurable.__init__(self, config, **kwargs)

        self._catalogs = []
        for c in self.config['catalogs']:
            self._catalogs += [catalog.Catalog.create(c)]
        
        self.logger = Logger.get(self.__class__.__name__,
                                 self.config['fileio']['logfile'],
                                 logLevel(self.config['logging']['verbosity']))

    def run(self, gta, mcube_map, **kwargs):
        """Make all plots."""
        
        prefix = kwargs.get('prefix', 'test')
        format = kwargs.get('format', gta.config['plotting']['format'])

        erange = [None] + gta.config['plotting']['erange']

        for x in erange:
            self.make_roi_plots(gta, mcube_map, prefix, erange=x,
                                format=format)
        # self.make_extension_plots(gta,prefix, erange=x,
        # format=format)

        self.make_sed_plots(gta, prefix, format=format)

        imfile = utils.format_filename(gta.config['fileio']['workdir'],
                                       'counts_spectrum', prefix=[prefix],
                                       extension=format)

        make_counts_spectrum_plot(gta._roi_model, gta.roi, gta.energies,
                                  imfile)

    def make_residual_plots(self, gta, maps, **kwargs):

        format = kwargs.get('format', gta.config['plotting']['format'])

        if 'sigma' not in maps: 
            return

        # Reload maps from FITS file

        sigma_levels = [-5,-3,3,5,7] + list(np.logspace(1,3,17))

        kwargs.setdefault('graticule_radii',self.config['graticule_radii'])
        kwargs.setdefault('label_ts_threshold',
                          self.config['label_ts_threshold'])
        kwargs.setdefault('cmap',self.config['cmap'])
        kwargs.setdefault('catalogs',self._catalogs)
                 
        load_bluered_cmap()

        prefix = maps['name']
        fig = plt.figure()
        p = ROIPlotter(maps['sigma'], roi=gta.roi, **kwargs)
        p.plot(vmin=-5, vmax=5, levels=sigma_levels,
               cb_label='Significance [$\sigma$]',interpolation='bicubic', cmap='bluered')
        plt.savefig(utils.format_filename(gta.config['fileio']['workdir'],
                                          'residmap_sigma',
                                          prefix=[prefix],
                                          extension=format))
        plt.close(fig)

        fig = plt.figure()
        p = ROIPlotter(maps['data'], roi=gta.roi, **kwargs)
        p.plot(cb_label='Counts',interpolation='bicubic')
        plt.savefig(utils.format_filename(gta.config['fileio']['workdir'],
                                          'residmap_data',
                                          prefix=[prefix],
                                          extension=format))
        plt.close(fig)

        fig = plt.figure()
        p = ROIPlotter(maps['model'], roi=gta.roi, **kwargs)
        p.plot(cb_label='Counts',interpolation='bicubic')
        plt.savefig(utils.format_filename(gta.config['fileio']['workdir'],
                                          'residmap_model',
                                          prefix=[prefix],
                                          extension=format))
        plt.close(fig)

        fig = plt.figure()
        p = ROIPlotter(maps['excess'], roi=gta.roi, **kwargs)
        p.plot(cb_label='Counts',interpolation='bicubic')
        plt.savefig(utils.format_filename(gta.config['fileio']['workdir'],
                                          'residmap_excess',
                                          prefix=[prefix],
                                          extension=format))
        plt.close(fig)

    def make_tsmap_plots(self, gta, maps, **kwargs):

        format = kwargs.get('format', gta.config['plotting']['format'])
        suffix = kwargs.get('suffix', 'tsmap')
        zoom = kwargs.get('zoom',None)
        
        if 'ts' not in maps: 
            return

        sigma_levels = [3,5,7] + list(np.logspace(1,3,17))

        kwargs.setdefault('graticule_radii',self.config['graticule_radii'])
        kwargs.setdefault('label_ts_threshold',
                          self.config['label_ts_threshold'])
        kwargs.setdefault('cmap',self.config['cmap'])
        kwargs.setdefault('catalogs',self.config['catalogs'])
        
        prefix = maps['name']
        fig = plt.figure()
        p = ROIPlotter(maps['sqrt_ts'], roi=gta.roi, **kwargs)
        p.plot(vmin=0, vmax=5, levels=sigma_levels,
               cb_label='Sqrt(TS) [$\sigma$]',interpolation='bicubic',
               zoom=zoom)
        plt.savefig(utils.format_filename(gta.config['fileio']['workdir'],
                                          '%s_sqrt_ts'%suffix,
                                          prefix=[prefix],
                                          extension=format))
        plt.close(fig)

        fig = plt.figure()
        p = ROIPlotter(maps['npred'], roi=gta.roi, **kwargs)
        p.plot(vmin=0, cb_label='NPred [Counts]',interpolation='bicubic',
               zoom=zoom)
        plt.savefig(utils.format_filename(gta.config['fileio']['workdir'],
                                          '%s_npred'%suffix,
                                          prefix=[prefix],
                                          extension=format))
        plt.close(fig)

    def make_roi_plots(self, gta, mcube_map, prefix, erange=None, **kwargs):
        """Make various diagnostic plots for the 1D and 2D
        counts/model distributions.

        Parameters
        ----------

        prefix : str
            Prefix that will be appended to all filenames.

        """

        format = kwargs.get('format', gta.config['plotting']['format'])

        roi_kwargs = {}
        roi_kwargs.setdefault('erange',erange)
        roi_kwargs.setdefault('graticule_radii',self.config['graticule_radii'])
        roi_kwargs.setdefault('label_ts_threshold',
                              self.config['label_ts_threshold'])
        roi_kwargs.setdefault('cmap',self.config['cmap'])
        roi_kwargs.setdefault('catalogs',self._catalogs)
        
        if erange is None:
            erange = (gta.energies[0], gta.energies[-1])
        esuffix = '_%.3f_%.3f' % (erange[0], erange[1])

        mcube_diffuse = gta.model_counts_map('diffuse')

        fig = plt.figure()
        p = ROIPlotter(mcube_map, roi=gta.roi, **roi_kwargs)
        p.plot(cb_label='Counts', zscale='pow', gamma=1. / 3.)
        plt.savefig(os.path.join(gta.config['fileio']['workdir'],
                                 '%s_model_map%s.%s' % (
                    prefix, esuffix, format)))
        plt.close(fig)


        colors = ['k', 'b', 'g', 'r']
        data_style = {'marker': 's', 'linestyle': 'None'}

        fig = plt.figure()
        p = ROIPlotter(gta.counts_map(), roi=gta.roi, **roi_kwargs)
        
        if p.projtype == "WCS":
            model_data = mcube_map.counts.T
            diffuse_data = mcube_diffuse.counts.T
        elif p.projtype == "HPX":
            dummy,model_dataT = p.cmap.convert_to_cached_wcs(mcube_map.counts,sum_ebins=False)
            dummy,diffuse_dataT = p.cmap.convert_to_cached_wcs(mcube_diffuse.counts,sum_ebins=False)
            model_data = model_dataT.T
            diffuse_data = diffuse_dataT.T

        p.plot(cb_label='Counts', zscale='sqrt')
        plt.savefig(os.path.join(gta.config['fileio']['workdir'],
                                 '%s_counts_map%s.%s' % (
                                     prefix, esuffix, format)))
        plt.close(fig)

        fig = plt.figure()
        p.plot_projection(0, label='Data', color='k', **data_style)

        
        p.plot_projection(0, data=model_data, label='Model',
                          noerror=True)
        p.plot_projection(0, data=diffuse_data, label='Diffuse',
                          noerror=True)
        plt.gca().set_ylabel('Counts')
        plt.gca().set_xlabel('LON Offset [deg]')
        plt.gca().legend(frameon=False)
        annotate(erange=erange)
        #        plt.gca().set_yscale('log')
        plt.savefig(os.path.join(gta.config['fileio']['workdir'],
                                 '%s_counts_map_xproj%s.%s' % (
                                     prefix, esuffix, format)))
        plt.close(fig)

        fig = plt.figure()
        p.plot_projection(1, label='Data', color='k', **data_style)
        p.plot_projection(1, data=model_data, label='Model',
                          noerror=True)
        p.plot_projection(1, data=diffuse_data, label='Diffuse',
                          noerror=True)
        plt.gca().set_ylabel('Counts')
        plt.gca().set_xlabel('LAT Offset [deg]')
        plt.gca().legend(frameon=False)
        annotate(erange=erange)
        #        plt.gca().set_yscale('log')
        plt.savefig(os.path.join(gta.config['fileio']['workdir'],
                                 '%s_counts_map_yproj%s.%s' % (
                                     prefix, esuffix, format)))

        plt.close(fig)

    def make_components_plots(self):

        figx = plt.figure('xproj')
        figy = plt.figure('yproj')

        colors = ['k', 'b', 'g', 'r']
        data_style = {'marker': 's', 'linestyle': 'None'}

        roi_kwargs = {}
        roi_kwargs.setdefault('erange',erange)
        roi_kwargs.setdefault('graticule_radii',self.config['graticule_radii'])
        roi_kwargs.setdefault('cmap',self.config['cmap'])
        roi_kwargs.setdefault('catalogs',self.config['catalogs'])

        
        for i, c in enumerate(gta.components):
                
            fig = plt.figure()
            p = ROIPlotter(mcube_maps[i + 1], roi=gta.roi, **roi_kwargs)

            mcube_data = p.data

            p.plot(cb_label='Counts', zscale='pow', gamma=1. / 3.)
            plt.savefig(os.path.join(gta.config['fileio']['workdir'],
                                     '%s_model_map%s_%02i.%s' % (
                                     prefix, esuffix, i, format)))
            plt.close(fig)

            plt.figure(figx.number)
            p = ROIPlotter(c.counts_map(), roi=gta.roi, **roi_kwargs)
            p.plot_projection(0, color=colors[i % 4], label='Component %i' % i,
                              **data_style)

            p.plot_projection(0, data=mcube_data,
                              color=colors[i % 4], noerror=True,
                              label='__nolegend__')

            plt.figure(figy.number)
            p.plot_projection(1, color=colors[i % 4], label='Component %i' % i,
                              **data_style)
           
            p.plot_projection(1, data=mcube_data,
                              color=colors[i % 4], noerror=True,
                              label='__nolegend__')

        plt.figure(figx.number)
        ROIPlotter.setup_projection_axis(0)
        annotate(erange=erange)
        figx.savefig(os.path.join(gta.config['fileio']['workdir'],
                                  '%s_counts_map_comp_xproj%s.%s' % (
                                      prefix, esuffix, format)))

        plt.figure(figy.number)
        ROIPlotter.setup_projection_axis(1)
        annotate(erange=erange)
        figy.savefig(os.path.join(gta.config['fileio']['workdir'],
                                  '%s_counts_map_comp_yproj%s.%s' % (
                                      prefix, esuffix, format)))
        plt.close(figx)
        plt.close(figy)
        
        
    def make_extension_plots(self, prefix, erange=None, **kwargs):

        format = kwargs.get('format', self.config['plotting']['format'])

        for s in self.roi.sources:

            if 'extension' not in s: 
                continue
            if s['extension'] is None: 
                continue
            if not s['extension']['config']['save_model_map']: 
                continue

            self._plot_extension(prefix, s, erange=erange, format=format)

    def make_sed_plots(self, gta, prefix='', **kwargs):

        format = kwargs.get('format', gta.config['plotting']['format'])

        for s in gta.roi.sources:

            if 'sed' not in s: 
                continue
            if s['sed'] is None: 
                continue

            name = s.name.lower().replace(' ', '_')

            self.logger.debug('Making SED plot for %s' % s.name)

            p = SEDPlotter(s)
            fig = plt.figure()
            p.plot()
            plt.savefig(os.path.join(gta.config['fileio']['workdir'],
                                     '%s_%s_sed.%s' % (prefix, name, format)))
            plt.close(fig)

            p = SEDPlotter(s)
            fig = plt.figure()
            p.plot(showlnl=True)
            plt.savefig(os.path.join(gta.config['fileio']['workdir'],
                                     '%s_%s_sedlnl.%s' % (
                                         prefix, name, format)))
            plt.close(fig)

    def make_sed_plot(self,gta,name,**kwargs):

        prefix = kwargs.get('prefix','')
        src = gta.roi[name]

        name = src.name.lower().replace(' ', '_')
        format = kwargs.get('format', gta.config['plotting']['format'])
        p = SEDPlotter(src)
        fig = plt.figure()
        p.plot()

        outfile = utils.format_filename(gta.config['fileio']['workdir'],
                                        'sed', prefix=[prefix, name],
                                        extension=format)

        plt.savefig(outfile)
        plt.close(fig)

        p = SEDPlotter(src)
        fig = plt.figure()
        p.plot(showlnl=True)

        outfile = utils.format_filename(gta.config['fileio']['workdir'],
                                        'sedlnl', prefix=[prefix, name],
                                        extension=format)
        plt.savefig(outfile)
        plt.close(fig)

    def make_localization_plot(self,gta,name,tsmap,**kwargs):
        
        tsmap_renorm = copy.deepcopy(tsmap['ts'])
        tsmap_renorm._counts -= np.max(tsmap_renorm._counts)
                
        prefix = kwargs.get('prefix','')
        skydir = kwargs.get('skydir',None)
        src = gta.roi[name]
        o = src['localize']
        
        name = src.name.lower().replace(' ', '_')
        format = kwargs.get('format', gta.config['plotting']['format'])

        p = ROIPlotter(tsmap_renorm,roi=gta.roi)
        fig = plt.figure()

        p.plot(levels=[-200,-100,-50,-20,-9.21,-5.99,-2.3,-1.0],
               cmap='BuGn',vmin=-50.0,
               interpolation='bicubic',cb_label='2$\\times\Delta\ln$L')

        cdelt0 = np.abs(tsmap['ts'].wcs.wcs.cdelt[0])
        cdelt1 = np.abs(tsmap['ts'].wcs.wcs.cdelt[1])

        tsmap_fit = o['tsmap_fit']
        
        peak_skydir = SkyCoord(tsmap_fit['glon'],tsmap_fit['glat'],
                               frame='galactic',unit='deg')
        peak_pix = peak_skydir.to_pixel(tsmap_renorm.wcs)
        peak_r68 = tsmap_fit['r68']
        peak_r99 = tsmap_fit['r99']
        
        scan_skydir = SkyCoord(o['glon'],o['glat'],frame='galactic',unit='deg')
        scan_pix = scan_skydir.to_pixel(tsmap_renorm.wcs)

        if skydir is not None:
            pix = skydir.to_pixel(tsmap_renorm.wcs)
            plt.gca().plot(pix[0],pix[1],linestyle='None',
                           marker='+',color='r')
        
        if np.isfinite(float(peak_pix[0])):

            sigma = tsmap_fit['sigma']
            sigmax = tsmap_fit['sigma_semimajor']
            sigmay = tsmap_fit['sigma_semiminor']
            theta = tsmap_fit['theta']
            
            e0 = Ellipse(xy=(float(peak_pix[0]),float(peak_pix[1])),
                         width=2.0*sigmax/cdelt0*peak_r68/sigma,
                         height=2.0*sigmay/cdelt1*peak_r68/sigma,
                         angle=np.degrees(theta),
                         facecolor='None',edgecolor='k')

            e1 = Ellipse(xy=(float(peak_pix[0]),float(peak_pix[1])),
                         width=2.0*sigmax/cdelt0*peak_r99/sigma,
                         height=2.0*sigmay/cdelt1*peak_r99/sigma,
                         angle=np.degrees(theta),
                         facecolor='None',edgecolor='k')

            plt.gca().add_artist(e0)
            plt.gca().add_artist(e1)
            plt.gca().plot(float(peak_pix[0]),float(peak_pix[1]),
                           marker='x',color='k')

        if np.isfinite(float(scan_pix[0])):

            sigmax = o['sigma_semimajor']
            sigmay = o['sigma_semiminor']
            
            e0 = Ellipse(xy=(float(scan_pix[0]),float(scan_pix[1])),
                         width=2.0*sigmax/cdelt0*o['r68']/o['sigma'],
                         height=2.0*sigmay/cdelt1*o['r68']/o['sigma'],
                         angle=np.degrees(o['theta']),
                         facecolor='None',edgecolor='r')

            e1 = Ellipse(xy=(float(scan_pix[0]),float(scan_pix[1])),
                         width=2.0*sigmax/cdelt0*o['r99']/o['sigma'],
                         height=2.0*sigmay/cdelt1*o['r99']/o['sigma'],
                         angle=np.degrees(o['theta']),
                         facecolor='None',edgecolor='r')

            plt.gca().add_artist(e0)
            plt.gca().add_artist(e1)
            
            plt.gca().plot(float(scan_pix[0]),float(scan_pix[1]),
                           marker='x',color='r')
        
#        if gta.config['binning']['coordsys'] == 'GAL':        
#            plt.gca().scatter(o['peak_glon'], o['peak_glat'],
#                              transform=plt.gca().get_transform('galactic'),color='k')
#            plt.gca().scatter(o['glon'], o['glat'],
#                              transform=plt.gca().get_transform('galactic'),color='r')
#        else:
#            plt.gca().scatter(o['peak_ra'], o['peak_dec'],
#                              transform=plt.gca().get_transform('fk5'),color='k')
#            plt.gca().scatter(o['ra'], o['dec'],
#                              transform=plt.gca().get_transform('fk5'),color='r')
       
        outfile = utils.format_filename(gta.config['fileio']['workdir'],
                                        'localize', prefix=[prefix, name],
                                        extension=format)

        plt.savefig(outfile)
        plt.close(fig)
        
    def _plot_extension(self, gta, prefix, src, erange=None, **kwargs):
        """Utility function for generating diagnostic plots for the
        extension analysis."""

        format = kwargs.get('format', self.config['plotting']['format'])

        if erange is None:
            erange = (self.energies[0], self.energies[-1])

        name = src['name'].lower().replace(' ', '_')

        esuffix = '_%.3f_%.3f' % (erange[0], erange[1])

        p = ExtensionPlotter(src, self.roi, '',
                             self.config['fileio']['workdir'],
                             erange=erange)

        fig = plt.figure()
        p.plot(0)
        plt.gca().set_xlim(-2, 2)
        ROIPlotter.setup_projection_axis(0)
        annotate(src=src, erange=erange)
        plt.savefig(os.path.join(self.config['fileio']['workdir'],
                                 '%s_%s_extension_xproj%s.png' % (
                                     prefix, name, esuffix)))
        plt.close(fig)

        fig = plt.figure()
        p.plot(1)
        plt.gca().set_xlim(-2, 2)
        ROIPlotter.setup_projection_axis(1)
        annotate(src=src, erange=erange)
        plt.savefig(os.path.join(self.config['fileio']['workdir'],
                                 '%s_%s_extension_yproj%s.png' % (
                                     prefix, name, esuffix)))
        plt.close(fig)

        for i, c in enumerate(self.components):
            suffix = '_%02i' % i

            p = ExtensionPlotter(src, self.roi, suffix,
                                 self.config['fileio']['workdir'],
                                 erange=erange)

            fig = plt.figure()
            p.plot(0)
            ROIPlotter.setup_projection_axis(0, erange=erange)
            annotate(src=src, erange=erange)
            plt.gca().set_xlim(-2, 2)
            plt.savefig(os.path.join(self.config['fileio']['workdir'],
                                     '%s_%s_extension_xproj%s%s.png' % (
                                         prefix, name, esuffix, suffix)))
            plt.close(fig)

            fig = plt.figure()
            p.plot(1)
            plt.gca().set_xlim(-2, 2)
            ROIPlotter.setup_projection_axis(1, erange=erange)
            annotate(src=src, erange=erange)
            plt.savefig(os.path.join(self.config['fileio']['workdir'],
                                     '%s_%s_extension_yproj%s%s.png' % (
                                         prefix, name, esuffix, suffix)))
            plt.close(fig)
            
            
