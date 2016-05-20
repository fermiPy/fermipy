from __future__ import absolute_import, division, print_function, \
    unicode_literals

import os
import copy
import itertools
import functools
from multiprocessing import Pool, cpu_count

import numpy as np
import warnings

import pyLikelihood as pyLike

import astropy.io.fits as pyfits
from astropy.table import Table
import astropy.wcs as pywcs

import fermipy.config
import fermipy.defaults as defaults
import fermipy.utils as utils
import fermipy.wcs_utils as wcs_utils
import fermipy.fits_utils as fits_utils
import fermipy.plotting as plotting
from fermipy.skymap import Map
from fermipy.roi_model import Source
from fermipy.logger import Logger
from fermipy.logger import logLevel

import fermipy.sed as sed
from fermipy.spectrum import PowerLaw

MAX_NITER = 100

def extract_images_from_tscube(infile,outfile):
    """ Extract data from table HDUs in TSCube file and convert them to FITS images
    """
    inhdulist = pyfits.open(infile)
    wcs = pywcs.WCS(inhdulist[0].header)
    map_shape = inhdulist[0].data.shape

    t_eng = Table.read(infile,"EBOUNDS")
    t_scan = Table.read(infile,"SCANDATA")
    t_fit = Table.read(infile,"FITDATA")

    n_ebin = len(t_eng)
    energies = np.ndarray((n_ebin+1))
    energies[0:-1] = t_eng["E_MIN"]
    energies[-1] = t_eng["E_MAX"][-1]

    cube_shape = (n_ebin,map_shape[1],map_shape[0])

    wcs_cube = wcs_utils.wcs_add_energy_axis(wcs,np.log10(energies))

    outhdulist = [inhdulist[0],inhdulist["EBOUNDS"]]
    
    FIT_COLNAMES = ['FIT_TS','FIT_STATUS','FIT_NORM','FIT_NORM_ERR','FIT_NORM_ERRP','FIT_NORM_ERRN']
    SCAN_COLNAMES = ['TS','BIN_STATUS','NORM','NORM_UL','NORM_ERR','NORM_ERRP','NORM_ERRN','LOGLIKE']

    for c in FIT_COLNAMES:
        data = t_fit[c].data.reshape(map_shape)
        hdu = pyfits.ImageHDU(data,wcs.to_header(),name=c)
        outhdulist.append(hdu)
        pass

    for c in SCAN_COLNAMES:
        data = t_scan[c].data.swapaxes(0,1).reshape(cube_shape)
        hdu = pyfits.ImageHDU(data,wcs_cube.to_header(),name=c)
        outhdulist.append(hdu)
        pass

    hdulist = pyfits.HDUList(outhdulist)
    hdulist.writeto(outfile,clobber=True)
    return hdulist


def convert_tscube(infile,outfile):
    """Convert between old and new TSCube formats."""
    inhdulist = pyfits.open(infile)

    # If already in the new-style format just write and exit
    if 'DLOGLIKE_SCAN' in inhdulist['SCANDATA'].columns.names:
        if infile != outfile:
            hdulist.writeto(outfile,clobber=True)
        return
    
    # Get stuff out of the input file
    nrows = inhdulist['SCANDATA']._nrows
    nebins = inhdulist['EBOUNDS']._nrows
    npts = inhdulist['SCANDATA'].data.field('NORMSCAN').shape[1] / nebins

    emin = inhdulist['EBOUNDS'].data.field('E_MIN')/1E3
    emax = inhdulist['EBOUNDS'].data.field('E_MAX')/1E3
    eref = np.sqrt(emin*emax)
    dfde_emin = inhdulist['EBOUNDS'].data.field('E_MIN_FL')
    dfde_emax = inhdulist['EBOUNDS'].data.field('E_MAX_FL')
    index = np.log(dfde_emin/dfde_emax)/np.log(emin/emax)

    flux = PowerLaw.eval_flux(emin,emax,[dfde_emin,index],emin)
    eflux = PowerLaw.eval_eflux(emin,emax,[dfde_emin,index],emin)
    dfde = PowerLaw.eval_dfde(np.sqrt(emin*emax),[dfde_emin,index],emin)
 
    ts_map = inhdulist['PRIMARY'].data.reshape((nrows))
    ok_map = inhdulist['TSMAP_OK'].data.reshape((nrows))
    n_map = inhdulist['N_MAP'].data.reshape((nrows))
    errp_map = inhdulist['ERRP_MAP'].data.reshape((nrows))
    errn_map = inhdulist['ERRN_MAP'].data.reshape((nrows))
    err_map = np.ndarray((nrows))
    m = errn_map > 0
    err_map[m] = 0.5*(errp_map[m]+errn_map[m])
    err_map[~m] = errp_map[~m]
    ul_map = n_map + 2.0 * errp_map

    ncube = np.rollaxis(inhdulist['N_CUBE'].data,0,3).reshape((nrows,nebins))
    errpcube = np.rollaxis(inhdulist['ERRPCUBE'].data,0,3).reshape((nrows,nebins))
    errncube = np.rollaxis(inhdulist['ERRNCUBE'].data,0,3).reshape((nrows,nebins))
    tscube = np.rollaxis(inhdulist['TSCUBE'].data,0,3).reshape((nrows,nebins))
    nll_cube = np.rollaxis(inhdulist['NLL_CUBE'].data,0,3).reshape((nrows,nebins))
    ok_cube = np.rollaxis(inhdulist['TSCUBE_OK'].data,0,3).reshape((nrows,nebins))

    ul_cube = ncube + 2.0 * errpcube
    m = errncube > 0
    errcube = np.ndarray((nrows,nebins))
    errcube[m] = 0.5*(errpcube[m]+errncube[m])
    errcube[~m] = errpcube[~m]
    
    norm_scan = inhdulist['SCANDATA'].data.field('NORMSCAN').reshape((nrows,npts,nebins)).swapaxes(1,2)
    nll_scan = inhdulist['SCANDATA'].data.field('NLL_SCAN').reshape((nrows,npts,nebins)).swapaxes(1,2)

    # Adjust the "EBOUNDS" hdu
    columns = inhdulist['EBOUNDS'].columns
    columns.add_col(pyfits.Column(name=str('E_REF'),
                                  format='E', array=eref*1E3,
                                  unit='keV'))    
    columns.add_col(pyfits.Column(name=str('REF_FLUX'),
                                  format='D', array=flux,
                                  unit='ph / (cm2 s)'))
    columns.add_col(pyfits.Column(name=str('REF_EFLUX'),
                                  format='D', array=eflux,
                                  unit='MeV / (cm2 s)'))
    columns.add_col(pyfits.Column(name=str('REF_DFDE'),
                                  format='D', array=dfde,
                                  unit='ph / (MeV cm2 s)'))
    
    columns.change_name('E_MIN_FL',str('REF_DFDE_E_MIN'))
    columns.change_unit('REF_DFDE_E_MIN','ph / (MeV cm2 s)')
    columns.change_name('E_MAX_FL',str('REF_DFDE_E_MAX'))
    columns.change_unit('REF_DFDE_E_MAX','ph / (MeV cm2 s)')
    columns.change_name('NPRED',str('REF_NPRED'))
    
    
    hdu_e = pyfits.BinTableHDU.from_columns(columns,name='EBOUNDS')

    # Make the "FITDATA" hdu
    columns = pyfits.ColDefs([])

    columns.add_col(pyfits.Column(name=str('FIT_TS'), format='E', array=ts_map))
    columns.add_col(pyfits.Column(name=str('FIT_STATUS'), format='E', array=ok_map))
    columns.add_col(pyfits.Column(name=str('FIT_NORM'), format='E', array=n_map))
    columns.add_col(pyfits.Column(name=str('FIT_NORM_ERR'), format='E', array=err_map))
    columns.add_col(pyfits.Column(name=str('FIT_NORM_ERRP'), format='E', array=errp_map))
    columns.add_col(pyfits.Column(name=str('FIT_NORM_ERRN'), format='E', array=errn_map))    
    hdu_f = pyfits.BinTableHDU.from_columns(columns,name='FITDATA')
   
    # Make the "SCANDATA" hdu
    columns = pyfits.ColDefs([])
   
    columns.add_col(pyfits.Column(name=str('TS'), format='%iE'%nebins, array=tscube,
                                  dim=str('(%i)'%nebins)))
   
    columns.add_col(pyfits.Column(name=str('BIN_STATUS'), format='%iE'%nebins, array=ok_cube,
                                  dim=str('(%i)'%nebins)))
       
    columns.add_col(pyfits.Column(name=str('NORM'), format='%iE'%nebins, array=ncube,
                                  dim=str('(%i)'%nebins)))

    columns.add_col(pyfits.Column(name=str('NORM_UL'), format='%iE'%nebins, array=ul_cube,
                                  dim=str('(%i)'%nebins)))
    
    columns.add_col(pyfits.Column(name=str('NORM_ERR'), format='%iE'%nebins, array=errcube,
                                  dim=str('(%i)'%nebins)))
    
    columns.add_col(pyfits.Column(name=str('NORM_ERRP'), format='%iE'%nebins, array=errpcube,
                                  dim=str('(%i)'%nebins)))

    columns.add_col(pyfits.Column(name=str('NORM_ERRN'), format='%iE'%nebins, array=errncube,
                                  dim=str('(%i)'%nebins)))

    columns.add_col(pyfits.Column(name=str('LOGLIKE'), format='%iE'%nebins, array=nll_cube,
                                  dim=str('(%i)'%nebins)))

    columns.add_col(pyfits.Column(name=str('NORM_SCAN'), format='%iE'%(nebins*npts), array=norm_scan,
                                  dim=str('(%i,%i)'%(npts,nebins))))

    columns.add_col(pyfits.Column(name=str('DLOGLIKE_SCAN'), format='%iE'%(nebins*npts), array=nll_scan,
                                  dim=str('(%i,%i)'%(npts,nebins))))

       
    hdu_s = pyfits.BinTableHDU.from_columns(columns,name='SCANDATA')

 
    hdulist = pyfits.HDUList([inhdulist[0],
                              hdu_s,
                              hdu_f,
                              inhdulist["BASELINE"],
                              hdu_e])

    hdulist['SCANDATA'].header['UL_CONF'] = 0.95

    hdulist.writeto(outfile,clobber=True)

    return hdulist
    

def overlap_slices(large_array_shape, small_array_shape, position):
    """
    Modified version of `~astropy.nddata.utils.overlap_slices`.

    Get slices for the overlapping part of a small and a large array.

    Given a certain position of the center of the small array, with
    respect to the large array, tuples of slices are returned which can be
    used to extract, add or subtract the small array at the given
    position. This function takes care of the correct behavior at the
    boundaries, where the small array is cut of appropriately.

    Parameters
    ----------
    large_array_shape : tuple
        Shape of the large array.
    small_array_shape : tuple
        Shape of the small array.
    position : tuple
        Position of the small array's center, with respect to the large array.
        Coordinates should be in the same order as the array shape.

    Returns
    -------
    slices_large : tuple of slices
        Slices in all directions for the large array, such that
        ``large_array[slices_large]`` extracts the region of the large array
        that overlaps with the small array.
    slices_small : slice
        Slices in all directions for the small array, such that
        ``small_array[slices_small]`` extracts the region that is inside the
        large array.
    """
    # Get edge coordinates
    edges_min = [int(pos - small_shape // 2) for (pos, small_shape) in
                 zip(position, small_array_shape)]
    edges_max = [int(pos + (small_shape - small_shape // 2)) for
                 (pos, small_shape) in
                 zip(position, small_array_shape)]

    # Set up slices
    slices_large = tuple(slice(max(0, edge_min), min(large_shape, edge_max))
                         for (edge_min, edge_max, large_shape) in
                         zip(edges_min, edges_max, large_array_shape))
    slices_small = tuple(slice(max(0, -edge_min),
                               min(large_shape - edge_min, edge_max - edge_min))
                         for (edge_min, edge_max, large_shape) in
                         zip(edges_min, edges_max, large_array_shape))

    return slices_large, slices_small


def truncate_array(array1, array2, position):
    """Truncate array1 by finding the overlap with array2 when the
    array1 center is located at the given position in array2."""

    slices = []
    for i in range(array1.ndim):
        xmin = 0
        xmax = array1.shape[i]
        dxlo = array1.shape[i] // 2
        dxhi = array1.shape[i] - dxlo
        if position[i] - dxlo < 0:
            xmin = max(dxlo - position[i], 0)

        if position[i] + dxhi > array2.shape[i]:
            xmax = array1.shape[i] - (position[i] + dxhi - array2.shape[i])
            xmax = max(xmax, 0)
        slices += [slice(xmin, xmax)]

    return array1[slices]


def extract_array(array_large, array_small, position):
    shape = array_small.shape
    slices = []
    for i in range(array_large.ndim):

        if shape[i] is None:
            slices += [slice(0, None)]
        else:
            xmin = max(position[i] - shape[i] // 2, 0)
            xmax = min(position[i] + shape[i] // 2, array_large.shape[i])
            slices += [slice(xmin, xmax)]

    return array_large[slices]


def extract_large_array(array_large, array_small, position):
    large_slices, small_slices = overlap_slices(array_large.shape,
                                                array_small.shape, position)
    return array_large[large_slices]


def extract_small_array(array_small, array_large, position):
    large_slices, small_slices = overlap_slices(array_large.shape,
                                                array_small.shape, position)
    return array_small[small_slices]


def _cast_args_to_list(args):
    maxlen = max([len(t) if isinstance(t, list) else 1 for t in args])
    new_args = []
    for i, arg in enumerate(args):
        if not isinstance(arg, list):
            new_args += [[arg] * maxlen]
        else:
            new_args += [arg]

    return new_args


def _sum_wrapper(fn):
    """
    Wrapper to perform row-wise aggregation of list arguments and pass
    them to a function.  The return value of the function is summed
    over the argument groups.  Non-list arguments will be
    automatically cast to a list.
    """

    def wrapper(*args, **kwargs):
        v = 0
        new_args = _cast_args_to_list(args)
        for arg in zip(*new_args): v += fn(*arg, **kwargs)
        return v

    return wrapper


def _collect_wrapper(fn):
    """
    Wrapper for element-wise dispatch of list arguments to a function.
    """

    def wrapper(*args, **kwargs):
        v = []
        new_args = _cast_args_to_list(args)
        for arg in zip(*new_args): v += [fn(*arg, **kwargs)]
        return v

    return wrapper


def _amplitude_bounds(counts, background, model):
    """
    Compute bounds for the root of `_f_cash_root_cython`.

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Count map.
    background : `~numpy.ndarray`
        Background map.
    model : `~numpy.ndarray`
        Source template (multiplied with exposure).
    """

    if isinstance(counts, list):
        counts = np.concatenate([t.flat for t in counts])
        background = np.concatenate([t.flat for t in background])
        model = np.concatenate([t.flat for t in model])

    s_model = np.sum(model)
    s_counts = np.sum(counts)

    sn = background / model
    imin = np.argmin(sn)
    sn_min = sn.flat[imin]
    c_min = counts.flat[imin]

    b_min = c_min / s_model - sn_min
    b_max = s_counts / s_model - sn_min
    return max(b_min, 0), b_max


def _f_cash_root(x, counts, background, model):
    """
    Function to find root of. Described in Appendix A, Stewart (2009).

    Parameters
    ----------
    x : float
        Model amplitude.
    counts : `~numpy.ndarray`
        Count map slice, where model is defined.
    background : `~numpy.ndarray`
        Background map slice, where model is defined.
    model : `~numpy.ndarray`
        Source template (multiplied with exposure).
    """

    return np.sum(model * (counts / (x * model + background) - 1.0))


def _root_amplitude_brentq(counts, background, model, root_fn=_f_cash_root):
    """Fit amplitude by finding roots using Brent algorithm.

    See Appendix A Stewart (2009).

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Slice of count map.
    background : `~numpy.ndarray`
        Slice of background map.
    model : `~numpy.ndarray`
        Model template to fit.

    Returns
    -------
    amplitude : float
        Fitted flux amplitude.
    niter : int
        Number of function evaluations needed for the fit.
    """
    from scipy.optimize import brentq

    # Compute amplitude bounds and assert counts > 0
    amplitude_min, amplitude_max = _amplitude_bounds(counts, background, model)

    if not sum_arrays(counts) > 0:
        return amplitude_min, 0

    args = (counts, background, model)

    if root_fn(0.0, *args) < 0:
        return 0.0, 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = brentq(root_fn, amplitude_min, amplitude_max, args=args,
                            maxiter=MAX_NITER, full_output=True, rtol=1E-4)
            return result[0], result[1].iterations
        except (RuntimeError, ValueError):
            # Where the root finding fails NaN is set as amplitude
            return np.nan, MAX_NITER


def poisson_log_like(counts, model):
    """Compute the Poisson log-likelihood function for the given
    counts and model arrays."""
    return model - counts * np.log(model)


def cash(counts, model):
    """Compute the Poisson log-likelihood function."""
    return 2 * poisson_log_like(counts, model)


def f_cash_sum(x, counts, background, model):
    return np.sum(f_cash(x, counts, background, model))


def f_cash(x, counts, background, model):
    """
    Wrapper for cash statistics, that defines the model function.

    Parameters
    ----------
    x : float
        Model amplitude.
    counts : `~numpy.ndarray`
        Count map slice, where model is defined.
    background : `~numpy.ndarray`
        Background map slice, where model is defined.
    model : `~numpy.ndarray`
        Source template (multiplied with exposure).
    """

    return 2.0 * poisson_log_like(counts, background + x * model)

def sum_arrays(x):
    return sum([t.sum() for t in x])    

def _ts_value(position, counts, background, model, C_0_map, method,logger=None):
    """
    Compute TS value at a given pixel position using the approach described
    in Stewart (2009).

    Parameters
    ----------
    position : tuple
        Pixel position.
    counts : `~numpy.ndarray`
        Count map.
    background : `~numpy.ndarray`
        Background map.
    model : `~numpy.ndarray`
        Source model map.

    Returns
    -------
    TS : float
        TS value at the given pixel position.
    """

    if not isinstance(position,list): position = [position]
    if not isinstance(counts,list): counts = [counts]
    if not isinstance(background,list): background = [background]
    if not isinstance(model,list): model = [model]
    if not isinstance(C_0_map,list): C_0_map = [C_0_map]

    extract_fn = _collect_wrapper(extract_large_array)
    truncate_fn = _collect_wrapper(extract_small_array)

    # Get data slices
    counts_ = extract_fn(counts, model, position)
    background_ = extract_fn(background, model, position)
    C_0_ = extract_fn(C_0_map, model, position)
    model_ = truncate_fn(model, counts, position)

#    C_0 = sum(C_0_).sum()
#    C_0 = _sum_wrapper(sum)(C_0_).sum()
    C_0 = sum_arrays(C_0_)    
    if method == 'root brentq':
        amplitude, niter = _root_amplitude_brentq(counts_, background_, model_,
                                                  root_fn=_sum_wrapper(
                                                      _f_cash_root))
    else:
        raise ValueError('Invalid fitting method.')

    if niter > MAX_NITER:
        #log.warning('Exceeded maximum number of function evaluations!')
        if logger is not None:
            logger.warning('Exceeded maximum number of function evaluations!')
        return np.nan, amplitude, niter

    with np.errstate(invalid='ignore', divide='ignore'):
        C_1 = _sum_wrapper(f_cash_sum)(amplitude, counts_, background_, model_)

    # Compute and return TS value
    return (C_0 - C_1) * np.sign(amplitude), amplitude, niter


class TSMapGenerator(object):
    """Mixin class for `~fermipy.gtanalysis.GTAnalysis` that
    generates TS maps."""

    def tsmap(self, prefix='', **kwargs):
        """Generate a spatial TS map for a source component with
        properties defined by the `model` argument.  The TS map will
        have the same geometry as the ROI.  The output of this method
        is a dictionary containing `~fermipy.skymap.Map` objects with
        the TS and amplitude of the best-fit test source.  By default
        this method will also save maps to FITS files and render them
        as image files.

        This method uses a simplified likelihood fitting
        implementation that only fits for the normalization of the
        test source.  Before running this method it is recommended to
        first optimize the ROI model (e.g. by running
        :py:meth:`~fermipy.gtanalysis.GTAnalysis.optimize`).

        Parameters
        ----------

        prefix : str
           Optional string that will be prepended to all output files
           (FITS and rendered images).

        model : dict
           Dictionary defining the properties of the test source.

        exclude : str or list of str
            Source or sources that will be removed from the model when
            computing the TS map.

        erange : list
           Restrict the analysis to an energy range (emin,emax) in
           log10(E/MeV) that is a subset of the analysis energy range.
           By default the full analysis energy range will be used.  If
           either emin/emax are None then only an upper/lower bound on
           the energy range wil be applied.

        max_kernel_radius : float
           Set the maximum radius of the test source kernel.  Using a
           smaller value will speed up the TS calculation at the loss of
           accuracy.  The default value is 3 degrees.

        make_plots : bool
           Write image files.

        write_fits : bool
           Write a FITS file.
           
        write_npy : bool
           Write a numpy file.
        
        Returns
        -------

        maps : dict
           A dictionary containing the `~fermipy.skymap.Map` objects
           for TS and source amplitude.

        """

        self.logger.info('Generating TS map')

        config = copy.deepcopy(self.config['tsmap'])
        config = utils.merge_dict(config,kwargs,add_new_keys=True)

        # Defining default properties of test source model
        config['model'].setdefault('Index', 2.0)
        config['model'].setdefault('SpectrumType', 'PowerLaw')
        config['model'].setdefault('SpatialModel', 'PointSource')
        config['model'].setdefault('Prefactor', 1E-13)
        
        make_plots = kwargs.get('make_plots', True)
        maps = self._make_tsmap_fast(prefix, config, **kwargs)

        if make_plots:
            plotter = plotting.AnalysisPlotter(self.config['plotting'],
                                               fileio=self.config['fileio'],
                                               logging=self.config['logging'])

            plotter.make_tsmap_plots(self, maps)
            
        self.logger.info('Finished TS map')
        return maps
    
    def _make_tsmap_fast(self, prefix, config, **kwargs):
        """
        Make a TS map from a GTAnalysis instance.  This is a
        simplified implementation optimized for speed that only fits
        for the source normalization (all background components are
        kept fixed). The spectral/spatial characteristics of the test
        source can be defined with the src_dict argument.  By default
        this method will generate a TS map for a point source with an
        index=2.0 power-law spectrum.
        
        Parameters
        ----------
        model : dict or `~fermipy.roi_model.Source` object        
           Dictionary or Source object defining the properties of the
           test source that will be used in the scan.

        """
        
        write_fits = kwargs.get('write_fits', True)
        write_npy = kwargs.get('write_npy', True)
        map_skydir = kwargs.get('map_skydir',None)
        map_size = kwargs.get('map_size',1.0)
        exclude = kwargs.get('exclude', None)
        
        src_dict = copy.deepcopy(config.setdefault('model',{}))
        multithread = config.setdefault('multithread',False)
        threshold = config.setdefault('threshold',1E-2)
        max_kernel_radius = config.get('max_kernel_radius')
        erange = config.setdefault('erange', None)        

        if erange is not None:            
            if len(erange) == 0: erange = [None,None]
            elif len(erange) == 1: erange += [None]            
            erange[0] = (erange[0] if erange[0] is not None 
                         else self.energies[0])
            erange[1] = (erange[1] if erange[1] is not None 
                         else self.energies[-1])
        else:
            erange = [self.energies[0],self.energies[-1]]
        
        # Put the test source at the pixel closest to the ROI center
        xpix, ypix = (np.round((self.npix - 1.0) / 2.),
                      np.round((self.npix - 1.0) / 2.))
        cpix = np.array([xpix, ypix])

        skywcs = self._skywcs
        skydir = wcs_utils.pix_to_skydir(cpix[0], cpix[1], skywcs)

        if src_dict is None:
            src_dict = {}
        src_dict['ra'] = skydir.ra.deg
        src_dict['dec'] = skydir.dec.deg
        src_dict.setdefault('SpatialModel', 'PointSource')
        src_dict.setdefault('SpatialWidth', 0.3)
        src_dict.setdefault('Index', 2.0)
        src_dict.setdefault('Prefactor', 1E-13)

        counts = []
        background = []
        model = []
        c0_map = []
        eslices = []
        enumbins = []
        model_npred = 0
        for c in self.components:

            imin = utils.val_to_edge(c.energies,erange[0])[0]
            imax = utils.val_to_edge(c.energies,erange[1])[0]

            eslice = slice(imin,imax)
            bm = c.model_counts_map(exclude=exclude).counts.astype('float')[eslice,...]
            cm = c.counts_map().counts.astype('float')[eslice,...]
            
            background += [bm]
            counts += [cm]
            c0_map += [cash(cm, bm)]
            eslices += [eslice]
            enumbins += [cm.shape[0]]

        
        self.add_source('tsmap_testsource', src_dict, free=True,
                       init_source=False)
        src = self.roi['tsmap_testsource']
        #self.logger.info(str(src_dict))
        modelname = utils.create_model_name(src)
        for c, eslice in zip(self.components,eslices):            
            mm = c.model_counts_map('tsmap_testsource').counts.astype('float')[eslice,...]            
            model_npred += np.sum(mm)
            model += [mm]
            
        self.delete_source('tsmap_testsource')
        
        for i, mm in enumerate(model):

            dpix = 3
            for j in range(mm.shape[0]):

                ix,iy = np.unravel_index(np.argmax(mm[j,...]),mm[j,...].shape)
                
                mx = mm[j,ix, :] > mm[j,ix,iy] * threshold
                my = mm[j,:, iy] > mm[j,ix,iy] * threshold
                dpix = max(dpix, np.round(np.sum(mx) / 2.))
                dpix = max(dpix, np.round(np.sum(my) / 2.))
                
            if max_kernel_radius is not None and \
                    dpix > int(max_kernel_radius/self.components[i].binsz):
                dpix = int(max_kernel_radius/self.components[i].binsz)

            xslice = slice(max(xpix-dpix,0),min(xpix+dpix+1,self.npix))
            model[i] = model[i][:,xslice,xslice]
            
        ts_values = np.zeros((self.npix, self.npix))
        amp_values = np.zeros((self.npix, self.npix))
        
        wrap = functools.partial(_ts_value, counts=counts, 
                                 background=background, model=model,
                                 C_0_map=c0_map, method='root brentq')

        if map_skydir is not None:
            map_offset = wcs_utils.skydir_to_pix(map_skydir, self._skywcs)
            map_delta = 0.5*map_size/self.components[0].binsz
            xmin = max(int(np.ceil(map_offset[1]-map_delta)),0)
            xmax = min(int(np.floor(map_offset[1]+map_delta))+1,self.npix)
            ymin = max(int(np.ceil(map_offset[0]-map_delta)),0)
            ymax = min(int(np.floor(map_offset[0]+map_delta))+1,self.npix)

            xslice = slice(xmin,xmax)
            yslice = slice(ymin,ymax)
            xyrange = [range(xmin,xmax), range(ymin,ymax)]
            
            map_wcs = skywcs.deepcopy()
            map_wcs.wcs.crpix[0] -= ymin
            map_wcs.wcs.crpix[1] -= xmin
        else:
            xyrange = [range(self.npix),range(self.npix)]
            map_wcs = skywcs

            xslice = slice(0,self.npix)
            yslice = slice(0,self.npix)
            
        positions = []
        for i,j in itertools.product(xyrange[0],xyrange[1]):
            p = [[k//2,i,j] for k in enumbins]
            positions += [p]

        if multithread:            
            pool = Pool()
            results = pool.map(wrap,positions)
            pool.close()
            pool.join()
        else:
            results = map(wrap,positions)

        for i, r in enumerate(results):
            ix = positions[i][0][1]
            iy = positions[i][0][2]
            ts_values[ix, iy] = r[0]
            amp_values[ix, iy] = r[1]

        ts_values = ts_values[xslice,yslice]
        amp_values = amp_values[xslice,yslice]
            
        ts_map = Map(ts_values, map_wcs)
        sqrt_ts_map = Map(ts_values**0.5, map_wcs)
        npred_map = Map(amp_values*model_npred, map_wcs)
        amp_map = Map(amp_values*src.get_norm(), map_wcs)

        o = {'name': '%s_%s' % (prefix, modelname),
             'src_dict': copy.deepcopy(src_dict),
             'file': None,
             'ts': ts_map,
             'sqrt_ts': sqrt_ts_map,
             'npred': npred_map,
             'amplitude': amp_map,
             'config' : config
             }

        fits_file = utils.format_filename(self.config['fileio']['workdir'],
                                          'tsmap.fits',
                                          prefix=[prefix,modelname])
        
        if write_fits:
                        
            fits_utils.write_maps(ts_map,
                             {'SQRT_TS_MAP': sqrt_ts_map,
                              'NPRED_MAP': npred_map,
                              'N_MAP': amp_map },
                             fits_file)
            o['file'] = os.path.basename(fits_file)            

        if write_npy:
            np.save(os.path.splitext(fits_file)[0] + '.npy', o)
            
        return o

    def _tsmap_pylike(self, prefix, **kwargs):
        """Evaluate the TS for an additional source component at each point
        in the ROI.  This is the brute force implementation of TS map
        generation that runs a full pyLikelihood fit
        at each point in the ROI."""

        logLike0 = -self.like()
        self.logger.info('LogLike: %f' % logLike0)

        saved_state = LikelihoodState(self.like)

        # Get the ROI geometry

        # Loop over pixels
        w = copy.deepcopy(self._skywcs)
        #        w = create_wcs(self._roi.skydir,cdelt=self._binsz,crpix=50.5)

        data = np.zeros((self.npix, self.npix))
        #        self.free_sources(free=False)

        xpix = np.linspace(0, self.npix - 1, self.npix)[:,
               np.newaxis] * np.ones(data.shape)
        ypix = np.linspace(0, self.npix - 1, self.npix)[np.newaxis,
               :] * np.ones(data.shape)

        radec = wcs_utils.pix_to_skydir(xpix, ypix, w)
        radec = (np.ravel(radec.ra.deg), np.ravel(radec.dec.deg))

        testsource_dict = {
            'ra': radec[0][0],
            'dec': radec[1][0],
            'SpectrumType': 'PowerLaw',
            'Index': 2.0,
            'Scale': 1000,
            'Prefactor': {'value': 0.0, 'scale': 1e-13},
            'SpatialModel': 'PSFSource',
        }

        #        src = self.roi.get_source_by_name('tsmap_testsource')

        for i, (ra, dec) in enumerate(zip(radec[0], radec[1])):
            testsource_dict['ra'] = ra
            testsource_dict['dec'] = dec
            #                        src.set_position([ra,dec])
            self.add_source('tsmap_testsource', testsource_dict, free=True,
                            init_source=False,save_source_maps=False)

            #            for c in self.components:
            #                c.update_srcmap_file([src],True)

            self.set_parameter('tsmap_testsource', 'Prefactor', 0.0,
                               update_source=False)
            self.fit(loglevel=logging.DEBUG,update=False)

            logLike1 = -self.like()
            ts = max(0, 2 * (logLike1 - logLike0))

            data.flat[i] = ts

            #            print i, ra, dec, ts
            #            print self.like()
            #            print self.components[0].like.model['tsmap_testsource']

            self.delete_source('tsmap_testsource')

        saved_state.restore()

        outfile = os.path.join(self.config['fileio']['workdir'], 'tsmap.fits')
        utils.write_fits_image(data, w, outfile)
    
class TSCubeGenerator(object):

    def tscube(self,  prefix='', **kwargs):
        """Generate a spatial TS map for a source component with
        properties defined by the `model` argument.  This method uses
        the `gttscube` ST application for source fitting and will
        simultaneously fit the test source normalization as well as
        the normalizations of any background components that are
        currently free.  The output of this method is a dictionary
        containing `~fermipy.skymap.Map` objects with the TS and
        amplitude of the best-fit test source.  By default this method
        will also save maps to FITS files and render them as image
        files.

        Parameters
        ----------

        prefix : str
           Optional string that will be prepended to all output files
           (FITS and rendered images).

        model : dict
           Dictionary defining the properties of the test source.

        do_sed : bool
           Compute the energy bin-by-bin fits.
        
        nnorm : int
           Number of points in the likelihood v. normalization scan.

        norm_sigma : float
           Number of sigma to use for the scan range.
        
        tol : float        
           Critetia for fit convergence (estimated vertical distance
           to min < tol ).
        
        tol_type : int
           Absoulte (0) or relative (1) criteria for convergence.
        
        max_iter : int
           Maximum number of iterations for the Newton's method fitter
        
        remake_test_source : bool
           If true, recomputes the test source image (otherwise just shifts it)
        
        st_scan_level : int
           
        make_plots : bool
           Write image files.

        write_fits : bool
           Write a FITS file with the results of the analysis.       

        Returns
        -------
        
        maps : dict
           A dictionary containing the `~fermipy.skymap.Map` objects
           for TS and source amplitude.

        """

        self.logger.info('Generating TS cube')

        config = copy.deepcopy(self.config['tscube'])
        config = utils.merge_dict(config,kwargs,add_new_keys=True)
        
        make_plots = kwargs.get('make_plots', True)
        maps = self._make_ts_cube(prefix, config, **kwargs)

        if make_plots:
            plotter = plotting.AnalysisPlotter(self.config['plotting'],
                                               fileio=self.config['fileio'],
                                               logging=self.config['logging'])
            
            plotter.make_tsmap_plots(self, maps, suffix='tscube')
            
        self.logger.info("Finished TS cube")
        return maps
        
    def _make_ts_cube(self, prefix, config, **kwargs):

        write_fits = kwargs.get('write_fits', True)
        skywcs = kwargs.get('wcs',self._skywcs)
        npix = kwargs.get('npix',self.npix)
        
        galactic = wcs_utils.is_galactic(skywcs)
        ref_skydir = wcs_utils.wcs_to_skydir(skywcs)
        refdir = pyLike.SkyDir(ref_skydir.ra.deg,
                               ref_skydir.dec.deg)
        pixsize = np.abs(skywcs.wcs.cdelt[0])
        
        skyproj = pyLike.FitScanner.buildSkyProj(str("AIT"),
                                                 refdir, pixsize, npix,
                                                 galactic)


        src_dict = copy.deepcopy(config.setdefault('model',{}))
        if src_dict is None:
            src_dict = {}

        xpix, ypix = (np.round((self.npix - 1.0) / 2.),
                      np.round((self.npix - 1.0) / 2.))
        skydir = wcs_utils.pix_to_skydir(xpix, ypix, skywcs)
            
        src_dict['ra'] = skydir.ra.deg
        src_dict['dec'] = skydir.dec.deg
        src_dict.setdefault('SpatialModel', 'PointSource')
        src_dict.setdefault('SpatialWidth', 0.3)
        src_dict.setdefault('Index', 2.0)
        src_dict.setdefault('Prefactor', 1E-13)
        src_dict['name'] = 'tscube_testsource'
        
        src = Source.create_from_dict(src_dict)
        
        modelname = utils.create_model_name(src)

        optFactory = pyLike.OptimizerFactory_instance()        
        optObject = optFactory.create(str("MINUIT"),
                                      self.components[0].like.logLike)

        pylike_src = self.components[0]._create_source(src)
        fitScanner = pyLike.FitScanner(self.like.composite, optObject, skyproj,
                                       npix, npix)

        pylike_src.spectrum().normPar().setBounds(0,1E6)
        
        fitScanner.setTestSource(pylike_src)
        
        self.logger.info("Running tscube")
        outfile = utils.format_filename(self.config['fileio']['workdir'],
                                        'tscube.fits',
                                        prefix=[prefix])

        # doSED         : Compute the energy bin-by-bin fits
        # nNorm         : Number of points in the likelihood v. normalization scan
        # covScale_bb   : Scale factor to apply to global fitting cov. matrix 
        #                 in broadband fits ( < 0 -> no prior )
        # covScale      : Scale factor to apply to broadband fitting cov.
        #                 matrix in bin-by-bin fits ( < 0 -> fixed )
        # normSigma     : Number of sigma to use for the scan range 
        # tol           : Critetia for fit convergence (estimated vertical distance to min < tol )
        # maxIter       : Maximum number of iterations for the Newton's method fitter
        # tolType       : Absoulte (0) or relative (1) criteria for convergence
        # remakeTestSource : If true, recomputes the test source image (otherwise just shifts it)
        # ST_scan_level : Level to which to do ST-based fitting (for testing)
        fitScanner.run_tscube(True,
                              config['do_sed'], config['nnorm'],
                              config['norm_sigma'], 
                              config['cov_scale_bb'],config['cov_scale'],
                              config['tol'], config['max_iter'],
                              config['tol_type'], config['remake_test_source'],
                              config['st_scan_level'])
        self.logger.info("Writing FITS output")
                                        
        fitScanner.writeFitsFile(str(outfile), str("gttscube"))

        convert_tscube(str(outfile),str(outfile))
        
        tscube = sed.TSCube.create_from_fits(outfile)
        ts_map = tscube.tsmap        
        norm_map = tscube.normmap
        npred_map = copy.deepcopy(norm_map)
        npred_map._counts *= tscube.specData.npred.sum()
        amp_map = copy.deepcopy(norm_map) 
        amp_map._counts *= src_dict['Prefactor']

        sqrt_ts_map = copy.deepcopy(ts_map)
        sqrt_ts_map._counts = np.abs(sqrt_ts_map._counts)**0.5

        
        o = {'name': '%s_%s' % (prefix, modelname),
             'src_dict': copy.deepcopy(src_dict),
             'file': os.path.basename(outfile),
             'ts': ts_map,
             'sqrt_ts': sqrt_ts_map,
             'npred': npred_map,
             'amplitude': amp_map,
             'config' : config,
             'tscube' : tscube
             }
       

        self.logger.info("Done")
        return o
