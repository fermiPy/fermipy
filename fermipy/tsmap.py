# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import os
import copy
import logging
import itertools
import functools
import json
from multiprocessing import Pool
import numpy as np
import warnings
import pyLikelihood as pyLike
from scipy.optimize import brentq
import astropy
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.wcs as pywcs
from gammapy.maps.geom import coordsys_to_frame
from gammapy.maps import WcsNDMap, WcsGeom
import fermipy.utils as utils
import fermipy.wcs_utils as wcs_utils
import fermipy.fits_utils as fits_utils
import fermipy.plotting as plotting
import fermipy.castro as castro
from fermipy.roi_model import Source
from fermipy.spectrum import PowerLaw
from fermipy.config import ConfigSchema
from fermipy.timing import Timer
from LikelihoodState import LikelihoodState

MAX_NITER = 100


def extract_images_from_tscube(infile, outfile):
    """ Extract data from table HDUs in TSCube file and convert them to FITS images
    """
    inhdulist = fits.open(infile)
    wcs = pywcs.WCS(inhdulist[0].header)
    map_shape = inhdulist[0].data.shape

    t_eng = Table.read(infile, "EBOUNDS")
    t_scan = Table.read(infile, "SCANDATA")
    t_fit = Table.read(infile, "FITDATA")

    n_ebin = len(t_eng)
    energies = np.ndarray((n_ebin + 1))
    energies[0:-1] = t_eng["E_MIN"]
    energies[-1] = t_eng["E_MAX"][-1]

    cube_shape = (n_ebin, map_shape[1], map_shape[0])

    wcs_cube = wcs_utils.wcs_add_energy_axis(wcs, energies)

    outhdulist = [inhdulist[0], inhdulist["EBOUNDS"]]

    FIT_COLNAMES = ['FIT_TS', 'FIT_STATUS', 'FIT_NORM',
                    'FIT_NORM_ERR', 'FIT_NORM_ERRP', 'FIT_NORM_ERRN']
    SCAN_COLNAMES = ['TS', 'BIN_STATUS', 'NORM', 'NORM_UL',
                     'NORM_ERR', 'NORM_ERRP', 'NORM_ERRN', 'LOGLIKE']

    for c in FIT_COLNAMES:
        data = t_fit[c].data.reshape(map_shape)
        hdu = fits.ImageHDU(data, wcs.to_header(), name=c)
        outhdulist.append(hdu)

    for c in SCAN_COLNAMES:
        data = t_scan[c].data.swapaxes(0, 1).reshape(cube_shape)
        hdu = fits.ImageHDU(data, wcs_cube.to_header(), name=c)
        outhdulist.append(hdu)

    hdulist = fits.HDUList(outhdulist)
    hdulist.writeto(outfile, clobber=True)
    return hdulist


def convert_tscube(infile, outfile):

    inhdulist = fits.open(infile)
    if 'dloglike_scan' in inhdulist['SCANDATA'].columns.names:
        if infile != outfile:
            inhdulist.writeto(outfile, clobber=True)
        return
    elif 'E_MIN_FL' in inhdulist['EBOUNDS'].columns.names:
        return convert_tscube_old(infile, outfile)

    for hdu in inhdulist:

        if not isinstance(hdu, fits.BinTableHDU):
            continue

        for col in hdu.columns:

            if hdu.name == 'EBOUNDS':
                col.name = col.name.replace('DFDE', 'DNDE')
            else:
                colname = col.name.lower()
                col.name = colname.replace('dfde', 'dnde')

    inhdulist.writeto(outfile, clobber=True)
    return inhdulist


def convert_tscube_old(infile, outfile):
    """Convert between old and new TSCube formats."""
    inhdulist = fits.open(infile)

    # If already in the new-style format just write and exit
    if 'DLOGLIKE_SCAN' in inhdulist['SCANDATA'].columns.names:
        if infile != outfile:
            inhdulist.writeto(outfile, clobber=True)
        return

    # Get stuff out of the input file
    nrows = inhdulist['SCANDATA']._nrows
    nebins = inhdulist['EBOUNDS']._nrows
    npts = inhdulist['SCANDATA'].data.field('NORMSCAN').shape[1] / nebins

    emin = inhdulist['EBOUNDS'].data.field('e_min') / 1E3
    emax = inhdulist['EBOUNDS'].data.field('e_max') / 1E3
    eref = np.sqrt(emin * emax)
    dnde_emin = inhdulist['EBOUNDS'].data.field('E_MIN_FL')
    dnde_emax = inhdulist['EBOUNDS'].data.field('E_MAX_FL')
    index = np.log(dnde_emin / dnde_emax) / np.log(emin / emax)

    flux = PowerLaw.eval_flux(emin, emax, [dnde_emin, index], emin)
    eflux = PowerLaw.eval_eflux(emin, emax, [dnde_emin, index], emin)
    dnde = PowerLaw.eval_dnde(np.sqrt(emin * emax), [dnde_emin, index], emin)

    ts_map = inhdulist['PRIMARY'].data.reshape((nrows))
    ok_map = inhdulist['TSMAP_OK'].data.reshape((nrows))
    n_map = inhdulist['N_MAP'].data.reshape((nrows))
    errp_map = inhdulist['ERRP_MAP'].data.reshape((nrows))
    errn_map = inhdulist['ERRN_MAP'].data.reshape((nrows))
    err_map = np.ndarray((nrows))
    m = errn_map > 0
    err_map[m] = 0.5 * (errp_map[m] + errn_map[m])
    err_map[~m] = errp_map[~m]
    ul_map = n_map + 2.0 * errp_map

    ncube = np.rollaxis(inhdulist['N_CUBE'].data,
                        0, 3).reshape((nrows, nebins))
    errpcube = np.rollaxis(
        inhdulist['ERRPCUBE'].data, 0, 3).reshape((nrows, nebins))
    errncube = np.rollaxis(
        inhdulist['ERRNCUBE'].data, 0, 3).reshape((nrows, nebins))
    tscube = np.rollaxis(inhdulist['TSCUBE'].data,
                         0, 3).reshape((nrows, nebins))
    nll_cube = np.rollaxis(
        inhdulist['NLL_CUBE'].data, 0, 3).reshape((nrows, nebins))
    ok_cube = np.rollaxis(
        inhdulist['TSCUBE_OK'].data, 0, 3).reshape((nrows, nebins))

    ul_cube = ncube + 2.0 * errpcube
    m = errncube > 0
    errcube = np.ndarray((nrows, nebins))
    errcube[m] = 0.5 * (errpcube[m] + errncube[m])
    errcube[~m] = errpcube[~m]

    norm_scan = inhdulist['SCANDATA'].data.field(
        'NORMSCAN').reshape((nrows, npts, nebins)).swapaxes(1, 2)
    nll_scan = inhdulist['SCANDATA'].data.field(
        'NLL_SCAN').reshape((nrows, npts, nebins)).swapaxes(1, 2)

    # Adjust the "EBOUNDS" hdu
    columns = inhdulist['EBOUNDS'].columns
    columns.add_col(fits.Column(name=str('e_ref'),
                                format='E', array=eref * 1E3,
                                unit='keV'))
    columns.add_col(fits.Column(name=str('ref_flux'),
                                format='D', array=flux,
                                unit='ph / (cm2 s)'))
    columns.add_col(fits.Column(name=str('ref_eflux'),
                                format='D', array=eflux,
                                unit='MeV / (cm2 s)'))
    columns.add_col(fits.Column(name=str('ref_dnde'),
                                format='D', array=dnde,
                                unit='ph / (MeV cm2 s)'))

    columns.change_name('E_MIN_FL', str('ref_dnde_e_min'))
    columns.change_unit('ref_dnde_e_min', 'ph / (MeV cm2 s)')
    columns.change_name('E_MAX_FL', str('ref_dnde_e_max'))
    columns.change_unit('ref_dnde_e_max', 'ph / (MeV cm2 s)')
    columns.change_name('NPRED', str('ref_npred'))

    hdu_e = fits.BinTableHDU.from_columns(columns, name='EBOUNDS')

    # Make the "FITDATA" hdu
    columns = fits.ColDefs([])

    columns.add_col(fits.Column(
        name=str('fit_ts'), format='E', array=ts_map))
    columns.add_col(fits.Column(
        name=str('fit_status'), format='E', array=ok_map))
    columns.add_col(fits.Column(
        name=str('fit_norm'), format='E', array=n_map))
    columns.add_col(fits.Column(
        name=str('fit_norm_err'), format='E', array=err_map))
    columns.add_col(fits.Column(
        name=str('fit_norm_errp'), format='E', array=errp_map))
    columns.add_col(fits.Column(
        name=str('fit_norm_errn'), format='E', array=errn_map))
    hdu_f = fits.BinTableHDU.from_columns(columns, name='FITDATA')

    # Make the "SCANDATA" hdu
    columns = fits.ColDefs([])

    columns.add_col(fits.Column(name=str('ts'),
                                format='%iE' % nebins, array=tscube,
                                dim=str('(%i)' % nebins)))

    columns.add_col(fits.Column(name=str('bin_status'),
                                format='%iE' % nebins, array=ok_cube,
                                dim=str('(%i)' % nebins)))

    columns.add_col(fits.Column(name=str('norm'),
                                format='%iE' % nebins, array=ncube,
                                dim=str('(%i)' % nebins)))

    columns.add_col(fits.Column(name=str('norm_ul'),
                                format='%iE' % nebins, array=ul_cube,
                                dim=str('(%i)' % nebins)))

    columns.add_col(fits.Column(name=str('norm_err'),
                                format='%iE' % nebins, array=errcube,
                                dim=str('(%i)' % nebins)))

    columns.add_col(fits.Column(name=str('norm_errp'),
                                format='%iE' % nebins, array=errpcube,
                                dim=str('(%i)' % nebins)))

    columns.add_col(fits.Column(name=str('norm_errn'),
                                format='%iE' % nebins, array=errncube,
                                dim=str('(%i)' % nebins)))

    columns.add_col(fits.Column(name=str('loglike'),
                                format='%iE' % nebins, array=nll_cube,
                                dim=str('(%i)' % nebins)))

    columns.add_col(fits.Column(name=str('norm_scan'),
                                format='%iE' % (nebins * npts),
                                array=norm_scan,
                                dim=str('(%i,%i)' % (npts, nebins))))

    columns.add_col(fits.Column(name=str('dloglike_scan'),
                                format='%iE' % (nebins * npts),
                                array=nll_scan,
                                dim=str('(%i,%i)' % (npts, nebins))))

    hdu_s = fits.BinTableHDU.from_columns(columns, name='SCANDATA')

    hdulist = fits.HDUList([inhdulist[0],
                            hdu_s,
                            hdu_f,
                            inhdulist["BASELINE"],
                            hdu_e])

    hdulist['SCANDATA'].header['UL_CONF'] = 0.95

    hdulist.writeto(outfile, clobber=True)

    return hdulist


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
    large_slices, small_slices = utils.overlap_slices(array_large.shape,
                                                      array_small.shape, position)
    return array_large[large_slices]


def extract_small_array(array_small, array_large, position):
    large_slices, small_slices = utils.overlap_slices(array_large.shape,
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
        for arg in zip(*new_args):
            v += fn(*arg, **kwargs)
        return v

    return wrapper


def _collect_wrapper(fn):
    """
    Wrapper for element-wise dispatch of list arguments to a function.
    """

    def wrapper(*args, **kwargs):
        v = []
        new_args = _cast_args_to_list(args)
        for arg in zip(*new_args):
            v += [fn(*arg, **kwargs)]
        return v

    return wrapper


def _amplitude_bounds(counts, bkg, model):
    """
    Compute bounds for the root of `_f_cash_root_cython`.

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Count map.
    bkg : `~numpy.ndarray`
        Background map.
    model : `~numpy.ndarray`
        Source template (multiplied with exposure).
    """

    if isinstance(counts, list):
        counts = np.concatenate([t.flat for t in counts])
        bkg = np.concatenate([t.flat for t in bkg])
        model = np.concatenate([t.flat for t in model])

    s_model = np.sum(model)
    s_counts = np.sum(counts)

    sn = bkg / model
    imin = np.argmin(sn)
    sn_min = sn[imin]
    c_min = counts[imin]

    b_min = c_min / s_model - sn_min
    b_max = s_counts / s_model - sn_min
    return max(b_min, 0), b_max


def _f_cash_root(x, counts, bkg, model):
    """
    Function to find root of. Described in Appendix A, Stewart (2009).

    Parameters
    ----------
    x : float
        Model amplitude.
    counts : `~numpy.ndarray`
        Count map slice, where model is defined.
    bkg : `~numpy.ndarray`
        Background map slice, where model is defined.
    model : `~numpy.ndarray`
        Source template (multiplied with exposure).
    """
    return np.sum(model * (counts / (x * model + bkg) - 1.0))


def _fit_amplitude_newton(counts, bkg, model, msum=0, tol=1E-4):

    norm = 0.0

    if isinstance(counts, list):
        counts = np.concatenate([t.flat for t in counts])
        bkg = np.concatenate([t.flat for t in bkg])
        model = np.concatenate([t.flat for t in model])

    m = counts > 0
    if np.any(~m):
        msum += np.sum(model[~m])
        counts = counts[m]
        bkg = bkg[m]
        model = model[m]

    for iiter in range(1, MAX_NITER):

        fdiff = (1.0 - (counts / (bkg + norm * model)))
        grad = np.sum(fdiff * model) + msum

        if (iiter == 1 and grad > 0):
            break

        w2 = counts / (bkg + norm * model)**2
        hess = np.sum(w2 * model**2)
        delta = grad / hess
        edm = delta * grad

        norm = max(0, norm - delta)

        if edm < tol:
            break

    return norm, iiter


def _root_amplitude_brentq(counts, bkg, model, root_fn=_f_cash_root):
    """Fit amplitude by finding roots using Brent algorithm.

    See Appendix A Stewart (2009).

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Slice of count map.
    bkg : `~numpy.ndarray`
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

    # Compute amplitude bounds and assert counts > 0
    amplitude_min, amplitude_max = _amplitude_bounds(counts, bkg, model)

    if not np.sum(counts) > 0:
        return amplitude_min, 0

    args = (counts, bkg, model)

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
    loglike = np.array(model)
    m = counts > 0
    loglike[m] -= counts[m] * np.log(model[m])
    return loglike


def cash(counts, model):
    """Compute the Poisson log-likelihood function."""
    return 2 * poisson_log_like(counts, model)


def f_cash_sum(x, counts, bkg, model, bkg_sum=0, model_sum=0):
    return np.sum(f_cash(x, counts, bkg, model)) + 2.0 * (bkg_sum + x * model_sum)


def f_cash(x, counts, bkg, model):
    """
    Wrapper for cash statistics, that defines the model function.

    Parameters
    ----------
    x : float
        Model amplitude.
    counts : `~numpy.ndarray`
        Count map slice, where model is defined.
    bkg : `~numpy.ndarray`
        Background map slice, where model is defined.
    model : `~numpy.ndarray`
        Source template (multiplied with exposure).
    """

    return 2.0 * poisson_log_like(counts, bkg + x * model)


def _ts_value(position, counts, bkg, model, C_0_map):
    """
    Compute TS value at a given pixel position using the approach described
    in Stewart (2009).

    Parameters
    ----------
    position : tuple
        Pixel position.
    counts : `~numpy.ndarray`
        Count map.
    bkg : `~numpy.ndarray`
        Background map.
    model : `~numpy.ndarray`
        Source model map.

    Returns
    -------
    TS : float
        TS value at the given pixel position.
    """
    extract_fn = _collect_wrapper(extract_large_array)
    truncate_fn = _collect_wrapper(extract_small_array)

    # Get data slices
    counts_slice = extract_fn(counts, model, position)
    bkg_slice = extract_fn(bkg, model, position)
    C_0_slice = extract_fn(C_0_map, model, position)
    model_slice = truncate_fn(model, counts, position)

    # Flattened Arrays
    counts_ = np.concatenate([t.flat for t in counts_slice])
    bkg_ = np.concatenate([t.flat for t in bkg_slice])
    model_ = np.concatenate([t.flat for t in model_slice])
    C_0_ = np.concatenate([t.flat for t in C_0_slice])
    C_0 = np.sum(C_0_)

    root_fn = _sum_wrapper(_f_cash_root)
    amplitude, niter = _root_amplitude_brentq(counts_, bkg_, model_,
                                              root_fn=_f_cash_root)

    if niter > MAX_NITER:
        print('Exceeded maximum number of function evaluations!')
        return np.nan, amplitude, niter

    with np.errstate(invalid='ignore', divide='ignore'):
        C_1 = f_cash_sum(amplitude, counts_, bkg_, model_)

    # Compute and return TS value
    return (C_0 - C_1) * np.sign(amplitude), amplitude, niter


def _ts_value_newton(position, counts, bkg, model, C_0_map):
    """
    Compute TS value at a given pixel position using the newton
    method.

    Parameters
    ----------
    position : tuple
        Pixel position.

    counts : `~numpy.ndarray`
        Count map.

    bkg : `~numpy.ndarray`
        Background map.

    model : `~numpy.ndarray`
        Source model map.

    Returns
    -------
    TS : float
        TS value at the given pixel position.

    amp : float
        Best-fit amplitude of the test source.

    niter : int
        Number of fit iterations.
    """
    extract_fn = _collect_wrapper(extract_large_array)
    truncate_fn = _collect_wrapper(extract_small_array)

    # Get data slices
    counts_slice = extract_fn(counts, model, position)
    bkg_slice = extract_fn(bkg, model, position)
    C_0_map_slice = extract_fn(C_0_map, model, position)
    model_slice = truncate_fn(model, counts, position)

    # Mask of pixels with > 0 counts
    mask = [c > 0 for c in counts_slice]

    # Sum of background and model in empty pixels
    bkg_sum = np.sum(np.array([np.sum(t[~m])
                               for t, m in zip(bkg_slice, mask)]))
    model_sum = np.sum(np.array([np.sum(t[~m])
                                 for t, m in zip(model_slice, mask)]))

    # Flattened Arrays
    counts_ = np.concatenate([t[m].flat for t, m in zip(counts_slice, mask)])
    bkg_ = np.concatenate([t[m].flat for t, m in zip(bkg_slice, mask)])
    model_ = np.concatenate([t[m].flat for t, m in zip(model_slice, mask)])
    C_0 = np.sum(np.array([np.sum(t) for t in C_0_map_slice]))

    amplitude, niter = _fit_amplitude_newton(counts_, bkg_, model_,
                                             model_sum)

    if niter > MAX_NITER:
        print('Exceeded maximum number of function evaluations!')
        return np.nan, amplitude, niter

    with np.errstate(invalid='ignore', divide='ignore'):
        C_1 = f_cash_sum(amplitude, counts_, bkg_, model_, bkg_sum, model_sum)

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
           Optional string that will be prepended to all output files.

        {options}

        Returns
        -------
        tsmap : dict
           A dictionary containing the `~fermipy.skymap.Map` objects
           for TS and source amplitude.

        """
        timer = Timer.create(start=True)

        schema = ConfigSchema(self.defaults['tsmap'])
        schema.add_option('loglevel', logging.INFO)
        schema.add_option('map_skydir', None, '', astropy.coordinates.SkyCoord)
        schema.add_option('map_size', 1.0)
        schema.add_option('threshold', 1E-2, '', float)
        schema.add_option('use_pylike', True, '', bool)
        schema.add_option('outfile', None, '', str)
        config = schema.create_config(self.config['tsmap'], **kwargs)

        # Defining default properties of test source model
        config['model'].setdefault('Index', 2.0)
        config['model'].setdefault('SpectrumType', 'PowerLaw')
        config['model'].setdefault('SpatialModel', 'PointSource')

        self.logger.log(config['loglevel'], 'Generating TS map')

        o = self._make_tsmap_fast(prefix, **config)

        if config['make_plots']:
            plotter = plotting.AnalysisPlotter(self.config['plotting'],
                                               fileio=self.config['fileio'],
                                               logging=self.config['logging'])

            plotter.make_tsmap_plots(o, self.roi)

        self.logger.log(config['loglevel'], 'Finished TS map')

        outfile = config.get('outfile', None)
        if outfile is None:
            outfile = utils.format_filename(self.workdir, 'tsmap',
                                            prefix=[o['name']])
        else:
            outfile = os.path.join(self.workdir,
                                   os.path.splitext(outfile)[0])

        if config['write_fits']:
            o['file'] = os.path.basename(outfile) + '.fits'
            self._make_tsmap_fits(o, outfile + '.fits')

        if config['write_npy']:
            np.save(outfile + '.npy', o)

        self.logger.log(config['loglevel'],
                        'Execution time: %.2f s', timer.elapsed_time)
        return o

    def _make_tsmap_fits(self, data, filename, **kwargs):

        maps = {'SQRT_TS_MAP': data['sqrt_ts'],
                'NPRED_MAP': data['npred'],
                'N_MAP': data['amplitude']}

        hdu_images = []
        for k, v in sorted(maps.items()):
            if v is None:
                continue
            hdu_images += [v.make_hdu(hdu=k)]

        tab = fits_utils.dict_to_table(data)
        hdu_data = fits.table_to_hdu(tab)
        hdu_data.name = 'TSMAP_DATA'

        hdus = [data['ts'].make_hdu(hdu='PRIMARY'),
                hdu_data] + hdu_images

        data['config'].pop('map_skydir', None)
        hdus[0].header['CONFIG'] = json.dumps(data['config'])
        hdus[1].header['CONFIG'] = json.dumps(data['config'])
        fits_utils.write_hdus(hdus, filename)

    def _make_tsmap_fast(self, prefix, **kwargs):
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
        model : dict or `~fermipy.roi_model.Source`
           Dictionary or Source object defining the properties of the
           test source that will be used in the scan.

        """
        loglevel = kwargs.get('loglevel', self.loglevel)

        src_dict = copy.deepcopy(kwargs.setdefault('model', {}))
        src_dict = {} if src_dict is None else src_dict

        multithread = kwargs.setdefault('multithread', False)
        threshold = kwargs.setdefault('threshold', 1E-2)
        max_kernel_radius = kwargs.get('max_kernel_radius')
        loge_bounds = kwargs.setdefault('loge_bounds', None)
        use_pylike = kwargs.setdefault('use_pylike', True)

        if loge_bounds:
            if len(loge_bounds) != 2:
                raise Exception('Wrong size of loge_bounds array.')
            loge_bounds[0] = (loge_bounds[0] if loge_bounds[0] is not None
                              else self.log_energies[0])
            loge_bounds[1] = (loge_bounds[1] if loge_bounds[1] is not None
                              else self.log_energies[-1])
        else:
            loge_bounds = [self.log_energies[0], self.log_energies[-1]]

        # Put the test source at the pixel closest to the ROI center
        xpix, ypix = (np.round((self.npix - 1.0) / 2.),
                      np.round((self.npix - 1.0) / 2.))
        cpix = np.array([xpix, ypix])

        map_geom = self._geom.to_image()
        frame = coordsys_to_frame(map_geom.coordsys)
        skydir = SkyCoord(*map_geom.pix_to_coord((cpix[0], cpix[1])),
                          frame=frame, unit='deg')
        skydir = skydir.transform_to('icrs')

        src_dict['ra'] = skydir.ra.deg
        src_dict['dec'] = skydir.dec.deg
        src_dict.setdefault('SpatialModel', 'PointSource')
        src_dict.setdefault('SpatialWidth', 0.3)
        src_dict.setdefault('Index', 2.0)
        src_dict.setdefault('Prefactor', 1E-13)

        counts = []
        bkg = []
        model = []
        c0_map = []
        eslices = []
        enumbins = []
        model_npred = 0
        for c in self.components:

            imin = utils.val_to_edge(c.log_energies, loge_bounds[0])[0]
            imax = utils.val_to_edge(c.log_energies, loge_bounds[1])[0]

            eslice = slice(imin, imax)
            bm = c.model_counts_map(exclude=kwargs['exclude']).data.astype('float')[
                eslice, ...]
            cm = c.counts_map().data.astype('float')[eslice, ...]

            bkg += [bm]
            counts += [cm]
            c0_map += [cash(cm, bm)]
            eslices += [eslice]
            enumbins += [cm.shape[0]]

        self.add_source('tsmap_testsource', src_dict, free=True,
                        init_source=False, use_single_psf=True,
                        use_pylike=use_pylike,
                        loglevel=logging.DEBUG)
        src = self.roi['tsmap_testsource']
        # self.logger.info(str(src_dict))
        modelname = utils.create_model_name(src)
        for c, eslice in zip(self.components, eslices):
            mm = c.model_counts_map('tsmap_testsource').data.astype('float')[
                eslice, ...]
            model_npred += np.sum(mm)
            model += [mm]

        self.delete_source('tsmap_testsource', loglevel=logging.DEBUG)

        for i, mm in enumerate(model):

            dpix = 3
            for j in range(mm.shape[0]):

                ix, iy = np.unravel_index(
                    np.argmax(mm[j, ...]), mm[j, ...].shape)

                mx = mm[j, ix, :] > mm[j, ix, iy] * threshold
                my = mm[j, :, iy] > mm[j, ix, iy] * threshold
                dpix = max(dpix, np.round(np.sum(mx) / 2.))
                dpix = max(dpix, np.round(np.sum(my) / 2.))

            if max_kernel_radius is not None and \
                    dpix > int(max_kernel_radius / self.components[i].binsz):
                dpix = int(max_kernel_radius / self.components[i].binsz)

            xslice = slice(max(int(xpix - dpix), 0),
                           min(int(xpix + dpix + 1), self.npix))
            model[i] = model[i][:, xslice, xslice]

        ts_values = np.zeros((self.npix, self.npix))
        amp_values = np.zeros((self.npix, self.npix))

        wrap = functools.partial(_ts_value_newton, counts=counts,
                                 bkg=bkg, model=model,
                                 C_0_map=c0_map)

        if kwargs['map_skydir'] is not None:

            map_offset = wcs_utils.skydir_to_pix(kwargs['map_skydir'],
                                                 map_geom.wcs)

            map_delta = 0.5 * kwargs['map_size'] / self.components[0].binsz
            xmin = max(int(np.ceil(map_offset[1] - map_delta)), 0)
            xmax = min(int(np.floor(map_offset[1] + map_delta)) + 1, self.npix)
            ymin = max(int(np.ceil(map_offset[0] - map_delta)), 0)
            ymax = min(int(np.floor(map_offset[0] + map_delta)) + 1, self.npix)

            xslice = slice(xmin, xmax)
            yslice = slice(ymin, ymax)
            xyrange = [range(xmin, xmax), range(ymin, ymax)]

            wcs = map_geom.wcs.deepcopy()
            npix = (ymax - ymin, xmax - xmin)
            crpix = (map_geom._crpix[0] - ymin, map_geom._crpix[1] - xmin)
            wcs.wcs.crpix[0] -= ymin
            wcs.wcs.crpix[1] -= xmin

            # FIXME: We should implement this with a proper cutout method
            map_geom = WcsGeom(wcs, npix, crpix=crpix)
        else:
            xyrange = [range(self.npix), range(self.npix)]
            xslice = slice(0, self.npix)
            yslice = slice(0, self.npix)

        positions = []
        for i, j in itertools.product(xyrange[0], xyrange[1]):
            p = [[k // 2, i, j] for k in enumbins]
            positions += [p]

        self.logger.log(loglevel, 'Fitting test source.')
        if multithread:
            pool = Pool()
            results = pool.map(wrap, positions)
            pool.close()
            pool.join()
        else:
            results = map(wrap, positions)

        for i, r in enumerate(results):
            ix = positions[i][0][1]
            iy = positions[i][0][2]
            ts_values[ix, iy] = r[0]
            amp_values[ix, iy] = r[1]

        ts_values = ts_values[xslice, yslice]
        amp_values = amp_values[xslice, yslice]

        ts_map = WcsNDMap(map_geom, ts_values)
        sqrt_ts_map = WcsNDMap(map_geom, ts_values**0.5)
        npred_map = WcsNDMap(map_geom, amp_values * model_npred)
        amp_map = WcsNDMap(map_geom, amp_values * src.get_norm())

        o = {'name': utils.join_strings([prefix, modelname]),
             'src_dict': copy.deepcopy(src_dict),
             'file': None,
             'ts': ts_map,
             'sqrt_ts': sqrt_ts_map,
             'npred': npred_map,
             'amplitude': amp_map,
             'loglike': -self.like(),
             'config': kwargs
             }

        return o


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

        schema = ConfigSchema(self.defaults['tscube'])
        schema.add_option('make_plots', True)
        schema.add_option('write_fits', True)
        schema.add_option('write_npy', True)
        config = schema.create_config(self.config['tscube'], **kwargs)

        maps = self._make_ts_cube(prefix, **config)

        if config['make_plots']:
            plotter = plotting.AnalysisPlotter(self.config['plotting'],
                                               fileio=self.config['fileio'],
                                               logging=self.config['logging'])

            plotter.make_tsmap_plots(maps, self.roi, suffix='tscube')

        self.logger.info("Finished TS cube")
        return maps

    def _make_ts_cube(self, prefix, **kwargs):

        skywcs = kwargs.get('wcs', self.geom.wcs)
        npix = kwargs.get('npix', self.npix)

        galactic = wcs_utils.is_galactic(skywcs)
        ref_skydir = wcs_utils.wcs_to_skydir(skywcs)
        refdir = pyLike.SkyDir(ref_skydir.ra.deg,
                               ref_skydir.dec.deg)
        pixsize = np.abs(skywcs.wcs.cdelt[0])

        skyproj = pyLike.FitScanner.buildSkyProj(str("AIT"),
                                                 refdir, pixsize, npix,
                                                 galactic)

        src_dict = copy.deepcopy(kwargs.setdefault('model', {}))
        src_dict = {} if src_dict is None else src_dict

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

        pylike_src.spectrum().normPar().setBounds(0, 1E6)

        fitScanner.setTestSource(pylike_src)

        self.logger.info("Running tscube")
        outfile = utils.format_filename(self.config['fileio']['workdir'],
                                        'tscube.fits',
                                        prefix=[prefix])

        try:
            fitScanner.run_tscube(True,
                                  kwargs['do_sed'], kwargs['nnorm'],
                                  kwargs['norm_sigma'],
                                  kwargs['cov_scale_bb'], kwargs['cov_scale'],
                                  kwargs['tol'], kwargs['max_iter'],
                                  kwargs['tol_type'],
                                  kwargs['remake_test_source'],
                                  kwargs['st_scan_level'],
                                  str(''),
                                  kwargs['init_lambda'])
        except Exception:
            fitScanner.run_tscube(True,
                                  kwargs['do_sed'], kwargs['nnorm'],
                                  kwargs['norm_sigma'],
                                  kwargs['cov_scale_bb'], kwargs['cov_scale'],
                                  kwargs['tol'], kwargs['max_iter'],
                                  kwargs['tol_type'],
                                  kwargs['remake_test_source'],
                                  kwargs['st_scan_level'])

        self.logger.info("Writing FITS output")

        fitScanner.writeFitsFile(str(outfile), str("gttscube"))

        convert_tscube(str(outfile), str(outfile))

        tscube = castro.TSCube.create_from_fits(outfile)
        ts_map = tscube.tsmap
        norm_map = tscube.normmap
        npred_map = copy.deepcopy(norm_map)
        npred_map.data *= tscube.refSpec.ref_npred.sum()
        amp_map = copy.deepcopy(norm_map)
        amp_map.data *= src_dict['Prefactor']

        sqrt_ts_map = copy.deepcopy(ts_map)
        sqrt_ts_map.data[...] = np.abs(sqrt_ts_map.data)**0.5

        o = {'name': utils.join_strings([prefix, modelname]),
             'src_dict': copy.deepcopy(src_dict),
             'file': os.path.basename(outfile),
             'ts': ts_map,
             'sqrt_ts': sqrt_ts_map,
             'npred': npred_map,
             'amplitude': amp_map,
             'config': kwargs,
             'tscube': tscube
             }

        if not kwargs['write_fits']:
            os.remove(outfile)
            os['file'] = None

        self.logger.info("Done")
        return o
