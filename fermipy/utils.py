# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import os
import re
import copy
import tempfile
import functools
from collections import OrderedDict
import xml.etree.cElementTree as et
import yaml
import numpy as np
import scipy.optimize
from scipy.ndimage import map_coordinates
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq
from scipy.ndimage.measurements import label
import scipy.special as special
from numpy.core import defchararray
from astropy.extern import six


def init_matplotlib_backend(backend=None):
    """This function initializes the matplotlib backend.  When no
    DISPLAY is available the backend is automatically set to 'Agg'.

    Parameters
    ----------
    backend : str
       matplotlib backend name.
    """

    import matplotlib

    try:
        os.environ['DISPLAY']
    except KeyError:
        matplotlib.use('Agg')
    else:
        if backend is not None:
            matplotlib.use(backend)


def unicode_representer(dumper, uni):
    node = yaml.ScalarNode(tag=u'tag:yaml.org,2002:str', value=uni)
    return node


yaml.add_representer(six.text_type, unicode_representer)


def load_yaml(infile, **kwargs):
    return yaml.load(open(infile), **kwargs)


def write_yaml(o, outfile, **kwargs):
    yaml.dump(tolist(o), open(outfile, 'w'), **kwargs)


def load_npy(infile):
    return np.load(infile).flat[0]


def load_data(infile, workdir=None):
    """Load python data structure from either a YAML or numpy file. """
    infile = resolve_path(infile, workdir=workdir)
    infile, ext = os.path.splitext(infile)

    if os.path.isfile(infile + '.npy'):
        infile += '.npy'
    elif os.path.isfile(infile + '.yaml'):
        infile += '.yaml'
    else:
        raise Exception('Input file does not exist.')

    ext = os.path.splitext(infile)[1]

    if ext == '.npy':
        return infile, load_npy(infile)
    elif ext == '.yaml':
        return infile, load_yaml(infile)
    else:
        raise Exception('Unrecognized extension.')


def resolve_path(path, workdir=None):
    if os.path.isabs(path):
        return path
    elif workdir is None:
        return os.path.abspath(path)
    else:
        return os.path.join(workdir, path)


def resolve_file_path(path, **kwargs):
    dirs = kwargs.get('search_dirs', [])
    expand = kwargs.get('expand', False)

    if path is None:
        return None

    out_path = None
    if os.path.isabs(os.path.expandvars(path)) and \
            os.path.isfile(os.path.expandvars(path)):
        out_path = path
    else:
        for d in dirs:
            if not os.path.isdir(os.path.expandvars(d)):
                continue
            p = os.path.join(d, path)
            if os.path.isfile(os.path.expandvars(p)):
                out_path = p
                break

    if out_path is None:
        raise Exception('Failed to resolve file path: %s' % path)

    if expand:
        out_path = os.path.expandvars(out_path)

    return out_path


def resolve_file_path_list(pathlist, workdir, prefix='',
                           randomize=False):
    """Resolve the path of each file name in the file ``pathlist`` and
    write the updated paths to a new file.
    """
    files = []
    with open(pathlist, 'r') as f:
        files = [line.strip() for line in f]

    newfiles = []
    for f in files:
        f = os.path.expandvars(f)
        if os.path.isfile(f):
            newfiles += [f]
        else:
            newfiles += [os.path.join(workdir, f)]

    if randomize:
        _, tmppath = tempfile.mkstemp(prefix=prefix, dir=workdir)
    else:
        tmppath = os.path.join(workdir, prefix)

    tmppath += '.txt'

    with open(tmppath, 'w') as tmpfile:
        tmpfile.write("\n".join(newfiles))
    return tmppath


def is_fits_file(path):

    if (path.endswith('.fit') or path.endswith('.fits') or
            path.endswith('.fit.gz') or path.endswith('.fits.gz')):
        return True
    else:
        return False


def collect_dirs(path, max_depth=1, followlinks=True):
    """Recursively find directories under the given path."""

    if not os.path.isdir(path):
        return []

    o = [path]

    if max_depth == 0:
        return o

    for subdir in os.listdir(path):

        subdir = os.path.join(path, subdir)

        if not os.path.isdir(subdir):
            continue

        o += [subdir]

        if os.path.islink(subdir) and not followlinks:
            continue

        if max_depth > 0:
            o += collect_dirs(subdir, max_depth=max_depth - 1)

    return list(set(o))


def match_regex_list(patterns, string):
    """Perform a regex match of a string against a list of patterns.
    Returns true if the string matches at least one pattern in the
    list."""

    for p in patterns:

        if re.findall(p, string):
            return True

    return False


def find_rows_by_string(tab, names, colnames=['assoc']):
    """Find the rows in a table ``tab`` that match at least one of the
    strings in ``names``.  This method ignores whitespace and case
    when matching strings.

    Parameters
    ----------
    tab : `astropy.table.Table`
       Table that will be searched.

    names : list
       List of strings.

    colname : str
       Name of the table column that will be searched for matching string.

    Returns
    -------
    mask : `~numpy.ndarray`
       Boolean mask for rows with matching strings.

    """
    mask = np.empty(len(tab), dtype=bool)
    mask.fill(False)
    names = [name.lower().replace(' ', '') for name in names]

    for colname in colnames:

        if colname not in tab.columns:
            continue

        col = tab[[colname]].copy()
        col[colname] = defchararray.replace(defchararray.lower(col[colname]).astype(str),
                                        ' ', '')
        for name in names:
            mask |= col[colname] == name
    return mask


def join_strings(strings, sep='_'):
    if strings is None:
        return ''
    else:
        if not isinstance(strings, list):
            strings = [strings]
        return sep.join([s for s in strings if s])


def format_filename(outdir, basename, prefix=None, extension=None):
    filename = join_strings(prefix)
    filename = join_strings([filename, basename])

    if extension is not None:

        if extension.startswith('.'):
            filename += extension
        else:
            filename += '.' + extension

    return os.path.join(outdir, filename)


def strip_suffix(filename, suffix):
    for s in suffix:
        filename = re.sub(r'\.%s$' % s, '', filename)

    return filename


def met_to_mjd(time):
    """"Convert mission elapsed time to mean julian date."""
    return 54682.65 + (time - 239557414.0) / (86400.)


RA_NGP = np.radians(192.8594812065348)
DEC_NGP = np.radians(27.12825118085622)
L_CP = np.radians(122.9319185680026)


def gal2eq(l, b):
    L_0 = L_CP - np.pi / 2.
    RA_0 = RA_NGP + np.pi / 2.

    l = np.array(l, ndmin=1)
    b = np.array(b, ndmin=1)

    l, b = np.radians(l), np.radians(b)

    sind = np.sin(b) * np.sin(DEC_NGP) + np.cos(b) * np.cos(DEC_NGP) * np.sin(
        l - L_0)

    dec = np.arcsin(sind)

    cosa = np.cos(l - L_0) * np.cos(b) / np.cos(dec)
    sina = (np.cos(b) * np.sin(DEC_NGP) * np.sin(l - L_0) - np.sin(b) * np.cos(
        DEC_NGP)) / np.cos(dec)

    dec = np.degrees(dec)

    cosa[cosa < -1.0] = -1.0
    cosa[cosa > 1.0] = 1.0
    ra = np.arccos(cosa)
    ra[np.where(sina < 0.)] = -ra[np.where(sina < 0.)]

    ra = np.degrees(ra + RA_0)

    ra = np.mod(ra, 360.)
    dec = np.mod(dec + 90., 180.) - 90.

    return ra, dec


def eq2gal(ra, dec):
    L_0 = L_CP - np.pi / 2.
    RA_0 = RA_NGP + np.pi / 2.
    DEC_0 = np.pi / 2. - DEC_NGP

    ra = np.array(ra, ndmin=1)
    dec = np.array(dec, ndmin=1)

    ra, dec = np.radians(ra), np.radians(dec)

    np.sinb = np.sin(dec) * np.cos(DEC_0) - np.cos(dec) * np.sin(
        ra - RA_0) * np.sin(DEC_0)

    b = np.arcsin(np.sinb)

    cosl = np.cos(dec) * np.cos(ra - RA_0) / np.cos(b)
    sinl = (np.sin(dec) * np.sin(DEC_0) + np.cos(dec) * np.sin(
        ra - RA_0) * np.cos(DEC_0)) / np.cos(b)

    b = np.degrees(b)

    cosl[cosl < -1.0] = -1.0
    cosl[cosl > 1.0] = 1.0
    l = np.arccos(cosl)
    l[np.where(sinl < 0.)] = - l[np.where(sinl < 0.)]

    l = np.degrees(l + L_0)

    l = np.mod(l, 360.)
    b = np.mod(b + 90., 180.) - 90.

    return l, b


def xyz_to_lonlat(*args):
    if len(args) == 1:
        x, y, z = args[0][0], args[0][1], args[0][2]
    else:
        x, y, z = args[0], args[1], args[2]

    lat = np.pi / 2. - np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
    lon = np.arctan2(y, x)
    return lon, lat


def lonlat_to_xyz(lon, lat):
    phi = lon
    theta = np.pi / 2. - lat
    return np.array([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)])


def project(lon0, lat0, lon1, lat1):
    """This function performs a stereographic projection on the unit
    vector (lon1,lat1) with the pole defined at the reference unit
    vector (lon0,lat0)."""

    costh = np.cos(np.pi / 2. - lat0)
    cosphi = np.cos(lon0)

    sinth = np.sin(np.pi / 2. - lat0)
    sinphi = np.sin(lon0)

    xyz = lonlat_to_xyz(lon1, lat1)
    x1 = xyz[0]
    y1 = xyz[1]
    z1 = xyz[2]

    x1p = x1 * costh * cosphi + y1 * costh * sinphi - z1 * sinth
    y1p = -x1 * sinphi + y1 * cosphi
    z1p = x1 * sinth * cosphi + y1 * sinth * sinphi + z1 * costh

    r = np.arctan2(np.sqrt(x1p ** 2 + y1p ** 2), z1p)
    phi = np.arctan2(y1p, x1p)

    return r * np.cos(phi), r * np.sin(phi)


def separation_cos_angle(lon0, lat0, lon1, lat1):
    """Evaluate the cosine of the angular separation between two
    direction vectors."""
    return (np.sin(lat1) * np.sin(lat0) + np.cos(lat1) * np.cos(lat0) *
            np.cos(lon1 - lon0))


def dot_prod(xyz0, xyz1):
    """Compute the dot product between two cartesian vectors where the
    second dimension contains the vector components."""
    return np.sum(xyz0 * xyz1, axis=1)


def angle_to_cartesian(lon, lat):
    """Convert spherical coordinates to cartesian unit vectors."""
    theta = np.array(np.pi / 2. - lat)
    return np.vstack((np.sin(theta) * np.cos(lon),
                      np.sin(theta) * np.sin(lon),
                      np.cos(theta))).T


def scale_parameter(p):
    if isstr(p):
        p = float(p)

    if p > 0:
        scale = 10 ** -np.round(np.log10(1. / p))
        return p / scale, scale
    else:
        return p, 1.0


def update_bounds(val, bounds):
    return min(val, bounds[0]), max(val, bounds[1])


def apply_minmax_selection(val, val_minmax):
    if val_minmax is None:
        return True

    if val_minmax[0] is None:
        min_cut = True
    elif np.isfinite(val) and val >= val_minmax[0]:
        min_cut = True
    else:
        min_cut = False

    if val_minmax[1] is None:
        max_cut = True
    elif np.isfinite(val) and val <= val_minmax[1]:
        max_cut = True
    else:
        max_cut = False

    return (min_cut and max_cut)


def create_source_name(skydir, floor=True, prefix='PS'):
    hms = skydir.icrs.ra.hms
    dms = skydir.icrs.dec.dms

    if floor:
        ra_ms = np.floor(10. * (hms.m + hms.s / 60.)) / 10.
        dec_ms = np.floor(np.abs(dms.m + dms.s / 60.))
    else:
        ra_ms = (hms.m + hms.s / 60.)
        dec_ms = np.abs(dms.m + dms.s / 60.)

    return '%s J%02.f%04.1f%+03.f%02.f' % (prefix, hms.h, ra_ms,
                                           dms.d, dec_ms)


def create_model_name(src):
    """Generate a name for a source object given its spatial/spectral
    properties.

    Parameters
    ----------
    src : `~fermipy.roi_model.Source`
          A source object.

    Returns
    -------
    name : str
           A source name.
    """
    o = ''
    spatial_type = src['SpatialModel'].lower()
    o += spatial_type

    if spatial_type == 'gaussian':
        o += '_s%04.2f' % src['SpatialWidth']

    if src['SpectrumType'] == 'PowerLaw':
        o += '_powerlaw_%04.2f' % float(src.spectral_pars['Index']['value'])
    else:
        o += '_%s' % (src['SpectrumType'].lower())

    return o


def cov_to_correlation(cov):
    """Compute the correlation matrix given the covariance matrix.

    Parameters
    ----------
    cov : `~numpy.ndarray`
        N x N matrix of covariances among N parameters.

    Returns
    -------
    corr : `~numpy.ndarray`
        N x N matrix of correlations among N parameters.
    """
    err = np.sqrt(np.diag(cov))
    errinv = np.ones_like(err) * np.nan
    m = np.isfinite(err) & (err != 0)
    errinv[m] = 1. / err[m]
    corr = np.array(cov)
    return corr * np.outer(errinv, errinv)


def ellipse_to_cov(sigma_maj, sigma_min, theta):
    """Compute the covariance matrix in two variables x and y given
    the std. deviation along the semi-major and semi-minor axes and
    the rotation angle of the error ellipse.

    Parameters
    ----------
    sigma_maj : float
        Std. deviation along major axis of error ellipse.

    sigma_min : float
        Std. deviation along minor axis of error ellipse.

    theta : float
        Rotation angle in radians from x-axis to ellipse major axis.
    """
    cth = np.cos(theta)
    sth = np.sin(theta)
    covxx = cth**2 * sigma_maj**2 + sth**2 * sigma_min**2
    covyy = sth**2 * sigma_maj**2 + cth**2 * sigma_min**2
    covxy = cth * sth * sigma_maj**2 - cth * sth * sigma_min**2
    return np.array([[covxx, covxy], [covxy, covyy]])


def twosided_cl_to_dlnl(cl):
    """Compute the delta-loglikehood value that corresponds to a
    two-sided interval of the given confidence level.

    Parameters
    ----------
    cl : float
        Confidence level.

    Returns
    -------
    dlnl : float    
        Delta-loglikelihood value with respect to the maximum of the
        likelihood function.
    """
    return 0.5 * np.power(np.sqrt(2.) * special.erfinv(cl), 2)


def twosided_dlnl_to_cl(dlnl):
    """Compute the confidence level that corresponds to a two-sided
    interval with a given change in the loglikelihood value.

    Parameters
    ----------
    dlnl : float
        Delta-loglikelihood value with respect to the maximum of the
        likelihood function.

    Returns
    -------
    cl : float
        Confidence level.
    """
    return special.erf(dlnl**0.5)


def onesided_cl_to_dlnl(cl):
    """Compute the delta-loglikehood values that corresponds to an
    upper limit of the given confidence level.

    Parameters
    ----------
    cl : float
        Confidence level.

    Returns
    -------
    dlnl : float
        Delta-loglikelihood value with respect to the maximum of the
        likelihood function.
    """
    alpha = 1.0 - cl
    return 0.5 * np.power(np.sqrt(2.) * special.erfinv(1 - 2 * alpha), 2.)


def onesided_dlnl_to_cl(dlnl):
    """Compute the confidence level that corresponds to an upper limit
    with a given change in the loglikelihood value.

    Parameters
    ----------
    dlnl : float
        Delta-loglikelihood value with respect to the maximum of the
        likelihood function.

    Returns
    -------
    cl : float
        Confidence level.
    """
    alpha = (1.0 - special.erf(dlnl**0.5)) / 2.0
    return 1.0 - alpha


def interpolate_function_min(x, y):
    sp = scipy.interpolate.splrep(x, y, k=2, s=0)

    def fn(t):
        return scipy.interpolate.splev(t, sp, der=1)

    if np.sign(fn(x[0])) == np.sign(fn(x[-1])):

        if np.sign(fn(x[0])) == -1:
            return x[-1]
        else:
            return x[0]

    x0 = scipy.optimize.brentq(fn,
                               x[0], x[-1],
                               xtol=1e-10 * np.median(x))

    return x0


def find_function_root(fn, x0, xb, delta=0.0, bounds=None):
    """Find the root of a function: f(x)+delta in the interval encompassed
    by x0 and xb.

    Parameters
    ----------

    fn : function
       Python function.

    x0 : float
       Fixed bound for the root search.  This will either be used as
       the lower or upper bound depending on the relative value of xb.

    xb : float
       Upper or lower bound for the root search.  If a root is not
       found in the interval [x0,xb]/[xb,x0] this value will be
       increased/decreased until a change in sign is found.

    """

    if x0 == xb:
        return np.nan

    for i in range(10):
        if np.sign(fn(xb) + delta) != np.sign(fn(x0) + delta):
            break
        if bounds is not None and (xb < bounds[0] or xb > bounds[1]):
            break
        if xb < x0:
            xb *= 0.5
        else:
            xb *= 2.0

    # Failed to find a root
    if np.sign(fn(xb) + delta) == np.sign(fn(x0) + delta):
        return np.nan

    if x0 == 0:
        xtol = 1e-10 * np.abs(xb)
    else:
        xtol = 1e-10 * np.abs(xb + x0)

    return brentq(lambda t: fn(t) + delta, x0, xb, xtol=xtol)


def get_parameter_limits(xval, loglike, cl_limit=0.95, cl_err=0.68269, tol=1E-2,
                         bounds=None):
    """Compute upper/lower limits, peak position, and 1-sigma errors
    from a 1-D likelihood function.  This function uses the
    delta-loglikelihood method to evaluate parameter limits by
    searching for the point at which the change in the log-likelihood
    value with respect to the maximum equals a specific value.  A
    cubic spline fit to the log-likelihood values is used to
    improve the accuracy of the calculation.

    Parameters
    ----------

    xval : `~numpy.ndarray`
       Array of parameter values.

    loglike : `~numpy.ndarray`
       Array of log-likelihood values.

    cl_limit : float
       Confidence level to use for limit calculation.

    cl_err : float
       Confidence level to use for two-sided confidence interval
       calculation.

    tol : float
       Absolute precision of likelihood values.

    Returns
    -------

    x0 : float
        Coordinate at maximum of likelihood function.

    err_lo : float    
        Lower error for two-sided confidence interval with CL
        ``cl_err``.  Corresponds to point (x < x0) at which the
        log-likelihood falls by a given value with respect to the
        maximum (0.5 for 1 sigma).  Set to nan if the change in the
        log-likelihood function at the lower bound of the ``xval``
        input array is less than than the value for the given CL.

    err_hi : float
        Upper error for two-sided confidence interval with CL
        ``cl_err``. Corresponds to point (x > x0) at which the
        log-likelihood falls by a given value with respect to the
        maximum (0.5 for 1 sigma).  Set to nan if the change in the
        log-likelihood function at the upper bound of the ``xval``
        input array is less than the value for the given CL.

    err : float
        Symmetric 1-sigma error.  Average of ``err_lo`` and ``err_hi``
        if both are defined.

    ll : float
        Lower limit evaluated at confidence level ``cl_limit``.

    ul : float
        Upper limit evaluated at confidence level ``cl_limit``.

    lnlmax : float
        Log-likelihood value at ``x0``.

    """

    dlnl_limit = onesided_cl_to_dlnl(cl_limit)
    dlnl_err = twosided_cl_to_dlnl(cl_err)

    try:
        # Pad the likelihood function
        # if len(xval) >= 3 and np.max(loglike) - loglike[-1] < 1.5*dlnl_limit:
        #    p = np.polyfit(xval[-3:], loglike[-3:], 2)
        #    x = np.linspace(xval[-1], 10 * xval[-1], 3)[1:]
        #    y = np.polyval(p, x)
        #    x = np.concatenate((xval, x))
        #    y = np.concatenate((loglike, y))
        # else:
        x, y = xval, loglike
        spline = UnivariateSpline(x, y, k=2,
                                  #k=min(len(xval) - 1, 3),
                                  w=(1 / tol) * np.ones(len(x)))
    except:
        print("Failed to create spline: ", xval, loglike)
        return {'x0': np.nan, 'ul': np.nan, 'll': np.nan,
                'err_lo': np.nan, 'err_hi': np.nan, 'err': np.nan,
                'lnlmax': np.nan}

    sd = spline.derivative()

    imax = np.argmax(loglike)
    ilo = max(imax - 1, 0)
    ihi = min(imax + 1, len(xval) - 1)

    # Find the peak
    x0 = xval[imax]

    # Refine the peak position
    if np.sign(sd(xval[ilo])) != np.sign(sd(xval[ihi])):
        x0 = find_function_root(sd, xval[ilo], xval[ihi])

    lnlmax = float(spline(x0))

    def fn(t): return spline(t) - lnlmax
    fn_val = fn(xval)
    if np.any(fn_val[imax:] < -dlnl_limit):
        xhi = xval[imax:][fn_val[imax:] < -dlnl_limit][0]
    else:
        xhi = xval[-1]        
    # EAC: brute force check that xhi is greater than x0
    # The fabs is here in case x0 is negative
    if xhi <= x0:
        xhi = x0 + np.fabs(x0)

    if np.any(fn_val[:imax] < -dlnl_limit):
        xlo = xval[:imax][fn_val[:imax] < -dlnl_limit][-1]
    else:
        xlo = xval[0]
    # EAC: brute force check that xlo is less than x0
    # The fabs is here in case x0 is negative        
    if xlo >= x0:
        xlo = x0 - 0.5*np.fabs(x0)

    ul = find_function_root(fn, x0, xhi, dlnl_limit, bounds=bounds)
    ll = find_function_root(fn, x0, xlo, dlnl_limit, bounds=bounds)
    err_lo = np.abs(x0 - find_function_root(fn, x0, xlo, dlnl_err,
                                            bounds=bounds))
    err_hi = np.abs(x0 - find_function_root(fn, x0, xhi, dlnl_err,
                                            bounds=bounds))

    err = np.nan
    if np.isfinite(err_lo) and np.isfinite(err_hi):
        err = 0.5 * (err_lo + err_hi)
    elif np.isfinite(err_hi):
        err = err_hi
    elif np.isfinite(err_lo):
        err = err_lo

    o = {'x0': x0, 'ul': ul, 'll': ll,
         'err_lo': err_lo, 'err_hi': err_hi, 'err': err,
         'lnlmax': lnlmax}
    return o


def poly_to_parabola(coeff):
    sigma = np.sqrt(1. / np.abs(2.0 * coeff[0]))
    x0 = -coeff[1] / (2 * coeff[0])
    y0 = (1. - (coeff[1] ** 2 - 4 * coeff[0] * coeff[2])) / (4 * coeff[0])

    return x0, sigma, y0


def parabola(xy, amplitude, x0, y0, sx, sy, theta):
    """Evaluate a 2D parabola given by:

    f(x,y) = f_0 - (1/2) * \delta^T * R * \Sigma * R^T * \delta

    where

    \delta = [(x - x_0), (y - y_0)]

    and R is the matrix for a 2D rotation by angle \theta and \Sigma
    is the covariance matrix:

    \Sigma = [[1/\sigma_x^2, 0           ],
              [0           , 1/\sigma_y^2]] 

    Parameters
    ----------
    xy : tuple    
       Tuple containing x and y arrays for the values at which the
       parabola will be evaluated.

    amplitude : float
       Constant offset value.

    x0 : float
       Centroid in x coordinate.

    y0 : float
       Centroid in y coordinate.

    sx : float
       Standard deviation along first axis (x-axis when theta=0).

    sy : float
       Standard deviation along second axis (y-axis when theta=0).

    theta : float
       Rotation angle in radians.

    Returns
    -------
    vals : `~numpy.ndarray`    
       Values of the parabola evaluated at the points defined in the
       `xy` input tuple.

    """

    x = xy[0]
    y = xy[1]

    cth = np.cos(theta)
    sth = np.sin(theta)
    a = (cth ** 2) / (2 * sx ** 2) + (sth ** 2) / (2 * sy ** 2)
    b = -(np.sin(2 * theta)) / (4 * sx ** 2) + (np.sin(2 * theta)) / (
        4 * sy ** 2)
    c = (sth ** 2) / (2 * sx ** 2) + (cth ** 2) / (2 * sy ** 2)
    vals = amplitude - (a * ((x - x0) ** 2) +
                        2 * b * (x - x0) * (y - y0) +
                        c * ((y - y0) ** 2))

    return vals


def get_bounded_slice(idx, dpix, shape):

    dpix = int(dpix)
    idx_lo = idx - dpix
    idx_hi = idx + dpix + 1

    if idx_lo < 0:
        idx_lo = max(idx_lo, 0)
        idx_hi = idx_lo + (2 * dpix + 1)
    elif idx_hi > shape:
        idx_hi = min(idx_hi, shape)
        idx_lo = idx_hi - (2 * dpix + 1)

    return slice(idx_lo, idx_hi)


def get_region_mask(z, delta, xy=None):
    """Get mask of connected region within delta of max(z)."""

    if xy is None:
        ix, iy = np.unravel_index(np.argmax(z), z.shape)
    else:
        ix, iy = xy

    mz = (z > z[ix, iy] - delta)
    labels = label(mz)[0]
    mz &= labels == labels[ix, iy]
    return mz


def fit_parabola(z, ix, iy, dpix=3, zmin=None):
    """Fit a parabola to a 2D numpy array.  This function will fit a
    parabola with the functional form described in
    `~fermipy.utils.parabola` to a 2D slice of the input array `z`.
    The fit region encompasses pixels that are within `dpix` of the
    pixel coordinate (iz,iy) OR that have a value relative to the peak
    value greater than `zmin`.

    Parameters
    ----------
    z : `~numpy.ndarray`

    ix : int
       X index of center pixel of fit region in array `z`.

    iy : int
       Y index of center pixel of fit region in array `z`.

    dpix : int
       Max distance from center pixel of fit region.

    zmin : float

    """
    offset = make_pixel_distance(z.shape, iy, ix)
    x, y = np.meshgrid(np.arange(z.shape[0]), np.arange(z.shape[1]),
                       indexing='ij')

    m = (offset <= dpix)
    if np.sum(m) < 9:
        m = (offset <= dpix + 0.5)

    if zmin is not None:
        m |= get_region_mask(z, np.abs(zmin), (ix, iy))

    sx = get_bounded_slice(ix, dpix, z.shape[0])
    sy = get_bounded_slice(iy, dpix, z.shape[1])

    coeffx = poly_to_parabola(np.polyfit(x[sx, iy], z[sx, iy], 2))
    coeffy = poly_to_parabola(np.polyfit(y[ix, sy], z[ix, sy], 2))
    #p0 = [coeffx[2], coeffx[0], coeffy[0], coeffx[1], coeffy[1], 0.0]
    p0 = [coeffx[2], float(ix), float(iy), coeffx[1], coeffy[1], 0.0]

    o = {'fit_success': True, 'p0': p0}

    def curve_fit_fn(*args):
        return np.ravel(parabola(*args))

    try:
        bounds = (-np.inf * np.ones(6), np.inf * np.ones(6))
        bounds[0][1] = -0.5
        bounds[0][2] = -0.5
        bounds[1][1] = z.shape[0] - 0.5
        bounds[1][2] = z.shape[1] - 0.5
        popt, pcov = scipy.optimize.curve_fit(curve_fit_fn,
                                              (np.ravel(x[m]), np.ravel(y[m])),
                                              np.ravel(z[m]), p0, bounds=bounds)
    except Exception:
        popt = copy.deepcopy(p0)
        o['fit_success'] = False

    fm = parabola((x[m], y[m]), *popt)
    df = fm - z[m]
    rchi2 = np.sum(df ** 2) / len(fm)

    o['rchi2'] = rchi2
    o['x0'] = popt[1]
    o['y0'] = popt[2]
    o['sigmax'] = np.abs(popt[3])
    o['sigmay'] = np.abs(popt[4])
    o['sigma'] = np.sqrt(o['sigmax'] ** 2 + o['sigmay'] ** 2)
    o['z0'] = popt[0]
    o['theta'] = popt[5]
    o['popt'] = popt
    o['mask'] = m

    a = max(o['sigmax'], o['sigmay'])
    b = min(o['sigmax'], o['sigmay'])

    o['eccentricity'] = np.sqrt(1 - b ** 2 / a ** 2)
    o['eccentricity2'] = np.sqrt(a ** 2 / b ** 2 - 1)

    return o


def split_bin_edges(edges, npts=2):
    """Subdivide an array of bins by splitting each bin into ``npts``
    subintervals.

    Parameters
    ----------
    edges : `~numpy.ndarray`
        Bin edge array.

    npts : int
        Number of intervals into which each bin will be subdivided.

    Returns
    -------
    edges : `~numpy.ndarray`
        Subdivided bin edge array.

    """
    if npts < 2:
        return edges

    x = (edges[:-1, None] +
         (edges[1:, None] - edges[:-1, None]) *
         np.linspace(0.0, 1.0, npts + 1)[None, :])
    return np.unique(np.ravel(x))


def center_to_edge(center):

    if len(center) == 1:
        delta = np.array(1.0, ndmin=1)
    else:
        delta = center[1:] - center[:-1]

    edges = 0.5 * (center[1:] + center[:-1])
    edges = np.insert(edges, 0, center[0] - 0.5 * delta[0])
    edges = np.append(edges, center[-1] + 0.5 * delta[-1])
    return edges


def edge_to_center(edges):
    return 0.5 * (edges[1:] + edges[:-1])


def edge_to_width(edges):
    return (edges[1:] - edges[:-1])


def val_to_bin(edges, x):
    """Convert axis coordinate to bin index."""
    ibin = np.digitize(np.array(x, ndmin=1), edges) - 1
    return ibin


def val_to_pix(center, x):
    return np.interp(x, center, np.arange(len(center)).astype(float))


def val_to_edge(edges, x):
    """Convert axis coordinate to bin index."""
    edges = np.array(edges)
    w = edges[1:] - edges[:-1]
    w = np.insert(w, 0, w[0])
    ibin = np.digitize(np.array(x, ndmin=1), edges - 0.5 * w) - 1
    ibin[ibin < 0] = 0
    return ibin


def val_to_bin_bounded(edges, x):
    """Convert axis coordinate to bin index."""
    nbins = len(edges) - 1
    ibin = val_to_bin(edges, x)
    ibin[ibin < 0] = 0
    ibin[ibin > nbins - 1] = nbins - 1
    return ibin


def extend_array(edges, binsz, lo, hi):
    """Extend an array to encompass lo and hi values."""

    numlo = int(np.ceil((edges[0] - lo) / binsz))
    numhi = int(np.ceil((hi - edges[-1]) / binsz))

    edges = copy.deepcopy(edges)
    if numlo > 0:
        edges_lo = np.linspace(edges[0] - numlo * binsz, edges[0], numlo + 1)
        edges = np.concatenate((edges_lo[:-1], edges))

    if numhi > 0:
        edges_hi = np.linspace(edges[-1], edges[-1] + numhi * binsz, numhi + 1)
        edges = np.concatenate((edges, edges_hi[1:]))

    return edges


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def fits_recarray_to_dict(table):
    """Convert a FITS recarray to a python dictionary."""

    cols = {}
    for icol, col in enumerate(table.columns.names):

        col_data = table.data[col]
        if type(col_data[0]) == np.float32:
            cols[col] = np.array(col_data, dtype=float)
        elif type(col_data[0]) == np.float64:
            cols[col] = np.array(col_data, dtype=float)
        elif type(col_data[0]) == str:
            cols[col] = np.array(col_data, dtype=str)
        elif type(col_data[0]) == np.string_:
            cols[col] = np.array(col_data, dtype=str)
        elif type(col_data[0]) == np.int16:
            cols[col] = np.array(col_data, dtype=int)
        elif type(col_data[0]) == np.ndarray:
            cols[col] = np.array(col_data)
        else:
            raise Exception(
                'Unrecognized column type: %s %s' % (col, str(type(col_data))))

    return cols


def unicode_to_str(args):
    o = {}
    for k, v in args.items():

        if isstr(v):
            o[k] = str(v)
        else:
            o[k] = v

    return o


def isstr(s):
    """String instance testing method that works under both Python 2.X
    and 3.X.  Returns true if the input is a string."""

    try:
        return isinstance(s, basestring)
    except NameError:
        return isinstance(s, str)


def xmlpath_to_path(path):
    if path is None:
        return path

    return re.sub(r'\$\(([a-zA-Z\_]+)\)', r'$\1', path)


def path_to_xmlpath(path):
    if path is None:
        return path

    return re.sub(r'\$([a-zA-Z\_]+)', r'$(\1)', path)


def create_xml_element(root, name, attrib):
    el = et.SubElement(root, name)
    for k, v in attrib.iteritems():

        if isinstance(v, bool):
            el.set(k, str(int(v)))
        elif isstr(v):
            el.set(k, v)
        elif np.isfinite(v):
            el.set(k, str(v))

    return el


def load_xml_elements(root, path):
    o = {}
    for p in root.findall(path):

        if 'name' in p.attrib:
            o[p.attrib['name']] = copy.deepcopy(p.attrib)
        else:
            o.update(p.attrib)
    return o


def prettify_xml(elem):
    """Return a pretty-printed XML string for the Element.
    """
    from xml.dom import minidom
    import xml.etree.cElementTree as et

    rough_string = et.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def arg_to_list(arg):
    if arg is None:
        return []
    elif isinstance(arg, list):
        return arg
    else:
        return [arg]


def update_keys(input_dict, key_map):
    o = {}
    for k, v in input_dict.items():

        if k in key_map.keys():
            k = key_map[k]

        if isinstance(v, dict):
            o[k] = update_keys(v, key_map)
        else:
            o[k] = v

    return o


def create_dict(d0, **kwargs):
    o = copy.deepcopy(d0)
    o = merge_dict(o, kwargs, add_new_keys=True)
    return o


def merge_dict(d0, d1, add_new_keys=False, append_arrays=False):
    """Recursively merge the contents of python dictionary d0 with
    the contents of another python dictionary, d1.

    Parameters
    ----------
    d0 : dict
       The input dictionary.

    d1 : dict
       Dictionary to be merged with the input dictionary.

    add_new_keys : str
       Do not skip keys that only exist in d1.

    append_arrays : bool
       If an element is a numpy array set the value of that element by
       concatenating the two arrays.
    """

    if d1 is None:
        return d0
    elif d0 is None:
        return d1
    elif d0 is None and d1 is None:
        return {}

    od = {}

    for k, v in d0.items():

        t0 = None
        t1 = None

        if k in d0:
            t0 = type(d0[k])
        if k in d1:
            t1 = type(d1[k])

        if k not in d1:
            od[k] = copy.deepcopy(d0[k])
        elif isinstance(v, dict) and isinstance(d1[k], dict):
            od[k] = merge_dict(d0[k], d1[k], add_new_keys, append_arrays)
        elif isinstance(v, list) and isstr(d1[k]):
            od[k] = d1[k].split(',')
        elif isinstance(v, dict) and d1[k] is None:
            od[k] = copy.deepcopy(d0[k])
        elif isinstance(v, np.ndarray) and append_arrays:
            od[k] = np.concatenate((v, d1[k]))
        elif (d0[k] is not None and d1[k] is not None) and t0 != t1:

            if t0 == dict or t0 == list:
                raise Exception('Conflicting types in dictionary merge for '
                                'key %s %s %s' % (k, t0, t1))
            od[k] = t0(d1[k])
        else:
            od[k] = copy.copy(d1[k])

    if add_new_keys:
        for k, v in d1.items():
            if k not in d0:
                od[k] = copy.deepcopy(d1[k])

    return od


def merge_list_of_dicts(listofdicts):
    # assumes every item in list has the same keys
    merged = copy.deepcopy(listofdicts[0])
    for k in merged.keys():
        merged[k] = []
    for i in xrange(len(listofdicts)):
        for k in merged.keys():
            merged[k].append(listofdicts[i][k])
    return merged


def tolist(x):
    """ convenience function that takes in a
        nested structure of lists and dictionaries
        and converts everything to its base objects.
        This is useful for dupming a file to yaml.

        (a) numpy arrays into python lists

            >>> type(tolist(np.asarray(123))) == int
            True
            >>> tolist(np.asarray([1,2,3])) == [1,2,3]
            True

        (b) numpy strings into python strings.

            >>> tolist([np.asarray('cat')])==['cat']
            True

        (c) an ordered dict to a dict

            >>> ordered=OrderedDict(a=1, b=2)
            >>> type(tolist(ordered)) == dict
            True

        (d) converts unicode to regular strings

            >>> type(u'a') == str
            False
            >>> type(tolist(u'a')) == str
            True

        (e) converts numbers & bools in strings to real represntation,
            (i.e. '123' -> 123)

            >>> type(tolist(np.asarray('123'))) == int
            True
            >>> type(tolist('123')) == int
            True
            >>> tolist('False') == False
            True
    """
    if isinstance(x, list):
        return map(tolist, x)
    elif isinstance(x, dict):
        return dict((tolist(k), tolist(v)) for k, v in x.items())
    elif isinstance(x, np.ndarray) or isinstance(x, np.number):
        # note, call tolist again to convert strings of numbers to numbers
        return tolist(x.tolist())
    elif isinstance(x, OrderedDict):
        return dict(x)
    elif isinstance(x, np.bool_):
        return bool(x)
    elif isstr(x) or isinstance(x, np.str):
        x = str(x)  # convert unicode & numpy strings
        try:
            return int(x)
        except:
            try:
                return float(x)
            except:
                if x == 'True':
                    return True
                elif x == 'False':
                    return False
                else:
                    return x
    else:
        return x


def create_hpx_disk_region_string(skyDir, coordsys, radius, inclusive=0):
    """
    """
    # Make an all-sky region
    if radius >= 90.:
        return None

    if coordsys == "GAL":
        xref = skyDir.galactic.l.deg
        yref = skyDir.galactic.b.deg
    elif coordsys == "CEL":
        xref = skyDir.ra.deg
        yref = skyDir.dec.deg
    else:
        raise Exception("Unrecognized coordinate system %s" % coordsys)

    if inclusive:
        val = "DISK_INC(%.3f,%.3f,%.3f,%i)" % (xref, yref, radius, inclusive)
    else:
        val = "DISK(%.3f,%.3f,%.3f)" % (xref, yref, radius)
    return val


def convolve2d_disk(fn, r, sig, nstep=200):
    """Evaluate the convolution f'(r) = f(r) * g(r) where f(r) is
    azimuthally symmetric function in two dimensions and g is a
    step function given by:

    g(r) = H(1-r/s)

    Parameters
    ----------

    fn : function
      Input function that takes a single radial coordinate parameter.

    r :  `~numpy.ndarray`
      Array of points at which the convolution is to be evaluated.

    sig : float
      Radius parameter of the step function.

    nstep : int
      Number of sampling point for numeric integration.
    """

    r = np.array(r, ndmin=1)
    sig = np.array(sig, ndmin=1)

    rmin = r - sig
    rmax = r + sig
    rmin[rmin < 0] = 0
    delta = (rmax - rmin) / nstep

    redge = rmin[..., np.newaxis] + \
        delta[..., np.newaxis] * np.linspace(0, nstep, nstep + 1)
    rp = 0.5 * (redge[..., 1:] + redge[..., :-1])
    dr = redge[..., 1:] - redge[..., :-1]
    fnv = fn(rp)

    r = r.reshape(r.shape + (1,))
    cphi = -np.ones(dr.shape)
    m = ((rp + r) / sig < 1) | (r == 0)

    rrp = r * rp
    sx = r ** 2 + rp ** 2 - sig ** 2
    cphi[~m] = sx[~m] / (2 * rrp[~m])
    dphi = 2 * np.arccos(cphi)
    v = rp * fnv * dphi * dr / (np.pi * sig * sig)
    s = np.sum(v, axis=-1)

    return s


def convolve2d_gauss(fn, r, sig, nstep=200):
    """Evaluate the convolution f'(r) = f(r) * g(r) where f(r) is
    azimuthally symmetric function in two dimensions and g is a
    2D gaussian with standard deviation s given by:

    g(r) = 1/(2*pi*s^2) Exp[-r^2/(2*s^2)]

    Parameters
    ----------

    fn : function
      Input function that takes a single radial coordinate parameter.

    r :  `~numpy.ndarray`
      Array of points at which the convolution is to be evaluated.

    sig : float
      Width parameter of the gaussian.

    nstep : int
      Number of sampling point for numeric integration.

    """
    r = np.array(r, ndmin=1)
    sig = np.array(sig, ndmin=1)

    rmin = r - 10 * sig
    rmax = r + 10 * sig
    rmin[rmin < 0] = 0
    delta = (rmax - rmin) / nstep

    redge = (rmin[..., np.newaxis] +
             delta[..., np.newaxis] *
             np.linspace(0, nstep, nstep + 1))

    rp = 0.5 * (redge[..., 1:] + redge[..., :-1])
    dr = redge[..., 1:] - redge[..., :-1]
    fnv = fn(rp)

    r = r.reshape(r.shape + (1,))
    sig2 = sig * sig
    x = r * rp / (sig2)

    if 'je_fn' not in convolve2d_gauss.__dict__:
        t = 10 ** np.linspace(-8, 8, 1000)
        t = np.insert(t, 0, [0])
        je = special.ive(0, t)
        convolve2d_gauss.je_fn = UnivariateSpline(t, je, k=2, s=0)

    je = convolve2d_gauss.je_fn(x.flat).reshape(x.shape)
    #je2 = special.ive(0,x)
    v = (rp * fnv / (sig2) * je * np.exp(x - (r * r + rp * rp) /
                                         (2 * sig2)) * dr)
    s = np.sum(v, axis=-1)

    return s


def make_pixel_distance(shape, xpix=None, ypix=None):
    """Fill a 2D array with dimensions `shape` with the distance of each
    pixel from a reference direction (xpix,ypix) in pixel coordinates.
    Pixel coordinates are defined such that (0,0) is located at the
    center of the corner pixel.

    """
    if np.isscalar(shape):
        shape = [shape, shape]

    if xpix is None:
        xpix = (shape[1] - 1.0) / 2.

    if ypix is None:
        ypix = (shape[0] - 1.0) / 2.

    dx = np.linspace(0, shape[1] - 1, shape[1]) - xpix
    dy = np.linspace(0, shape[0] - 1, shape[0]) - ypix
    dxy = np.zeros(shape)
    dxy += np.sqrt(dx[np.newaxis, :] ** 2 + dy[:, np.newaxis] ** 2)

    return dxy


def make_gaussian_kernel(sigma, npix=501, cdelt=0.01, xpix=None, ypix=None):
    """Make kernel for a 2D gaussian.

    Parameters
    ----------

    sigma : float
      Standard deviation in degrees.
    """

    sigma /= cdelt

    def fn(t, s): return 1. / (2 * np.pi * s ** 2) * np.exp(
        -t ** 2 / (s ** 2 * 2.0))
    dxy = make_pixel_distance(npix, xpix, ypix)
    k = fn(dxy, sigma)
    k /= (np.sum(k) * np.radians(cdelt) ** 2)

    return k


def make_disk_kernel(radius, npix=501, cdelt=0.01, xpix=None, ypix=None):
    """Make kernel for a 2D disk.

    Parameters
    ----------

    radius : float
      Disk radius in deg.
    """

    radius /= cdelt

    def fn(t, s): return 0.5 * (np.sign(s - t) + 1.0)

    dxy = make_pixel_distance(npix, xpix, ypix)
    k = fn(dxy, radius)
    k /= (np.sum(k) * np.radians(cdelt) ** 2)

    return k


def make_cdisk_kernel(psf, sigma, npix, cdelt, xpix, ypix, psf_scale_fn=None,
                      normalize=False):
    """Make a kernel for a PSF-convolved 2D disk.

    Parameters
    ----------

    psf : `~fermipy.irfs.PSFModel`

    sigma : float
      68% containment radius in degrees.
    """

    sigma /= 0.8246211251235321

    dtheta = psf.dtheta
    egy = psf.energies

    x = make_pixel_distance(npix, xpix, ypix)
    x *= cdelt

    k = np.zeros((len(egy), npix, npix))
    for i in range(len(egy)):
        def fn(t): return psf.eval(i, t, scale_fn=psf_scale_fn)
        psfc = convolve2d_disk(fn, dtheta, sigma)
        k[i] = np.interp(np.ravel(x), dtheta, psfc).reshape(x.shape)

    if normalize:
        k /= (np.sum(k, axis=0)[np.newaxis, ...] * np.radians(cdelt) ** 2)

    return k


def make_cgauss_kernel(psf, sigma, npix, cdelt, xpix, ypix, psf_scale_fn=None,
                       normalize=False):
    """Make a kernel for a PSF-convolved 2D gaussian.

    Parameters
    ----------

    psf : `~fermipy.irfs.PSFModel`

    sigma : float
      68% containment radius in degrees.
    """

    sigma /= 1.5095921854516636

    dtheta = psf.dtheta
    egy = psf.energies

    x = make_pixel_distance(npix, xpix, ypix)
    x *= cdelt

    k = np.zeros((len(egy), npix, npix))
    for i in range(len(egy)):
        def fn(t): return psf.eval(i, t, scale_fn=psf_scale_fn)
        psfc = convolve2d_gauss(fn, dtheta, sigma)
        k[i] = np.interp(np.ravel(x), dtheta, psfc).reshape(x.shape)

    if normalize:
        k /= (np.sum(k, axis=0)[np.newaxis, ...] * np.radians(cdelt) ** 2)

    return k


def memoize(obj):
    obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in obj.cache:
            obj.cache = {}
            obj.cache[key] = obj(*args, **kwargs)
        return obj.cache[key]
    return memoizer


def make_radial_kernel(psf, fn, sigma, npix, cdelt, xpix, ypix, psf_scale_fn=None,
                       normalize=False, klims=None, sparse=False):
    """Make a kernel for a general radially symmetric 2D function.

    Parameters
    ----------

    psf : `~fermipy.irfs.PSFModel`

    fn : callable
        Function that evaluates the kernel at a radial coordinate r.

    sigma : float
        68% containment radius in degrees.
    """

    if klims is None:
        egy = psf.energies
    else:
        egy = psf.energies[klims[0]:klims[1] + 1]
    ang_dist = make_pixel_distance(npix, xpix, ypix) * cdelt
    max_ang_dist = np.max(ang_dist) + cdelt
    #dtheta = np.linspace(0.0, (np.max(ang_dist) * 1.05)**0.5, 200)**2.0
    # z = create_kernel_function_lookup(psf, fn, sigma, egy,
    #                                  dtheta, psf_scale_fn)

    shape = (len(egy), npix, npix)
    k = np.zeros(shape)

    r99 = psf.containment_angle(energies=egy, fraction=0.997)
    r34 = psf.containment_angle(energies=egy, fraction=0.34)

    rmin = np.maximum(r34 / 4., 0.01)
    rmax = np.maximum(r99, 0.1)
    if sigma is not None:
        rmin = np.maximum(rmin, 0.5 * sigma)
        rmax = np.maximum(rmax, 2.0 * r34 + 3.0 * sigma)
    rmax = np.minimum(rmax, max_ang_dist)

    for i in range(len(egy)):

        rebin = min(int(np.ceil(cdelt / rmin[i])), 8)
        if sparse:
            dtheta = np.linspace(0.0, rmax[i]**0.5, 100)**2.0
        else:
            dtheta = np.linspace(0.0, max_ang_dist**0.5, 200)**2.0

        z = eval_radial_kernel(psf, fn, sigma, i, dtheta, psf_scale_fn)
        xdist = make_pixel_distance(npix * rebin,
                                    xpix * rebin + (rebin - 1.0) / 2.,
                                    ypix * rebin + (rebin - 1.0) / 2.)
        xdist *= cdelt / float(rebin)
        #x = val_to_pix(dtheta, np.ravel(xdist))

        if sparse:
            m = np.ravel(xdist) < rmax[i]
            kk = np.zeros(xdist.size)
            #kk[m] = map_coordinates(z, [x[m]], order=2, prefilter=False)
            kk[m] = np.interp(np.ravel(xdist)[m], dtheta, z)
            kk = kk.reshape(xdist.shape)
        else:
            kk = np.interp(np.ravel(xdist), dtheta, z).reshape(xdist.shape)
            # kk = map_coordinates(z, [x], order=2,
            #                     prefilter=False).reshape(xdist.shape)

        if rebin > 1:
            kk = sum_bins(kk, 0, rebin)
            kk = sum_bins(kk, 1, rebin)

        k[i] = kk / float(rebin)**2

    k = k.reshape((len(egy),) + ang_dist.shape)
    if normalize:
        k /= (np.sum(k, axis=0)[np.newaxis, ...] * np.radians(cdelt) ** 2)

    return k


def eval_radial_kernel(psf, fn, sigma, idx, dtheta, psf_scale_fn):

    if fn is None:
        return psf.eval(idx, dtheta, scale_fn=psf_scale_fn)
    else:
        return fn(lambda t: psf.eval(idx, t, scale_fn=psf_scale_fn),
                  dtheta, sigma)


#@memoize
def create_kernel_function_lookup(psf, fn, sigma, egy, dtheta, psf_scale_fn):

    z = np.zeros((len(egy), len(dtheta)))
    for i in range(len(egy)):

        if fn is None:
            z[i] = psf.eval(i, dtheta, scale_fn=psf_scale_fn)
        else:
            z[i] = fn(lambda t: psf.eval(i, t, scale_fn=psf_scale_fn),
                      dtheta, sigma)

    return z


def create_radial_spline(psf, fn, sigma, egy, dtheta, psf_scale_fn):

    from scipy.ndimage.interpolation import spline_filter

    z = create_kernel_function_lookup(
        psf, fn, sigma, egy, dtheta, psf_scale_fn)
    sp = []
    for i in range(z.shape[0]):
        sp += [spline_filter(z[i], order=2)]
    return sp


def make_psf_kernel(psf, npix, cdelt, xpix, ypix, psf_scale_fn=None, normalize=False):
    """
    Generate a kernel for a point-source.

    Parameters
    ----------

    psf : `~fermipy.irfs.PSFModel`

    npix : int
        Number of pixels in X and Y dimensions.

    cdelt : float
        Pixel size in degrees.

    """

    egy = psf.energies
    x = make_pixel_distance(npix, xpix, ypix)
    x *= cdelt

    k = np.zeros((len(egy), npix, npix))
    for i in range(len(egy)):
        k[i] = psf.eval(i, x, scale_fn=psf_scale_fn)

    if normalize:
        k /= (np.sum(k, axis=0)[np.newaxis, ...] * np.radians(cdelt) ** 2)

    return k


def rebin_map(k, nebin, npix, rebin):
    if rebin > 1:
        k = np.sum(k.reshape((nebin, npix * rebin, npix, rebin)), axis=3)
        k = k.swapaxes(1, 2)
        k = np.sum(k.reshape(nebin, npix, npix, rebin), axis=3)
        k = k.swapaxes(1, 2)

    k /= rebin ** 2

    return k


def sum_bins(x, dim, npts):
    if npts <= 1:
        return x
    shape = x.shape[:dim] + (int(x.shape[dim] / npts),
                             npts) + x.shape[dim + 1:]
    return np.sum(x.reshape(shape), axis=dim + 1)


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
                               min(large_shape - edge_min,
                                   edge_max - edge_min))
                         for (edge_min, edge_max, large_shape) in
                         zip(edges_min, edges_max, large_array_shape))

    return slices_large, slices_small
