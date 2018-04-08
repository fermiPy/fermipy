# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import numpy as np
import scipy
from scipy.ndimage.filters import maximum_filter
from astropy.coordinates import SkyCoord
from fermipy import utils
from fermipy import wcs_utils
from fermipy.utils import get_region_mask


def fit_error_ellipse(tsmap, xy=None, dpix=3, zmin=None):
    """Fit a positional uncertainty ellipse from a TS map.  The fit
    will be performed over pixels in the vicinity of the peak pixel
    with D < dpix OR z > zmin where D is the distance from the peak
    pixel in pixel coordinates and z is the difference in amplitude
    from the peak pixel.

    Parameters
    ----------
    tsmap : `~gammapy.maps.WcsMap`

    xy : tuple

    dpix : float

    zmin : float

    Returns
    -------
    fit : dict
        Dictionary with fit results.
    """

    if xy is None:
        ix, iy = np.unravel_index(np.argmax(tsmap.data.T),
                                  tsmap.data.T.shape)
    else:
        ix, iy = xy

    pbfit0 = utils.fit_parabola(tsmap.data.T, ix, iy, dpix=1.5)
    pbfit1 = utils.fit_parabola(tsmap.data.T, ix, iy, dpix=dpix,
                                zmin=zmin)

    wcs = tsmap.geom.wcs
    cdelt0 = tsmap.geom.wcs.wcs.cdelt[0]
    cdelt1 = tsmap.geom.wcs.wcs.cdelt[1]
    npix0 = tsmap.data.T.shape[0]
    npix1 = tsmap.data.T.shape[1]

    o = {}
    o['fit_success'] = pbfit0['fit_success']
    o['fit_inbounds'] = True

    if pbfit0['fit_success']:
        o['xpix'] = pbfit0['x0']
        o['ypix'] = pbfit0['y0']
        o['zoffset'] = pbfit0['z0']
    else:
        o['xpix'] = float(ix)
        o['ypix'] = float(iy)
        o['zoffset'] = tsmap.data.T[ix, iy]

    if pbfit1['fit_success']:
        sigmax = 2.0**0.5 * pbfit1['sigmax'] * np.abs(cdelt0)
        sigmay = 2.0**0.5 * pbfit1['sigmay'] * np.abs(cdelt1)
        theta = pbfit1['theta']
        sigmax = min(sigmax, np.abs(2.0 * npix0 * cdelt0))
        sigmay = min(sigmay, np.abs(2.0 * npix1 * cdelt1))
    elif pbfit0['fit_success']:
        sigmax = 2.0**0.5 * pbfit0['sigmax'] * np.abs(cdelt0)
        sigmay = 2.0**0.5 * pbfit0['sigmay'] * np.abs(cdelt1)
        theta = pbfit0['theta']
        sigmax = min(sigmax, np.abs(2.0 * npix0 * cdelt0))
        sigmay = min(sigmay, np.abs(2.0 * npix1 * cdelt1))
    else:
        pix_area = np.abs(cdelt0) * np.abs(cdelt1)
        mask = get_region_mask(tsmap.data, 1.0, (ix, iy))
        area = np.sum(mask) * pix_area
        sigmax = (area / np.pi)**0.5
        sigmay = (area / np.pi)**0.5
        theta = 0.0

    if (o['xpix'] <= 0 or o['xpix'] >= npix0 - 1 or
            o['ypix'] <= 0 or o['ypix'] >= npix1 - 1):
        o['fit_inbounds'] = False
        o['xpix'] = float(ix)
        o['ypix'] = float(iy)

    o['peak_offset'] = np.sqrt((float(ix) - o['xpix'])**2 +
                               (float(iy) - o['ypix'])**2)

    skydir = SkyCoord.from_pixel(o['xpix'], o['ypix'], wcs)
    sigma = (sigmax * sigmay)**0.5
    r68 = 2.30**0.5 * sigma
    r95 = 5.99**0.5 * sigma
    r99 = 9.21**0.5 * sigma

    if sigmax < sigmay:
        o['pos_err_semimajor'] = sigmay
        o['pos_err_semiminor'] = sigmax
        o['theta'] = np.fmod(2 * np.pi + np.pi / 2. + theta, np.pi)
    else:
        o['pos_err_semimajor'] = sigmax
        o['pos_err_semiminor'] = sigmay
        o['theta'] = np.fmod(2 * np.pi + theta, np.pi)

    o['pos_angle'] = np.degrees(o['theta'])
    o['pos_err'] = sigma
    o['pos_r68'] = r68
    o['pos_r95'] = r95
    o['pos_r99'] = r99
    o['ra'] = skydir.icrs.ra.deg
    o['dec'] = skydir.icrs.dec.deg
    o['glon'] = skydir.galactic.l.deg
    o['glat'] = skydir.galactic.b.deg
    a = o['pos_err_semimajor']
    b = o['pos_err_semiminor']

    o['pos_ecc'] = np.sqrt(1 - b**2 / a**2)
    o['pos_ecc2'] = np.sqrt(a**2 / b**2 - 1)
    o['skydir'] = skydir

    if tsmap.geom.coordsys == 'GAL':
        gal_cov = utils.ellipse_to_cov(o['pos_err_semimajor'],
                                       o['pos_err_semiminor'],
                                       o['theta'])
        theta_cel = wcs_utils.get_cel_to_gal_angle(skydir)
        cel_cov = utils.ellipse_to_cov(o['pos_err_semimajor'],
                                       o['pos_err_semiminor'],
                                       o['theta'] + theta_cel)

    else:
        cel_cov = utils.ellipse_to_cov(o['pos_err_semimajor'],
                                       o['pos_err_semiminor'],
                                       o['theta'])
        theta_gal = 2 * np.pi - wcs_utils.get_cel_to_gal_angle(skydir)
        gal_cov = utils.ellipse_to_cov(o['pos_err_semimajor'],
                                       o['pos_err_semiminor'],
                                       o['theta'] + theta_gal)

    o['pos_gal_cov'] = gal_cov
    o['pos_cel_cov'] = cel_cov
    o['pos_gal_corr'] = utils.cov_to_correlation(gal_cov)
    o['pos_cel_corr'] = utils.cov_to_correlation(cel_cov)
    o['glon_err'], o['glat_err'] = np.sqrt(
        gal_cov[0, 0]), np.sqrt(gal_cov[1, 1])
    o['ra_err'], o['dec_err'] = np.sqrt(cel_cov[0, 0]), np.sqrt(cel_cov[1, 1])

    return o


def find_peaks(input_map, threshold, min_separation=0.5):
    """Find peaks in a 2-D map object that have amplitude larger than
    `threshold` and lie a distance at least `min_separation` from another
    peak of larger amplitude.  The implementation of this method uses
    `~scipy.ndimage.filters.maximum_filter`.

    Parameters
    ----------
    input_map : `~gammapy.maps.WcsMap`

    threshold : float

    min_separation : float
       Radius of region size in degrees.  Sets the minimum allowable
       separation between peaks.

    Returns
    -------
    peaks : list
       List of dictionaries containing the location and amplitude of
       each peak.
    """

    data = input_map.data

    cdelt = max(input_map.geom.wcs.wcs.cdelt)
    min_separation = max(min_separation, 2 * cdelt)

    region_size_pix = int(min_separation / cdelt)
    region_size_pix = max(3, region_size_pix)

    deltaxy = utils.make_pixel_distance(region_size_pix * 2 + 3)
    deltaxy *= max(input_map.geom.wcs.wcs.cdelt)
    region = deltaxy < min_separation

    local_max = maximum_filter(data, footprint=region) == data
    local_max[data < threshold] = False

    labeled, num_objects = scipy.ndimage.label(local_max)
    slices = scipy.ndimage.find_objects(labeled)

    peaks = []
    for s in slices:
        skydir = SkyCoord.from_pixel(s[1].start, s[0].start,
                                     input_map.geom.wcs)
        peaks.append({'ix': s[1].start,
                      'iy': s[0].start,
                      'skydir': skydir,
                      'amp': data[s[0].start, s[1].start]})

    return sorted(peaks, key=lambda t: t['amp'], reverse=True)


def estimate_pos_and_err_parabolic(tsvals):
    """Solve for the position and uncertainty of source in one dimension
         assuming that you are near the maximum and the errors are parabolic

    Parameters
    ----------
    tsvals  :  `~numpy.ndarray`
       The TS values at the maximum TS, and for each pixel on either side

    Returns
    -------
    The position and uncertainty of the source, in pixel units
    w.r.t. the center of the maximum pixel

    """
    a = tsvals[2] - tsvals[0]
    bc = 2. * tsvals[1] - tsvals[0] - tsvals[2]
    s = a / (2 * bc)
    err = np.sqrt(2 / bc)
    return s, err


def refine_peak(tsmap, pix):
    """Solve for the position and uncertainty of source assuming that you
    are near the maximum and the errors are parabolic

    Parameters
    ----------
    tsmap : `~numpy.ndarray`
       Array with the TS data.

    Returns
    -------
    The position and uncertainty of the source, in pixel units
    w.r.t. the center of the maximum pixel

    """
    # Note the annoying WCS convention
    nx = tsmap.shape[1]
    ny = tsmap.shape[0]

    if pix[0] == 0 or pix[0] == (nx - 1):
        xval = float(pix[0])
        xerr = -1
    else:
        x_arr = tsmap[pix[1], pix[0] - 1:pix[0] + 2]
        xval, xerr = estimate_pos_and_err_parabolic(x_arr)
        xval += float(pix[0])

    if pix[1] == 0 or pix[1] == (ny - 1):
        yval = float(pix[1])
        yerr = -1
    else:
        y_arr = tsmap[pix[1] - 1:pix[1] + 2, pix[0]]
        yval, yerr = estimate_pos_and_err_parabolic(y_arr)
        yval += float(pix[1])

    return (xval, yval), (xerr, yerr)
