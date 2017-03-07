# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import numpy as np
import scipy
from scipy.ndimage.filters import maximum_filter
from astropy.coordinates import SkyCoord
from fermipy import utils


def fit_error_ellipse(tsmap, xy=None, dpix=3, zmin=None):
    """Fit a positional uncertainty ellipse from a TS map.  The fit
    will be performed over pixels in the vicinity of the peak pixel
    with D < dpix OR z > zmin where D is the distance from the peak
    pixel in pixel coordinates and z is the difference in amplitude
    from the peak pixel.

    Parameters
    ----------
    tsmap : `~fermipy.skymap.Map`

    xy : tuple

    dpix : float

    zmin : float
    """

    if xy is None:
        ix, iy = np.unravel_index(np.argmax(tsmap.counts.T),
                                  tsmap.counts.T.shape)
    else:
        ix, iy = xy

    pbfit = utils.fit_parabola(tsmap.counts.T, ix, iy, dpix=dpix,
                               zmin=zmin)

    wcs = tsmap.wcs
    cdelt0 = tsmap.wcs.wcs.cdelt[0]
    cdelt1 = tsmap.wcs.wcs.cdelt[1]
    npix0 = tsmap.counts.T.shape[0]
    npix1 = tsmap.counts.T.shape[1]



    o = {}
    o['fit_success'] = pbfit['fit_success']
    o['fit_inbounds'] = True

    if pbfit['fit_success']:
        sigmax = 2.0**0.5 * pbfit['sigmax'] * np.abs(cdelt0)
        sigmay = 2.0**0.5 * pbfit['sigmay'] * np.abs(cdelt1)
        theta = pbfit['theta']
        sigmax = min(sigmax, np.abs(2.0 * npix0 * cdelt0))
        sigmay = min(sigmay, np.abs(2.0 * npix1 * cdelt1))
        o['xpix'] = pbfit['x0']
        o['ypix'] = pbfit['y0']
        o['zoffset'] = pbfit['z0']
    else:
        sigmax = np.nan
        sigmay = np.nan
        theta = np.nan
        o['xpix'] = float(ix)
        o['ypix'] = float(iy)
        o['zoffset'] = tsmap.counts.T[ix,iy]
        
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
        o['sigma_semimajor'] = sigmay
        o['sigma_semiminor'] = sigmax
        o['theta'] = np.fmod(2 * np.pi + np.pi / 2. + theta, np.pi)
    else:
        o['sigma_semimajor'] = sigmax
        o['sigma_semiminor'] = sigmay
        o['theta'] = np.fmod(2 * np.pi + theta, np.pi)

    o['sigmax'] = sigmax
    o['sigmay'] = sigmay
    o['sigma'] = sigma
    o['r68'] = r68
    o['r95'] = r95
    o['r99'] = r99
    o['ra'] = skydir.icrs.ra.deg
    o['dec'] = skydir.icrs.dec.deg
    o['glon'] = skydir.galactic.l.deg
    o['glat'] = skydir.galactic.b.deg    
    a = o['sigma_semimajor']
    b = o['sigma_semiminor']

    o['eccentricity'] = np.sqrt(1 - b**2 / a**2)
    o['eccentricity2'] = np.sqrt(a**2 / b**2 - 1)

    return o, skydir


def find_peaks(input_map, threshold, min_separation=0.5):
    """Find peaks in a 2-D map object that have amplitude larger than
    `threshold` and lie a distance at least `min_separation` from another
    peak of larger amplitude.  The implementation of this method uses
    `~scipy.ndimage.filters.maximum_filter`.

    Parameters
    ----------
    input_map : `~fermipy.utils.Map`

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

    data = input_map.counts

    cdelt = max(input_map.wcs.wcs.cdelt)
    min_separation = max(min_separation, 2 * cdelt)

    region_size_pix = int(min_separation / cdelt)
    region_size_pix = max(3, region_size_pix)

    deltaxy = utils.make_pixel_distance(region_size_pix * 2 + 3)
    deltaxy *= max(input_map.wcs.wcs.cdelt)
    region = deltaxy < min_separation

    local_max = maximum_filter(data, footprint=region) == data
    local_max[data < threshold] = False

    labeled, num_objects = scipy.ndimage.label(local_max)
    slices = scipy.ndimage.find_objects(labeled)

    peaks = []
    for s in slices:
        skydir = SkyCoord.from_pixel(s[1].start, s[0].start,
                                     input_map.wcs)
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
