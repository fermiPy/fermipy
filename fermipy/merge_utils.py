# Licensed under a 3-clause BSD style license - see LICENSE.rst
""" Some utilities to merge various types of Fermi-LAT data files """

from __future__ import absolute_import, division, print_function

import sys
import argparse
import numpy as np
from astropy.io import fits


def update_null_primary(hdu_in, hdu=None):
    """ 'Update' a null primary HDU

    This actually just checks hdu exists and creates it from hdu_in if it does not.
    """
    if hdu is None:
        hdu = fits.PrimaryHDU(header=hdu_in.header)
    else:
        hdu = hdu_in
    return hdu


def update_primary(hdu_in, hdu=None):
    """ 'Update' a primary HDU

    This checks hdu exists and creates it from hdu_in if it does not.
    If hdu does exist, this adds the data in hdu_in to hdu
    """
    if hdu is None:
        hdu = fits.PrimaryHDU(data=hdu_in.data, header=hdu_in.header)
    else:
        hdu.data += hdu_in.data
    return hdu


def update_image(hdu_in, hdu=None):
    """ 'Update' an image HDU

    This checks hdu exists and creates it from hdu_in if it does not.
    If hdu does exist, this adds the data in hdu_in to hdu
    """
    if hdu is None:
        hdu = fits.ImageHDU(
            data=hdu_in.data, header=hdu_in.header, name=hdu_in.name)
    else:
        hdu.data += hdu_in.data
    return hdu


def update_ebounds(hdu_in, hdu=None):
    """ 'Update' the EBOUNDS HDU

    This checks hdu exists and creates it from hdu_in if it does not.
    If hdu does exist, this raises an exception if it doesn not match hdu_in
    """
    if hdu is None:
        hdu = fits.BinTableHDU(
            data=hdu_in.data, header=hdu_in.header, name=hdu_in.name)
    else:
        for col in ['CHANNEL', 'E_MIN', 'E_MAX']:
            if (hdu.data[col] != hdu_in.data[col]).any():
                raise ValueError("Energy bounds do not match : %s %s" %
                                 (hdu.data[col], hdu_in.data[col]))
    return hdu


def merge_all_gti_data(datalist_in, nrows, first):
    """ Merge together all the GTI data

    Parameters
    -------
    datalist_in : list of `astropy.io.fits.BinTableHDU` data
        The GTI data that is being merged

    nrows : `~numpy.ndarray` of ints
        Array with the number of nrows for each object in datalist_in

    first : `astropy.io.fits.BinTableHDU`
        BinTableHDU to use as a template

    Returns
    -------
    out_hdu : `astropy.io.fits.BinTableHDU`
        BinTableHDU with the merge GTIs

    """
    max_row = nrows.cumsum()
    min_row = max_row - nrows
    out_hdu = fits.BinTableHDU.from_columns(
        first.columns, header=first.header, nrows=nrows.sum())

    for (imin, imax, data_in) in zip(min_row, max_row, datalist_in):
        for col in first.columns:
            out_hdu.data[col.name][imin:imax] = data_in[col.name]

    return out_hdu


def extract_gti_data(hdu_in):
    """ Extract some GTI related data

    Parameters
    -------
    hdu_in : `astropy.io.fits.BinTableHDU`
        The GTI data

    Returns
    -------
    data : `astropy.io.fits.BinTableHDU` data

    exposure : float
        Exposure value taken from FITS header

    tstop : float
        TSTOP value taken from FITS header

    """
    data = hdu_in.data
    exposure = hdu_in.header['EXPOSURE']
    tstop = hdu_in.header['TSTOP']
    return (data, exposure, tstop)


def update_hpx_skymap_allsky(hdu_in, hdu):
    """ 'Update' a HEALPix skymap

    This checks hdu exists and creates it from hdu_in if it does not.
    If hdu does exist, this adds the data in hdu_in to hdu
    """
    if hdu is None:
        hdu = fits.BinTableHDU(
            data=hdu_in.data, header=hdu_in.header, name=hdu_in.name)
    else:
        for col in hdu.columns:
            hdu.data[col.name] += hdu_in.data[col.name]
    return hdu


def merge_wcs_counts_cubes(filelist):
    """ Merge all the files in filelist, assuming that they WCS counts cubes
    """
    out_prim = None
    out_ebounds = None

    datalist_gti = []
    exposure_sum = 0.
    nfiles = len(filelist)
    ngti = np.zeros(nfiles, int)

    for i, filename in enumerate(filelist):
        fin = fits.open(filename)
        sys.stdout.write('.')
        sys.stdout.flush()
        out_prim = update_primary(fin[0], out_prim)
        out_ebounds = update_ebounds(fin["EBOUNDS"], out_ebounds)
        (gti_data, exposure, tstop) = extract_gti_data(fin["GTI"])
        datalist_gti.append(gti_data)
        exposure_sum += exposure
        ngti[i] = len(gti_data)
        if i == 0:
            first = fin
        elif i == nfiles - 1:
            date_end = fin[0].header['DATE-END']
        else:
            fin.close()

    out_gti = merge_all_gti_data(datalist_gti, ngti, first['GTI'])
    out_gti.header['EXPOSURE'] = exposure_sum
    out_gti.header['TSTOP'] = tstop

    hdulist = [out_prim, out_ebounds, out_gti]
    for hdu in hdulist:
        hdu.header['DATE-END'] = date_end

    out_prim.update_header()
    return fits.HDUList(hdulist)


def merge_hpx_counts_cubes(filelist):
    """ Merge all the files in filelist, assuming that they HEALPix counts cubes
    """
    out_prim = None
    out_skymap = None
    out_ebounds = None

    datalist_gti = []
    exposure_sum = 0.
    nfiles = len(filelist)
    ngti = np.zeros(nfiles, int)

    for i, filename in enumerate(filelist):
        fin = fits.open(filename)
        sys.stdout.write('.')
        sys.stdout.flush()
        out_prim = update_null_primary(fin[0], out_prim)
        out_skymap = update_hpx_skymap_allsky(fin["SKYMAP"], out_skymap)
        out_ebounds = update_ebounds(fin["EBOUNDS"], out_ebounds)
        (gti_data, exposure, tstop) = extract_gti_data(fin["GTI"])
        datalist_gti.append(gti_data)
        exposure_sum += exposure
        ngti[i] = len(gti_data)
        if i == 0:
            first = fin
        elif i == nfiles - 1:
            date_end = fin[0].header['DATE-END']
        else:
            fin.close()

    out_gti = merge_all_gti_data(datalist_gti, ngti, first['GTI'])
    out_gti.header['EXPOSURE'] = exposure_sum
    out_gti.header['TSTOP'] = tstop

    hdulist = [out_prim, out_skymap, out_ebounds, out_gti]
    for hdu in hdulist:
        hdu.header['DATE-END'] = date_end
    out_prim.update_header()

    return fits.HDUList(hdulist)


def main():
    """ Main function for command line usage """
    usage = "usage: %(prog)s [options] "
    description = "Merge a set of Fermi-LAT files."
    parser = argparse.ArgumentParser(usage=usage, description=description)

    parser.add_argument('-o', '--output', default=None, type=str,
                        help='Output file.')
    parser.add_argument('--clobber', default=False, action='store_true',
                        help='Overwrite output file.')
    parser.add_argument('files', nargs='+', default=None,
                        help='List of input files.')

    args = parser.parse_args()

    hdulist = merge_hpx_counts_cubes(args.files)

    if args.output:
        hdulist.writeto(args.output, clobber=args.clobber)


if __name__ == '__main__':
    main()
