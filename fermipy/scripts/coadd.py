# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import argparse
from fermipy import merge_utils
from fermipy import fits_utils
from astropy.wcs import WCS
from fermipy.hpx_utils import HPX

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

    proj, f, hdu = fits_utils.read_projection_from_fits(args.files[0])
    if isinstance(proj, WCS):
        hdulist = merge_utils.merge_wcs_counts_cubes(args.files)
    elif isinstance(proj, HPX):
        hdulist = merge_utils.merge_hpx_counts_cubes(args.files)
    else:
        raise TypeError("Could not read projection from file %s"%args.files[0])

    if args.output:
        hdulist.writeto(args.output, clobber=args.clobber)


if __name__ == '__main__':
    main()

