# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import os
import glob
import argparse
from fermipy import merge_utils


def main():
    """ Main function for command line usage """
    usage = "usage: %(prog)s [options] "
    description = "Merge a set of Fermi-LAT files."

    parser = argparse.ArgumentParser(usage=usage, description=description)

    parser.add_argument('-o', '--output', default=None, type=str,
                        help='Output file.')
    parser.add_argument('--clobber', default=False, action='store_true',
                        help='Overwrite output file.')
    parser.add_argument('--hdu', default=None, type=str,
                        help='HDU name.')
    parser.add_argument('--gzip', action='store_true',
                        help='Compress output file')
    parser.add_argument('--rm', action='store_true',
                        help='Remove input files.')
    parser.add_argument('files', nargs='+', default=None,
                        help='List of input files.')

    args = parser.parse_args()

    hpx_map = merge_utils.stack_energy_planes_hpx(args.files, hdu=args.hdu)

    if args.output:
        hpx_map.hpx.write_fits(hpx_map.counts, args.output,
                               extname=args.hdu, clobber=args.clobber)

        if args.gzip:
            os.system('gzip -9 %s' % args.output)

        if args.rm:
            for farg in args.files:
                flist = glob.glob(farg)
                for ffound in flist:
                    os.path.unlink(ffound)


if __name__ == '__main__':
    main()
