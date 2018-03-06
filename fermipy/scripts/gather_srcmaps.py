# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import os
import glob
import argparse
from astropy.io import fits


def do_gather(flist):
    """ Gather all the HDUs from a list of files"""
    hlist = []
    nskip = 3
    for fname in flist:
        fin = fits.open(fname)
        if len(hlist) == 0:
            if fin[1].name == 'SKYMAP':
                nskip = 4
            start = 0
        else:
            start = nskip
        for h in fin[start:]:
            hlist.append(h)
    hdulistout = fits.HDUList(hlist)
    return hdulistout
        
                
def main():
    """ Main function for command line usage """
    usage = "usage: %(prog)s [options] "
    description = "Gather source maps from Fermi-LAT files."

    parser = argparse.ArgumentParser(usage=usage, description=description)

    parser.add_argument('-o', '--output', default=None, type=str,
                        help='Output file.')
    parser.add_argument('--clobber', default=False, action='store_true',
                        help='Overwrite output file.')
    parser.add_argument('--gzip', action='store_true', 
                        help='Compress output file')
    parser.add_argument('--rm', action='store_true', 
                        help='Remove input files.')
    parser.add_argument('files', nargs='+', default=None,
                        help='List of input files.')

    args = parser.parse_args()
    
    hdulistout = do_gather(args.files)
    
    if args.output:
        hdulistout.writeto(args.output, clobber=args.clobber)

        if args.gzip:
            os.system('gzip -9 %s'%args.output)
        
        if args.rm:
            for farg in args.files:
                flist = glob.glob(farg)
            for ffound in flist:
                os.path.unlink(ffound)



if __name__ == '__main__':
    main()
