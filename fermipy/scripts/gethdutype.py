#!/usr/bin/env python
#

""" Identify the type of image stored in an HDU
"""

__facility__ = "gethdutype.py"
__abstract__ = __doc__
__author__    = "E. Charles"
__date__      = "$Date: 2015/05/06 21:20:31 $"
__version__   = "$Revision: 1.4 $, $Author: echarles $"
__release__   = "$Name:  $"


import sys
import argparse

import os
import numpy 
from astropy.io import fits

def tryprint(header, key):
    try:
        print ("%s = %s"%(key,header[key]))
    except KeyError:
        print ("No key %s"%(key))

def gethdutype(hdu):
    if hdu.is_image:
        naxis = hdu.header['NAXIS']
        print("WCS Image, naxis = %i"%(naxis))
        return
    
    header = hdu.header
    try: 
        pixtype = header['PIXTYPE']
    except KeyError:
        print ("Unknown image type, PIXTYPE keyword is absent")
        return
    if pixtype != "HEALPIX":
        print ("Unknown image type PIXTYPE = %s"%pixtype)

    tryprint(header, "HPX_CONV")
    tryprint(header, "INDXSCHM")
    tryprint(header, "COORDSYS")
    tryprint(header, "NSIDE")
    tryprint(header, "ORDERING")



def main():

    # Argument defintion
    usage = "usage: %(prog)s [options]" 
    description = "Identify the type of image stored in an HDU"

    parser = argparse.ArgumentParser(usage,description=__abstract__)

    parser.add_argument("-i", "--input",type=argparse.FileType('r'),required=True,
                        help="Input file")

    parser.add_argument("--hdu", type=str, default=None,
                        help="FITS HDU with map")

    # Parse the command line
    args = parser.parse_args(sys.argv[1:])

    f = fits.open(args.input.name)
    if args.hdu is None:
        hdu = f[0]
    else:
        hdu = f[args.hdu]

    gethdutype(hdu)

if __name__ == "__main__":
    main()
