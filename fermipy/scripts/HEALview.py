#!/usr/bin/env python
#

""" View a ST produced HEALPix map with matplotlib
"""

__facility__ = "HEALview.py"
__abstract__ = __doc__
__author__    = "E. Charles"
__date__      = "$Date: 2015/05/06 21:20:31 $"
__version__   = "$Revision: 1.4 $, $Author: echarles $"
__release__   = "$Name:  $"


import sys
import os
import numpy 
import astropy.io.fits as pf

from fermipy.hpx_utils import HPX, HpxToWcsMapping
from fermipy.skymap import HpxMap
from fermipy.plotting import ImagePlotter
import matplotlib.pyplot as plt

def main():

    import sys
    import argparse

    # Argument defintion
    usage = "usage: %(prog)s [options]" 
    description = "Collect all the new source"

    parser = argparse.ArgumentParser(usage,description=__abstract__)

    parser.add_argument("-i", "--input",type=argparse.FileType('r'),required=True,
                        help="Input file")

    parser.add_argument("-e", "--extension",type=str,default="SKYMAP",
                        help="FITS HDU with HEALPix map")
 
    parser.add_argument("--ebin",type=str,default=None,
                        help="Energy bin, integer or 'ALL'")
    
    parser.add_argument("-o", "--output",type=argparse.FileType('w'),
                        help="Output file.  Leave blank for interactive.")
    
    # Parse the command line
    args = parser.parse_args(sys.argv[1:])

    # Get the model 
    f = pf.open(args.input.name)
    # We need a better check
    maptype = "None"

    model_hdu = f[args.extension]
        
    hpxmap = HpxMap.create_from_hdulist(f,extname=args.extension,ebounds="EBOUNDS")
    outdata = []
     
    if args.ebin == "ALL":
        wcsproj = hpxmap.hpx.make_wcs(naxis=2,proj='AIT',energies=None,oversample=2)
        mapping = HpxToWcsMapping(hpxmap.hpx,wcsproj)
        
        for i,data in enumerate(hpxmap.counts):
            ip =  ImagePlotter(data=data,proj=hpxmap.hpx,mapping=mapping)  
            fig = plt.figure(i)
            im,ax = ip.plot(zscale='log')
            outdata.append(fig)

    elif args.ebin is None:
        ip =  ImagePlotter(data=hpxmap.counts,proj=hpxmap.hpx)  
        im,ax = ip.plot(zscale='log')
        outdata.append((im,ax))        
    else:
        try:
            ibin = int(args.ebin)
            ip =  ImagePlotter(data=hpxmap.counts[ibin],proj=hpxmap.hpx)  
            im,ax = ip.plot(zscale='log')
            outdata.append((im,ax))        
        except:
            print("--ebin argument must be an integer or 'ALL'")

    if args.output is None:
        plt.show()
    else:
        plt.savefig(args.output.name)


if __name__ == "__main__":
    main()
