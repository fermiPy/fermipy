#!/usr/bin/env python
#

""" View a ST produced wcs map with matplotlib
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

from fermipy.utils import init_matplotlib_backend
init_matplotlib_backend()

from gammapy.maps import Map

from fermipy.plotting import ImagePlotter
import matplotlib.pyplot as plt

def main():

    import sys
    import argparse

    # Argument defintion
    usage = "usage: %(prog)s [options]" 
    description = "Plot a WCS-based image"

    parser = argparse.ArgumentParser(usage,description=__abstract__)

    parser.add_argument("-i", "--input",type=argparse.FileType('r'),required=True,
                        help="Input file")

    parser.add_argument("-e", "--extension", type=str, default=None,
                        help="FITS HDU with map")
 
    parser.add_argument("--ebin",type=str,default=None,
                        help="Energy bin, integer or 'ALL'")

    parser.add_argument("--zscale",type=str, default='log',
                        help="Scaling for color scale")

    parser.add_argument("--zmin",type=float, default=None,
                        help="Minimum z-axis value")
    
    parser.add_argument("--zmax",type=float, default=None,
                        help="Maximum z-axis value")
    
    parser.add_argument("-o", "--output",type=argparse.FileType('w'),
                        help="Output file.  Leave blank for interactive.")
    
    

    # Parse the command line
    args = parser.parse_args(sys.argv[1:])

    # Get the map 
    themap = Map.read(args.input.name)

    outdata = []

    if args.ebin == "ALL":
        for i,data in enumerate(counts):
            ip =  ImagePlotter(themap)
            fig = plt.figure(i)
            im,ax = ip.plot(zscale=args.zscale, vmin=args.zmin, vmax=args.zmax)
            outdata.append(fig)

    elif args.ebin is None:
        ip =  ImagePlotter(themap.sum_over_axes())
        im,ax = ip.plot(zscale=args.zscale, vmin=args.zmin, vmax=args.zmax)
        outdata.append((im,ax))        
    else:
        try:
            ibin = int(args.ebin)
            ip =  ImagePlotter(themap)
            im,ax = ip.plot(zscale=args.zscale, vmin=args.zmin, vmax=args.zmax)
            outdata.append((im,ax))        
        except:
            raise ValueError("--ebin argument must be an integer or 'ALL'")


    if args.output is None:
        plt.show()
    else:
        if len(outdata) == 1:
            plt.savefig(args.output.name)
        else:
            base,ext = os.path.splitext(args.output.name)
            for i, fig in enumerate(outdata):
                fig.savefig("%s_%02i%s"%(base,i,ext))


if __name__ == "__main__":
    main()
