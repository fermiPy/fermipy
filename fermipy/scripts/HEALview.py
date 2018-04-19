#!/usr/bin/env python
#

""" View a ST produced HEALPix map with matplotlib
"""

__facility__ = "HEALview.py"
__abstract__ = __doc__
__author__ = "E. Charles"
__date__ = "$Date: 2015/05/06 21:20:31 $"
__version__ = "$Revision: 1.4 $, $Author: echarles $"
__release__ = "$Name:  $"


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
    description = "Collect all the new source"

    parser = argparse.ArgumentParser(usage, description=__abstract__)

    parser.add_argument("-i", "--input", type=argparse.FileType('r'), required=True,
                        help="Input file")

    parser.add_argument("-e", "--extension", type=str, default="SKYMAP",
                        help="FITS HDU with HEALPix map")

    parser.add_argument("--ebin", type=str, default=None,
                        help="Energy bin, integer or 'ALL'")

    parser.add_argument("--zscale", type=str, default='log',
                        help="Scaling for color scale")

    parser.add_argument("--zmin", type=float, default=None,
                        help="Minimum z-axis value")

    parser.add_argument("--zmax", type=float, default=None,
                        help="Maximum z-axis value")

    parser.add_argument("--cbar", action='store_true', default=False, 
                        help="draw color bar")

    parser.add_argument("-o", "--output", type=argparse.FileType('w'),
                        help="Output file.  Leave blank for interactive.")

    # Parse the command line
    args = parser.parse_args(sys.argv[1:])

    hpxmap = Map.read(args.input.name, hdu=args.extension)
    outdata = []

    fig, ax, im = hpxmap.plot(norm=args.zscale, vmin=args.zmin, vmax=args.zmax)
    outdata.append(fig)

    if args.cbar:
        cbar = plt.colorbar(im, orientation='horizontal',shrink=0.7,pad=0.15, fraction=0.05)


    """
    if args.ebin == "ALL":
        wcsproj = hpxmap.geom.make_wcs(
            naxis=2, proj='MOL', energies=None, oversample=2)
        mapping = HpxToWcsMapping(hpxmap.hpx, wcsproj)

        for i, data in enumerate(hpxmap.counts):
            ip = ImagePlotter(data=data, proj=hpxmap.hpx, mapping=mapping)
            fig = plt.figure(i)
            im, ax = ip.plot(zscale=args.zscale,
                             vmin=args.zmin, vmax=args.zmax)
            outdata.append(fig)

    elif args.ebin is None:
        ip = ImagePlotter(data=hpxmap.counts, proj=hpxmap.hpx)
        im, ax = ip.plot(zscale=args.zscale, vmin=args.zmin, vmax=args.zmax)
        outdata.append((im, ax))
    else:
        try:
            ibin = int(args.ebin)
            ip = ImagePlotter(data=hpxmap.counts[ibin], proj=hpxmap.hpx)
            im, ax = ip.plot(zscale=args.zscale,
                             vmin=args.zmin, vmax=args.zmax)
            outdata.append((im, ax))
        except:
            raise ValueError("--ebin argument must be an integer or 'ALL'")
    """

    if args.output is None:
        plt.show()
    else:
        if len(outdata) == 1:
            plt.savefig(args.output.name)
        else:
            base, ext = os.path.splitext(args.output.name)
            for i, fig in enumerate(outdata):
                fig.savefig("%s_%02i%s" % (base, i, ext))


if __name__ == "__main__":
    
    main()
