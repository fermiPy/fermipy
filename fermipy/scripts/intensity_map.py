# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import argparse

import numpy as np

from fermipy.skymap import HpxMap
from astropy.io import fits

def intensity_cube(ccube, bexpcube, hpx_order):
    """
    """
    if hpx_order == ccube.hpx.order:
        ccube_at_order = ccube
    else:
        ccube_at_order = ccube.ud_grade(hpx_order, preserve_counts=True)
    
    if hpx_order == bexpcube.hpx.order:
        bexpcube_at_order = bexpcube
    else:
        bexpcube_at_order = bexpcube.ud_grade(hpx_order, preserve_counts=True)
    
    bexpcube_data = np.sqrt(bexpcube_at_order.data[0:-1,0:]*bexpcube_at_order.data[1:,0:])
    out_data = ccube_at_order.counts / bexpcube_data
    return HpxMap(out_data, ccube_at_order.hpx)    


def main():
    """ Main function for command line usage """
    usage = "usage: %(prog)s [options] "
    description = "Merge a set of Fermi-LAT files."

    parser = argparse.ArgumentParser(usage=usage, description=description)

    parser.add_argument('-o', '--output', default=None, type=str,
                        help='Output file.')
    parser.add_argument('--ccube', default=None, type=str,
                        help='Input counts cube file .')
    parser.add_argument('--bexpcube', default=None, type=str,
                        help='Input binned exposure cube.')
    parser.add_argument('--hpx_order', default=None, type=int,
                        help='Order of output map: default = counts map order')
    parser.add_argument('--clobber', action='store_true',
                        help='Overwrite output file')
 
    args = parser.parse_args()

    ccube = HpxMap.create_from_fits(args.ccube, hdu='SKYMAP')
    bexpcube = HpxMap.create_from_fits(args.bexpcube, hdu='HPXEXPOSURES')
    
    if args.hpx_order:
        hpx_order = args.hpx_order
    else:
        hpx_order = ccube.hpx.order

    out_cube = intensity_cube(ccube, bexpcube, hpx_order)
    out_cube.hpx.write_fits(out_cube.data, args.output, clobber=args.clobber)


if __name__ == '__main__':
    main()
