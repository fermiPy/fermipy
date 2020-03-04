import os

import sys

import argparse

from astropy.io import fits

import GtApp



def set_gtbin_pars_from_cccube(ccube, gtbin, padding):
    """Extract Parameters to run gtbin from a count cube FITS header
    and pass those parameters to the python object interface to gtbin
    """

    header = ccube[0].header

    pixsize = header['CDELT2']
    cproj = header['CTYPE1']
    xref = header['CRVAL1']
    yref = header['CRVAL2']

    coord = cproj.split('-')[0]
    proj = cproj.split('-')[1]

    npix_x = header['NAXIS1']
    npix_y = header['NAXIS2']

    extra_pix = 2*int(padding/pixsize)
    npix_x += extra_pix
    npix_y += extra_pix

    if coord == 'GLON':
        coordsys = 'GAL'
    elif coord == 'RA':
        coordsys = 'CEL'
    else:
        raise ValueError("Unknown coordinate system %s" % coord)

    emin_MeV = ccube['EBOUNDS'].data['E_MIN'][0] / 1000.
    emax_MeV = ccube['EBOUNDS'].data['E_MAX'][-1] / 1000.
    nebins = len(ccube['EBOUNDS'].data['E_MAX'])

    gtbin['algorithm'] = 'CCUBE'
    gtbin['ebinalg'] = 'log'
    gtbin['emin'] = emin_MeV
    gtbin['emax'] = emax_MeV
    gtbin['enumbins'] = nebins
    gtbin['coordsys'] = coordsys
    gtbin['proj'] = proj
    gtbin['xref'] = xref
    gtbin['yref'] = yref
    gtbin['binsz'] = pixsize
    gtbin['nxpix'] = npix_x
    gtbin['nypix'] = npix_y




def main():

    parser = argparse.ArgumentParser(usage='fermipy-make-wmap [options]',
                                     description='Compute the weights maps for weighted likelihood analysis')

    parser.add_argument('--ncomp', action='store', type=int,
                        help="Number of components", default=1)
    parser.add_argument('--padding', action='store', type=float,
                        help="ROI Padding (degrees)", default=5.)
    parser.add_argument('--epsilon', action='store', type=float,
                        help="Fractional Syst. Error", default=0.03)
    parser.add_argument('--ltcube', action='store', type=str, 
                        help="Livetime cube file prefix", default='ltcube')
    parser.add_argument('--ft1', action='store', type=str,
                        help="FT1 file prefix", default='ft1')
    parser.add_argument('--ft2', action='store', type=str,
                        help="Spacecraft file", default='ft2.fits')
    parser.add_argument('--ccube', action='store', type=str,
                        help="Input counts cube file prefix", default='ccube')
    parser.add_argument('--ccube_padded', action='store', type=str,
                        help="Padded counts cube file prefix", default='ccube_padded')
    parser.add_argument('--srcmdl', action='store', type=str,
                        help="Source model xml file prefix",  default='srcmdl')
    parser.add_argument('--bexmap', action='store', type=str,
                        help="Binned exposure map file prefix", default='bexpmap')
    parser.add_argument('--srcmaps_padded', action='store', type=str,
                        help="Padded source maps file prefix", default='srcmaps_padded')
    parser.add_argument('--mcube_padded', action='store', type=str,
                        help="Padded model cube file prefix", default='mcube_padded')
    parser.add_argument('--effbkg', action='store', type=str,
                        help="Effective background file prefix", default='effbkg_model')
    parser.add_argument('--wmap', action='store', type=str,
                        help="Weights map file prefix", default='wmap_model_0p03')
    parser.add_argument('--effbkg_list', action='store', type=str,
                        help="List of effective background files", default='effbkg_model_list.txt')
    parser.add_argument('--alphabkg', action='store', type=str,
                        help="Weights alpha factor map file", default='alphabkg_model_0p03.fits')

    args = parser.parse_args(sys.argv[1:])

    if args.ncomp > 1:
        effbkg_list = open(args.effbkg_list, 'w!')

    for i in range(args.ncomp):

        if args.ncomp == 1:
            suffix_fits = ".fits"
            suffix_xml = ".xml"
        else:
            suffix_fits = "_%02i.fits" % i
            suffix_xml = "_%02i.xml" % i

        ccube = fits.open(args.ccube + suffix_fits)
        gtbin  = GtApp.GtApp('gtbin')

        set_gtbin_pars_from_cccube(ccube, gtbin, args.padding)
        gtbin['evfile'] = args.ft1 + suffix_fits
        gtbin['outfile'] = args.ccube_padded + suffix_fits
        gtbin['scfile'] = args.ft2
        if not os.path.exists(gtbin['outfile']):
            gtbin.run(print_command=True)

        gtsrcmaps = GtApp.GtApp('gtsrcmaps')
        gtsrcmaps['expcube'] = args.ltcube + suffix_fits
        gtsrcmaps['cmap'] = args.ccube_padded + suffix_fits
        gtsrcmaps['srcmdl'] = args.srcmdl + suffix_xml
        gtsrcmaps['bexpmap'] = args.bexmap + suffix_fits
        gtsrcmaps['irfs'] = 'CALDB'
        gtsrcmaps['outfile'] = args.srcmaps_padded + suffix_fits
        if not os.path.exists(gtsrcmaps['outfile']):
            gtsrcmaps.run(print_command=True)

        gtmodel = GtApp.GtApp('gtmodel')
        gtmodel['outtype'] = 'CCUBE'
        gtmodel['srcmaps'] = args.srcmaps_padded + suffix_fits
        gtmodel['outfile'] = args.mcube_padded + suffix_fits
        gtmodel['srcmdl'] = args.srcmdl + suffix_xml
        gtmodel['bexpmap'] = args.bexmap + suffix_fits
        gtmodel['expcube'] = args.ltcube + suffix_fits
        gtmodel['irfs'] = 'CALDB'
        if not os.path.exists(gtmodel['outfile']):
            gtmodel.run(print_command=True)

        gteffbkg = GtApp.GtApp('gteffbkg')
        gteffbkg['cmap'] = args.mcube_padded + suffix_fits
        gteffbkg['bexpmap'] = args.bexmap + suffix_fits
        gteffbkg['expcube'] = args.ltcube + suffix_fits
        gteffbkg['outfile'] = args.effbkg + suffix_fits
        gteffbkg['irfs'] = 'CALDB'
        if not os.path.exists(gteffbkg['outfile']):
            gteffbkg.run(print_command=True)

        if args.ncomp > 1:
            effbkg_list.write("%s\n" % gteffbkg['outfile'])

    if args.ncomp > 1:
        effbkg_list.close()

        gtalphabkg = GtApp.GtApp('gtalphabkg')
        gtalphabkg['inputs'] = args.effbkg_list
        gtalphabkg['outfile'] = args.alphabkg
        gtalphabkg['epsilon'] = args.epsilon
        if not os.path.exists(gtalphabkg['outfile']):
            gtalphabkg.run(print_command=True)

    for i in range(args.ncomp):

        if args.ncomp == 1:
            suffix_fits = ".fits"
        else:
            suffix_fits = "_%02i.fits" % i

        gtwtsmap = GtApp.GtApp('gtwtsmap')
        gtwtsmap['effbkgfile'] = args.effbkg + suffix_fits
        gtwtsmap['alphafile'] = args.alphabkg
        gtwtsmap['outfile'] = args.wmap + suffix_fits
        gtwtsmap['epsilon'] = args.epsilon
        if not os.path.exists(gtwtsmap['outfile']):
            gtwtsmap.run(print_command=True)


if __name__ == '__main__':
    main()
