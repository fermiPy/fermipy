# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import os
import argparse

import pyLikelihood as pyLike

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column

from fermipy import utils
from fermipy import spectrum
from fermipy import irfs
from fermipy import skymap


def main():
    usage = "usage: %(prog)s [options]"
    description = "Calculate the LAT point-source flux sensitivity."
    parser = argparse.ArgumentParser(usage=usage, description=description)

    parser.add_argument('--ltcube', default=None,
                        help='Set the path to the livetime cube.')
    parser.add_argument('--galdiff', default=None, required=True,
                        help='Set the path to the galactic diffuse model.')
    parser.add_argument('--isodiff', default=None,
                        help='Set the path to the isotropic model.  If none then the '
                        'default model will be used for the given event class.')
    parser.add_argument('--ts_thresh', default=25.0, type=float,
                        help='Set the detection threshold.')
    parser.add_argument('--min_counts', default=3.0, type=float,
                        help='Set the minimum number of counts.')
    parser.add_argument('--joint', default=False, action='store_true',
                        help='Compute sensitivity using joint-likelihood of all event types.')
    parser.add_argument('--event_class', default='P8R2_SOURCE_V6',
                        help='Set the IRF name.')
    parser.add_argument('--glon', default=0.0, type=float,
                        help='Galactic longitude.')
    parser.add_argument('--glat', default=0.0, type=float,
                        help='Galactic latitude.')
    parser.add_argument('--index', default=2.0, type=float,
                        help='Source power-law index.')
    parser.add_argument('--emin', default=100., type=float,
                        help='Minimum energy in MeV.')
    parser.add_argument('--emax', default=100000., type=float,
                        help='Maximum energy in MeV.')
    parser.add_argument('--nbin', default=12, type=int,
                        help='Number of energy bins for differential flux calculation.')
    parser.add_argument('--output', default='output.fits', type=str,
                        help='Output filename.')
    parser.add_argument('--obs_time_yr', default=None, type=float,
                        help='Rescale the livetime cube to this observation time in years.  If none then the '
                        'calculation will use the intrinsic observation time of the livetime cube.')

    args = parser.parse_args()
    event_types = [['FRONT', 'BACK']]
    fn = spectrum.PowerLaw([1E-13, -args.index], scale=1E3)

    log_ebins = np.linspace(np.log10(args.emin),
                            np.log10(args.emax), args.nbin + 1)
    ebins = 10**log_ebins
    ectr = np.exp(utils.edge_to_center(np.log(ebins)))

    c = SkyCoord(args.glon, args.glat, unit='deg', frame='galactic')

    if args.ltcube is None:

        if args.obs_time_yr is None:
            raise Exception('No observation time defined.')

        ltc = irfs.LTCube.create_empty(0, args.obs_time_yr * 365 * 24 * 3600.,
                                       args.obs_time_yr * 365 * 24 * 3600.)
        ltc._counts *= ltc.domega[:, np.newaxis] / (4. * np.pi)
    else:
        ltc = irfs.LTCube.create(args.ltcube)
        if args.obs_time_yr is not None:
            ltc._counts *= args.obs_time_yr * 365 * \
                24 * 3600. / (ltc.tstop - ltc.tstart)

    m0 = skymap.Map.create_from_fits(args.galdiff)

    if args.isodiff is None:
        isodiff = utils.resolve_file_path('iso_%s_v06.txt' % args.event_class,
                                          search_dirs=[os.path.join('$FERMIPY_ROOT', 'data'),
                                                       '$FERMI_DIFFUSE_DIR'])
        isodiff = os.path.expandvars(isodiff)
    else:
        isodiff = args.isodiff

    iso = np.loadtxt(isodiff, unpack=True)
    sig = []
    bkg = []
    for et in event_types:
        psf = irfs.PSFModel(c.icrs, ltc, args.event_class, et, log_ebins)
        exp = irfs.Exposure.create(ltc, args.event_class, et, np.log10(ectr))

        expv = exp.get_map_values(c.icrs.ra.deg, c.icrs.dec.deg)
        bkgv = m0.interpolate(c.l.deg, c.b.deg, ectr)
        isov = np.exp(np.interp(np.log(ectr), np.log(iso[0]), np.log(iso[1])))
        bkgv += isov
        s, b = irfs.compute_ps_counts(ebins, expv, psf, bkgv, fn)
        sig += [s]
        bkg += [b]

    sig = np.concatenate([np.expand_dims(t, -1) for t in sig])
    bkg = np.concatenate([np.expand_dims(t, -1) for t in bkg])

    norms = irfs.compute_norm(sig, bkg, args.ts_thresh,
                              args.min_counts, sum_axes=[1, 2])
    npred = np.squeeze(np.apply_over_axes(np.sum, norms * sig, [1, 2]))
    norms = np.squeeze(norms)
    flux = norms * fn.flux(ebins[:-1], ebins[1:])
    eflux = norms * fn.eflux(ebins[:-1], ebins[1:])
    dnde = norms * fn.dnde(ectr)
    e2dnde = ectr**2 * dnde

    cols = [Column(name='e_min', dtype='f8', data=ebins[:-1], unit='MeV'),
            Column(name='e_ref', dtype='f8', data=ectr, unit='MeV'),
            Column(name='e_max', dtype='f8', data=ebins[1:], unit='MeV'),
            Column(name='flux', dtype='f8', data=flux, unit='ph / (cm2 s)'),
            Column(name='eflux', dtype='f8', data=eflux, unit='MeV / (cm2 s)'),
            Column(name='dnde', dtype='f8', data=dnde,
                   unit='ph / (MeV cm2 s)'),
            Column(name='e2dnde', dtype='f8',
                   data=e2dnde, unit='MeV / (cm2 s)'),
            Column(name='npred', dtype='f8', data=npred, unit='ph')]

    tab = Table(cols)
    tab.write(args.output, format='fits', overwrite=True)

if __name__ == "__main__":
    main()
