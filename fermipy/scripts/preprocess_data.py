#!/usr/bin/env python
import os
import time
import glob
import argparse
from os.path import join, basename, dirname, splitext
from collections import OrderedDict as odict

import yaml
import numpy as np
from astropy.io import fits

from fermipy.utils import mkdir
from fermipy.batch import bsub


def pwd():
    # Careful, won't work after a call to os.chdir...
    return os.environ['PWD']


# IRF to event class mapping
EVENTCLASS = odict(
    P7REP_CLEAN=("INDEF", "3"),
    P7REP_ULTRACLEAN=("INDEF", "4"),
    P7REP_SOURCE=("INDEF", "2"),
    P8R2_SOURCE=("INDEF", "128"),
    P8R2_TRANSIENT020E=("INDEF", "8"),
)


def main():

    usage = "usage: %(prog)s [options] "
    description = "Preselect data."
    parser = argparse.ArgumentParser(usage=usage, description=description)
    parser.add_argument("-q", "--queue", default="kipac-ibq")
    parser.add_argument("--evfile", default=None, required=True)
    parser.add_argument("--scfile", default=None, required=True)
    parser.add_argument("--evclass", default="P8R2_SOURCE",
                        choices=EVENTCLASS.keys(),
                        help="Event class selection.")
    parser.add_argument("--zmax", default=100., type=float,
                        help="Maximum zenith angle for selection")
    parser.add_argument("--emin", default=1., type=float,
                        help="Minimum energy for selection")
    parser.add_argument("--emax", default=1000000., type=float,
                        help="Minimum energy for selection")
    parser.add_argument("--rock", default=52., type=float,
                        help="Maximum rocking angle cut")
    parser.add_argument("--rock_min", default=None, type=float,
                        help="Minimum rocking angle cut")
    parser.add_argument("--chatter", default=2, type=int,
                        help="ST chatter level")
    parser.add_argument("--gtifile_grb", default=None, type=str,
                        help="GRB GTI file.")
    parser.add_argument("--gtifile_sun", default=None, type=str,
                        help="Sun GTI file.")

    args = parser.parse_args()

    basedir = pwd()
    evclsmin, evclass = EVENTCLASS[args.evclass]

    # Setup gtmktime filter
    gti_dir = '/u/gl/mdwood/ki20/mdwood/fermi/data'
    gti_grb = '%s/nogrb.gti' % gti_dir
    gti_sfr = '%s/nosolarflares.gti' % gti_dir

    #sfr_gticut = "gtifilter(\"%s\",(START+STOP)/2)" % (gti_sfr)
    #gticut_sun = "ANGSEP(RA_SUN,DEC_SUN,RA_ZENITH,DEC_ZENITH)>115"

    mktime_filter = 'DATA_QUAL==1 && LAT_CONFIG==1 '
    if args.rock is not None:
        mktime_filter += '&& ABS(ROCK_ANGLE)<%(rock)s ' % dict(rock=args.rock)

    if args.rock_min is not None:
        mktime_filter += '&& ABS(ROCK_ANGLE)>%(rock)s ' % dict(rock=args.rock_min)

    if args.gtifile_grb:
        gticut_grb = "gtifilter(\"%s\",START) && " % (args.gtifile_grb)
        gticut_grb += "gtifilter(\"%s\",STOP)" % (args.gtifile_grb)
        mktime_filter += '&& %s' % gticut_grb

    if args.gtifile_sun:
        gticut_sun = "gtifilter(\"%s\",(START+STOP)/2)" % (args.gtifile_sun)
        mktime_filter += '&& (ANGSEP(RA_SUN,DEC_SUN,RA_ZENITH,DEC_ZENITH)>115 || %s)' % gticut_sun

    # if args.transient_cut:
    #    mktime_filter += '&& %s' % gticut_grb
    #    mktime_filter += '&& (%s || %s)' % (sfr_gticut, gticut_sun)

    # First take care of the scfile
    scfile = os.path.basename(args.scfile)
    if not os.path.lexists(scfile):
        os.symlink(args.scfile.strip('@'), scfile)

    # Now take care of the evfile
    if args.evfile.startswith('@'):
        evfiles = np.loadtxt(args.evfile.strip('@'), dtype='str')
    else:
        evfiles = [args.evfile]

    # Now take care of the scfile
    if args.scfile.startswith('@'):
        scfiles = np.loadtxt(args.scfile.strip('@'), dtype='str')
    else:
        scfiles = [args.scfile] * len(evfiles)

    # Now create an output directory
    outdir = mkdir(join(basedir, 'ft1'))
    logdir = mkdir(join(basedir, 'log'))

    lst, cmnds, logs = [], [], []
    jobname = "preprocess"
    tstarts, tstops = [], []
    for evfile, scfile in zip(evfiles, scfiles):
        # PFILES
        scratch = join("/scratch", os.environ["USER"])
        os.environ['PFILES'] = scratch + ';' + \
            os.environ['PFILES'].split(';')[-1]

        # Deal with the times
        header = fits.open(evfile)[0].header
        tstart = int(float(header['TSTART']))
        tstop = int(float(header['TSTOP']))
        tstarts.append(tstart)
        tstops.append(tstop)
        outfile = join(outdir, "%s_%i_%i_z%g_r%g_ft1.fits" %
                       (args.evclass, tstart, tstop, args.zmax, args.rock))

        logfile = join(logdir, basename(outfile).replace('fits', 'log'))

        params = dict(evfile=evfile,
                      scfile=scfile,
                      select="${workdir}/select_ft1.fits",
                      outfile=outfile,
                      evclass=evclass,
                      zmax=args.zmax,
                      emin=args.emin,
                      emax=args.emax,
                      filter=mktime_filter,
                      chatter=args.chatter)

        setup = """
mkdir /scratch/$USER >/dev/null 2>&1;
workdir=$(mktemp -d -p /scratch/$USER);\n
"""

        select = """gtselect \
 infile=%(evfile)s \
 outfile=%(select)s \
 tmin=0 tmax=0 emin=%(emin)s emax=%(emax)s \
 ra=0 dec=0 rad=180 zmax=%(zmax)s \
 evclass=%(evclass)s \
 chatter=4;
""" % params

        mktime = """gtmktime \
 evfile=%(select)s \
 outfile=%(outfile)s \
 scfile=%(scfile)s \
 filter='%(filter)s' \
 roicut='no' \
 chatter=4;
""" % params

        cleanup = "\n\nstatus=$?;\nrm -rf $workdir;\nexit $status;"

        cmnd = setup + select + mktime + cleanup

        logs.append(logfile)
        cmnds.append(cmnd)
        lst.append(outfile)
    bsub(jobname, cmnds, logs, W='300', submit=True)

    lstfile = "%s_%s_%s_z%g_r%g_ft1.lst" % (
        args.evclass, min(tstarts), max(tstops), args.zmax, args.rock)
    print("Writing ft1 file list: %s\n" % lstfile)
    np.savetxt(lstfile, sorted(lst), fmt="%s")

    print("Done.")


if __name__ == "__main__":
    main()
