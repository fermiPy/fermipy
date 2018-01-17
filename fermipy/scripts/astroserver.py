# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import subprocess
from subprocess import call
import os
import glob
import argparse
from os.path import join, basename, dirname, splitext

import yaml
import numpy as np
#from dsphs.utils.utc2met import utc2met
from fermipy.utils import mkdir
from fermipy.batch import bsub


class astroserver(object):
    """ Wrapper around the glast astroserver.
    Pass in command-line args as kwargs changing
    '_' to '-'. Checks kwargs f"""

    def __init__(self):
        self.exe = "/u/gl/glast/astroserver/prod/astro"
        p = subprocess.Popen((self.exe + " --help").split(),
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        self.opts = stdout

    def __call__(self, arg, **kwargs):
        command = self.exe
        for key, val in kwargs.items():
            kwarg = key.replace('_', '-')
            if kwarg not in self.opts:
                raise Exception("%s\n%s" % (self.exe, self.opts))
            command += " -%s %s" % (kwarg, val)
        command += " %s" % arg
        return command


# Julian Year (365.25 days) in seconds
YEAR = 31557600


def main():

    usage = "Usage: %(prog)s  [options] input"
    description = "python script"
    parser = argparse.ArgumentParser(usage=usage, description=description)
    parser.add_argument("-d", "--dryrun", action='store_true')
    parser.add_argument("-s", "--sleep", default='1m',
                        help="Pause between")
    parser.add_argument("--ls1", action='store_true', default=False,
                        help='Fetch LS1 files.')
    parser.add_argument("--emin", default=100)
    parser.add_argument("--emax", default=1e6)
    parser.add_argument("--tmin", default=239557414, type=int,
                        help="Min time; default is start of first LAT run")
    parser.add_argument("--tmax", default=None, type=int,
                        help="Default is current time.")
    parser.add_argument("--evtclass", default="Source",
                        help="Event class")
    parser.add_argument("--evtsample", default="P7.6_P130_BASE",
                        choices=['P7.6_P130_BASE', 'P6_public_v3',
                                 'P7_P202_BASE', 'P7_P203_BASE', 'P8_P301_BASE',
                                 'P8_P302_BASE', 'P8_P302_ALL'],
                        help="Event sample")
    parser.add_argument("--chunk", default=int(YEAR // 12), type=int,
                        help="Time chunk for download. Default is ~1 month.")

    args = parser.parse_args()

    basedir = os.environ['PWD']
    codedir = join(basedir, dirname(os.path.relpath(__file__)))
    logdir = join(basedir, "log")
    if not args.dryrun:
        logdir = mkdir(logdir)
    astro = astroserver()

    chunk = args.chunk
    # Might want to think more about how tmin and tmax are set
    first = args.tmin
    if args.tmax is None:
        args.tmax = int(utc2met())

    emin, emax = args.emin, args.emax
    evtclass = args.evtclass
    evtsample = args.evtsample
    sample = '_'.join(evtsample.split('_')[:-1])
    events = evtclass.upper()

    # Break data into chunks
    epsilon = 1e-6
    times = np.arange(args.tmin, args.tmax + epsilon, chunk).astype(int)

    # Get new full ft2 file.
    # Assumption is that it is a longer time period...
    ft2 = join(basedir, "%s_%s_%s_%s_ft2.fits" %
               (sample, events, min(times), max(times)))
    jobname = 'ft2'
    if os.path.exists(ft2):
        # exact ft2 already exists; skip
        print("%s exists; skipping.\n" % ft2)
    else:
        # Remove old ft2 file and replace with link
        if not args.dryrun:
            # for f in glob.glob(join(basedir, "*ft2.fits")):
            #    os.remove(f)
            #    os.symlink(ft2, f)
            for f in glob.glob(join(basedir, "*ft2_fix_checksums.sh")):
                os.remove(f)

        logfile = join(logdir, basename(ft2).replace('fits', 'log'))
        command = astro('storeft2',
                        output_ft2_30s=ft2,
                        _event_sample=evtsample,
                        minTimestamp=min(times),
                        maxTimestamp=max(times),
                        excludeMaxTimestamp='',
                        quiet='',
                        brief='',
                        )

        print(command)
        bsub(jobname, command, logfile, sleep=args.sleep, submit=not args.dryrun,
             W=1000, R='rhel60')

    # Download ft1, ft2 files
    ft1dir = mkdir(join(basedir, 'ft1'))
    ft1_lst, ft1_cmnds, ft1_logs = [], [], []

    ls1dir = mkdir(join(basedir, 'ls1'))
    ls1_lst, ls1_cmnds, ls1_logs = [], [], []

    ft2dir = mkdir(join(basedir, 'ft2'))
    ft2_lst, ft2_cmnds, ft2_logs = [], [], []

    for tmin, tmax in zip(times[:-1], times[1:]):

        # If ft1 file exists, skip it...
        ft1 = join(ft1dir, "%s_%s_%s_%s_ft1.fits" %
                   (sample, events, tmin, tmax))
        if os.path.exists(ft1):
            print("%s exists; skipping.\n" % ft1)
        else:
            ft1_kw = dict(_output_ft1=ft1,
                          _event_sample=evtsample,
                          minTimestamp=tmin,
                          maxTimestamp=tmax,
                          minEnergy=emin,
                          maxEnergy=emax,
                          _event_class_name=evtclass,
                          excludeMaxTimestamp='',
                          quiet='',
                          brief='')

            ft1_cmnd = astro("store", **ft1_kw)
            ft1_logs.append(join(logdir, basename(ft1).replace('fits', 'log')))
            ft1_cmnds.append(ft1_cmnd)
        ft1_lst.append(ft1)

        # If ls1 file exists, skip it...
        ls1 = join(ls1dir, "%s_%s_%s_%s_ls1.fits" %
                   (sample, events, tmin, tmax))
        if not args.ls1:
            print("%s; skipping.\n" % ls1)
        elif os.path.exists(ls1):
            print("%s exists; skipping.\n" % ls1)
        else:
            ls1_kw = dict(_output_ls1=ls1,
                          _event_sample=evtsample,
                          _output_ls1_max_bytes_per_file=0,
                          minTimestamp=tmin,
                          maxTimestamp=tmax,
                          minEnergy=emin,
                          maxEnergy=emax,
                          _event_class_name=evtclass,
                          excludeMaxTimestamp='',
                          quiet='',
                          brief='')

            ls1_cmnd = astro("store", **ls1_kw)
            ls1_logs.append(join(logdir, basename(ls1).replace('fits', 'log')))
            ls1_cmnds.append(ls1_cmnd)
        ls1_lst.append(ls1)

        # If ft2 file exists, skip it...
        ft2 = join(ft2dir, "%s_%s_%s_%s_ft2.fits" %
                   (sample, events, tmin, tmax))
        if os.path.exists(ft2):
            print("%s exists; skipping.\n" % ft2)
        else:
            ft2_cmnd = astro('storeft2',
                             output_ft2_30s=ft2,
                             _event_sample=evtsample,
                             minTimestamp=tmin,
                             maxTimestamp=tmax,
                             excludeMaxTimestamp='',
                             quiet='',
                             brief='',
                             )
            ft2_logs.append(join(logdir, basename(ft2).replace('fits', 'log')))
            ft2_cmnds.append(ft2_cmnd)
        ft2_lst.append(ft2)

    resources = 'bullet,hequ,kiso'

    bsub('ft1', ft1_cmnds, ft1_logs, sleep=args.sleep, submit=not args.dryrun,
         W=1000, R=resources)
    bsub('ls1', ls1_cmnds, ls1_logs, sleep=args.sleep, submit=not args.dryrun,
         W=1000, R=resources)
    bsub('ft2', ft2_cmnds, ft2_logs, sleep=args.sleep, submit=not args.dryrun,
         W=1000, R=resources)

    # Create list of ft1 files
    ft1_lstfile = join(basedir, "%s_%s_%s_%s_ft1.lst" %
                       (sample, events, min(times), max(times)))
    ls1_lstfile = join(basedir, "%s_%s_%s_%s_ls1.lst" %
                       (sample, events, min(times), max(times)))
    ft2_lstfile = join(basedir, "%s_%s_%s_%s_ft2.lst" %
                       (sample, events, min(times), max(times)))
    if not args.dryrun:
        for f in glob.glob(join(basedir, "*.lst")):
            os.remove(f)
        print("Creating ft1 file list: %s" % ft1_lstfile)
        np.savetxt(ft1_lstfile, ft1_lst, fmt='%s')
        print("Creating ls1 file list: %s" % ls1_lstfile)
        np.savetxt(ls1_lstfile, ls1_lst, fmt='%s')
        print("Creating ft2 file list: %s" % ft2_lstfile)
        np.savetxt(ft2_lstfile, ft2_lst, fmt='%s')


if __name__ == "__main__":
    main()
