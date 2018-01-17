# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import os
import copy
import sys
import argparse
import tempfile
import logging
import re
import shutil
import pprint
from fermipy.utils import mkdir
from fermipy.batch import dispatch_job, add_lsf_args, submit_jobs
from fermipy.logger import Logger
from fermipy.gtanalysis import run_gtapp


def create_filelist(filelist, outfile):

    with open(outfile, 'w') as fd:
        for s in filelist:
            fd.write(s + '\n')


def main():

    usage = "usage: %(prog)s [options] "
    description = "Run gtselect and gtmktime on one or more FT1 files.  "
    "Note that gtmktime will be skipped if no FT2 file is provided."
    parser = argparse.ArgumentParser(usage=usage, description=description)

    add_lsf_args(parser)
    parser.add_argument('--zmax', default=100., type=float, help='')
    parser.add_argument('--dcostheta', default=0.025, type=float, help='')
    parser.add_argument('--binsz', default=1.0, type=float, help='')
    parser.add_argument('--outdir', default=None, type=str,
                        help='Path to output directory used when merge=False.')
    parser.add_argument('--outfile', default=None, type=str,
                        help='Path to output file used when merge=True.')
    parser.add_argument('--scfile', default=None, type=str, help='',
                        required=True)

    parser.add_argument('--dry_run', default=False, action='store_true')
    parser.add_argument('--overwrite', default=False, action='store_true')
    parser.add_argument('--merge', default=False, action='store_true',
                        help='Merge input FT1 files into a single file.')

    parser.add_argument('files', nargs='+', default=None,
                        help='List of directories in which the analysis will '
                             'be run.')

    args = parser.parse_args()

    args.outdir = os.path.abspath(args.outdir)
    args.scfile = os.path.abspath(args.scfile)
    mkdir(args.outdir)
    input_files = [[os.path.abspath(x)] for x in args.files]
    output_files = [os.path.join(args.outdir, os.path.basename(x))
                    for x in args.files]

    if args.batch:
        opts = copy.deepcopy(args.__dict__)
        opts.pop('files')
        opts.pop('batch')
        submit_jobs('python ' + os.path.abspath(__file__.rstrip('cd')),
                    input_files, output_files, {k: v for k, v in opts.items()})
        sys.exit(0)

    logger = Logger.get(os.path.basename(__file__), None, logging.INFO)
    logger.info('Starting.')
    cwd = os.getcwd()
    user = os.environ['USER']
    tmpdir = tempfile.mkdtemp(prefix=user + '.', dir='/scratch')
    os.chdir(tmpdir)

    logger.info('tmpdir %s', tmpdir)
    logger.info('outdir %s', args.outdir)
    logger.info('outfile %s', args.outfile)

    for infiles, outfile in zip(input_files, output_files):

        logger.info('infiles %s', pprint.pformat(infiles))
        logger.info('outfile %s', outfile)

        kw = dict(evfile='list.txt',
                  scfile=args.scfile,
                  outfile='ltcube.fits',
                  binsz=args.binsz,
                  dcostheta=args.dcostheta,
                  zmax=args.zmax)

        create_filelist(infiles, 'list.txt')
        staged_outfile = kw['outfile']
        run_gtapp('gtltcube', logger, kw)
        logger.info('cp %s %s', staged_outfile, outfile)
        shutil.copy(staged_outfile, outfile)

    os.chdir(cwd)
    logger.info('Deleting %s', tmpdir)
    shutil.rmtree(tmpdir)
    logger.info('Done.')


if __name__ == "__main__":
    main()
