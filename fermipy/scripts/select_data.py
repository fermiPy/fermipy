# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
import tempfile
import logging
import re
import shutil
import pprint
import numpy as np
from fermipy.utils import mkdir
from fermipy.batch import submit_jobs, add_lsf_args
from fermipy.logger import Logger
from fermipy.gtanalysis import run_gtapp
from fermipy.validate.utils import get_files


def create_filelist(filelist, outfile):

    with open(outfile, 'w') as fd:
        for s in filelist:
            fd.write(s + '\n')


def main():

    gtselect_keys = ['tmin', 'tmax', 'emin', 'emax', 'zmax', 'evtype', 'evclass',
                     'phasemin', 'phasemax', 'convtype', 'rad', 'ra', 'dec']

    gtmktime_keys = ['roicut', 'filter']

    usage = "usage: %(prog)s [options] "
    description = "Run gtselect and gtmktime on one or more FT1 files.  "
    "Note that gtmktime will be skipped if no FT2 file is provided."
    parser = argparse.ArgumentParser(usage=usage, description=description)

    add_lsf_args(parser)

    for k in gtselect_keys:

        if k in ['evtype', 'evclass', 'convtype']:
            parser.add_argument('--%s' % k, default=None, type=int, help='')
        else:
            parser.add_argument('--%s' % k, default=None, type=float, help='')

    for k in gtmktime_keys:
        parser.add_argument('--%s' % k, default=None, type=str, help='')

    parser.add_argument('--rock_angle', default=None, type=float, help='')

    parser.add_argument('--outdir', default=None, type=str,
                        help='Path to output directory used when merge=False.')
    parser.add_argument('--output', default=None, type=str,
                        help='Path to output file used when merge=True.')
    parser.add_argument('--scfile', default=None, type=str, help='')

    parser.add_argument('--dry_run', default=False, action='store_true')
    parser.add_argument('--overwrite', default=False, action='store_true')
    parser.add_argument('--merge', default=False, action='store_true',
                        help='Merge input FT1 files into N files where N is determined '
                        'by files_per_split.')

    parser.add_argument('--files_per_split', default=100,
                        type=int, help='Set the number of files to combine in each '
                        'split of the input file list.')

    parser.add_argument('--file_idx_min', default=None,
                        type=int, help='Set the number of files to assign to '
                        'each batch job.')

    parser.add_argument('--file_idx_max', default=None,
                        type=int, help='Set the number of files to assign to '
                        'each batch job.')

    parser.add_argument('files', nargs='+', default=None,
                        help='List of files.')

    args = parser.parse_args()

    batch = vars(args).pop('batch')
    files = vars(args).pop('files')
    args.outdir = os.path.abspath(args.outdir)
    files = [os.path.abspath(f) for f in files]

    ft1_files = get_files(files, ['.fit', '.fits'])
    for i, f in enumerate(ft1_files):
        if re.search('^root\:\/\/', f) is None:
            ft1_files[i] = os.path.abspath(f)

    input_files = []
    output_files = []
    files_idx_min = []
    files_idx_max = []
    opts = []

    if args.file_idx_min is not None and args.file_idx_max is not None:

        files_idx_min = [args.file_idx_min]
        files_idx_max = [args.file_idx_max]
        input_files = [files]
        output_files = [args.output]

    elif args.merge:
        if not args.output:
            raise Exception('No output file defined.')

        nfiles = len(ft1_files)
        njob = int(np.ceil(nfiles / float(args.files_per_split)))
        for ijob, i in enumerate(range(0, nfiles, args.files_per_split)):

            if args.outdir is not None:
                mkdir(args.outdir)
                outdir = os.path.abspath(args.outdir)
            else:
                outdir = os.path.dirname(os.path.dirname(args.output))

            outfile = os.path.splitext(os.path.basename(args.output))[0]
            outfile += '_%03i.fits' % (ijob)
            outfile = os.path.join(outdir, outfile)
            input_files += [files]
            output_files += [outfile]
            files_idx_min += [i]
            files_idx_max += [i + args.files_per_split]
            opts += [vars(args).copy()]
            opts[-1]['output'] = outfile
            opts[-1]['file_idx_min'] = i
            opts[-1]['file_idx_max'] = i + args.files_per_split

    else:
        input_files = ft1_files
        files_idx_min = [i for i in range(len(ft1_files))]
        files_idx_max = [i + 1 for i in range(len(ft1_files))]
        output_files = [os.path.join(
            args.outdir, os.path.basename(x)) for x in ft1_files]
        opts = [vars(args).copy() for x in ft1_files]

    if batch:
        submit_jobs('fermipy-select',
                    input_files, opts, output_files, overwrite=args.overwrite,
                    dry_run=args.dry_run)
        sys.exit(0)

    logger = Logger.configure(os.path.basename(__file__), None, logging.INFO)
    logger.info('Starting.')

    if args.scfile is not None:
        args.scfile = os.path.abspath(args.scfile)

    cwd = os.getcwd()
    user = os.environ['USER']
    tmpdir = tempfile.mkdtemp(prefix=user + '.', dir='/scratch')
    os.chdir(tmpdir)

    logger.info('tmpdir %s', tmpdir)
    logger.info('outdir %s', args.outdir)
    logger.info('output %s', args.output)

    for infiles, outfile, idx_min, idx_max in zip(input_files, output_files,
                                                  files_idx_min, files_idx_max):

        logger.info('infiles %s', pprint.pformat(infiles))
        logger.info('outfile %s', outfile)
        infiles = get_files(infiles, ['.fit', '.fits'])
        if idx_min is not None:
            infiles = infiles[idx_min:idx_max]

        for i, f in enumerate(infiles):

            if re.search('^root\:\/\/', f) is None:
                continue
            os.system('xrdcp %s %s' % (f, f.split('/')[-1]))
            infiles[i] = os.path.join(tmpdir, f.split('/')[-1])

        kw = {k: args.__dict__[k] for k in gtselect_keys}
        if kw['emax'] is None:
            kw['emax'] = 1E6

        create_filelist(infiles, 'list.txt')
        kw['infile'] = 'list.txt'
        kw['outfile'] = 'out.fits'
        staged_outfile = kw['outfile']
        run_gtapp('gtselect', logger, kw)

        kw = {k: args.__dict__[k] for k in gtmktime_keys}
        if kw['roicut'] is None:
            kw['roicut'] = 'no'

        if kw['filter'] is None:
            kw['filter'] = 'DATA_QUAL==1 && LAT_CONFIG==1'
            if args.rock_angle is not None:
                kw['filter'] += ' && ABS(ROCK_ANGLE)<%(rock)s ' % dict(
                    rock=args.rock_angle)
        kw['evfile'] = 'out.fits'
        kw['outfile'] = 'out_filtered.fits'
        if args.scfile is not None:
            kw['scfile'] = args.scfile
            staged_outfile = kw['outfile']
            run_gtapp('gtmktime', logger, kw)

        logger.info('cp %s %s', staged_outfile, outfile)
        shutil.copy(staged_outfile, outfile)

    os.chdir(cwd)
    logger.info('Deleting %s', tmpdir)
    shutil.rmtree(tmpdir)
    logger.info('Done.')


if __name__ == "__main__":
    main()
