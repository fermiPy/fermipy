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
from fermipy.utils import mkdir
from fermipy.batch import dispatch_jobs, add_lsf_args
from fermipy.logger import Logger
from fermipy.gtanalysis import run_gtapp

def create_filelist(filelist, outfile):

    with open(outfile, 'w') as fd:
        for s in filelist:
            fd.write(s + '\n')


def main():

    gtselect_keys = ['tmin','tmax','emin','emax','zmax','evtype','evclass',
                     'phasemin','phasemax','convtype','rad','ra','dec']

    gtmktime_keys = ['roicut','filter']
    
    usage = "usage: %(prog)s [options] "
    description = "Run gtselect and gtmktime on one or more FT1 files.  "
    "Note that gtmktime will be skipped if no FT2 file is provided."
    parser = argparse.ArgumentParser(usage=usage, description=description)

    add_lsf_args(parser)
    
    for k in gtselect_keys:

        if k in ['evtype','evclass','convtype']:
            parser.add_argument('--%s'%k, default=None, type=int, help='')
        else:
            parser.add_argument('--%s'%k, default=None, type=float, help='')

    for k in gtmktime_keys:
        parser.add_argument('--%s'%k, default=None, type=str, help='')
        
    parser.add_argument('--rock_angle', default=None, type=float, help='')
        
    parser.add_argument('--outdir', default=None, type=str,
                        help='Path to output directory used when merge=False.')
    parser.add_argument('--outfile', default=None, type=str,
                        help='Path to output file used when merge=True.')
    parser.add_argument('--scfile', default=None, type=str, help='')
        
    parser.add_argument('--dry_run', default=False, action='store_true')
    parser.add_argument('--overwrite', default=False, action='store_true')
    parser.add_argument('--merge', default=False, action='store_true',
                        help='Merge input FT1 files into a single file.')

    parser.add_argument('files', nargs='+', default=None,
                        help='List of directories in which the analysis will '
                             'be run.')
    
    args = parser.parse_args()

    if args.merge:
        if not args.outfile:
            raise Exception('No output file defined.')        
        input_files = [[os.path.abspath(x) for x in args.files]]
        output_files = [os.path.abspath(args.outfile)]
    else:
        args.outdir = os.path.abspath(args.outdir)
        mkdir(args.outdir)
        input_files = [[os.path.abspath(x)] for x in args.files]
        output_files = [os.path.join(args.outdir,os.path.basename(x)) for x in args.files]

    if args.batch:

        batch_opts = {'W' : args.time, 'R' : args.resources,
                      'oo' : 'batch.log' }
        args.batch=False
        for infile, outfile in zip(input_files,output_files):
            
            if os.path.isfile(outfile) and not args.overwrite:
                print('Output file exists, skipping.',outfile)
                continue
            
            batch_opts['oo'] = os.path.splitext(outfile)[0] + '_select.log'
            dispatch_jobs('python ' + os.path.abspath(__file__.rstrip('cd')),
                          infile, args, batch_opts, dry_run=args.dry_run)
        sys.exit(0)


    logger = Logger.get(os.path.basename(__file__),None,logging.INFO)

    logger.info('Starting.')
    
    if args.scfile is not None:
        args.scfile = os.path.abspath(args.scfile)
    
    cwd = os.getcwd()
    user = os.environ['USER']
    tmpdir = tempfile.mkdtemp(prefix=user + '.', dir='/scratch')
    os.chdir(tmpdir)

    logger.info('tmpdir %s',tmpdir)
    logger.info('outdir %s',args.outdir)
    logger.info('outfile %s',args.outfile)
    
    for infiles, outfile in zip(input_files,output_files):

        logger.info('infiles %s',pprint.pformat(infiles))
        logger.info('outfile %s',outfile)
        
        kw = { k : args.__dict__[k] for k in gtselect_keys }
        if kw['emax'] is None:
            kw['emax'] = 1E6

        create_filelist(infiles,'list.txt')
        kw['infile'] = 'list.txt'
        kw['outfile'] = 'out.fits'
        staged_outfile = kw['outfile']
        run_gtapp('gtselect',logger,kw)

        kw = { k : args.__dict__[k] for k in gtmktime_keys }
        if kw['roicut'] is None:
            kw['roicut'] = 'no'
        
        if kw['filter'] is None:
            kw['filter'] = 'DATA_QUAL==1 && LAT_CONFIG==1'
            if args.rock_angle is not None:
                kw['filter'] += ' && ABS(ROCK_ANGLE)<%(rock)s '%dict(rock=args.rock_angle)
        kw['evfile'] = 'out.fits'
        kw['outfile'] = 'out_filtered.fits'
        if args.scfile is not None:
            kw['scfile'] = args.scfile
            staged_outfile = kw['outfile']
            run_gtapp('gtmktime',logger,kw)
                            
        logger.info('cp %s %s',staged_outfile,outfile)
        shutil.copy(staged_outfile,outfile)
        
    os.chdir(cwd)
    logger.info('Deleting %s',tmpdir)
    shutil.rmtree(tmpdir)
    logger.info('Done.')

if __name__ == "__main__":
    main()
