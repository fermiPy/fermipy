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
from fermipy.validate.tools import AGNAccumulator

def make_outpath(f,outdir):

    filename = os.path.splitext(os.path.basename(f))[0] + '_hist.fits'

    if outdir is None:
        outdir = os.path.abspath(os.path.dirname(f))

    return os.path.join(outdir,filename)
    
def main():
    
    usage = "usage: %(prog)s [options] "
    description = "Run validation analysis"
    parser = argparse.ArgumentParser(usage=usage, description=description)

    add_lsf_args(parser)

    parser.add_argument('--outdir', default=None, type=str,
                        help='Path to output directory used when merge=False.')
    parser.add_argument('--outfile', default=None, type=str,
                        help='Path to output file used when merge=True.')
    parser.add_argument('--dry_run', default=False, action='store_true')
    parser.add_argument('--data_type', default='agn', type=str)
    parser.add_argument('--mode', default='fill', type=str)
    parser.add_argument('--overwrite', default=False, action='store_true')
    parser.add_argument('files', nargs='+', default=None,
                        help='List of directories in which the analysis will '
                             'be run.')
    
    args = parser.parse_args()
    
    if args.outdir is not None:
        args.outdir = os.path.abspath(args.outdir)
        mkdir(args.outdir)

    if args.mode == 'fill':
        input_files = [[os.path.abspath(x)] for x in args.files]
        output_files = [make_outpath(x,args.outdir) for x in args.files]
    elif args.mode == 'collect':        
        input_files = [[os.path.abspath(x) for x in args.files]]
        output_files = [args.outfile]

    print(input_files)
    print(output_files)
        
    if args.batch:

        batch_opts = {'W' : args.time, 'R' : args.resources,
                      'oo' : 'batch.log' }
        args.batch=False
        for infile, outfile in zip(input_files,output_files):
            
            if os.path.isfile(outfile) and not args.overwrite:
                print('Output file exists, skipping.',outfile)
                continue
            
            batch_opts['oo'] = os.path.splitext(outfile)[0] + '.log'
            dispatch_jobs('python ' + os.path.abspath(__file__.rstrip('cd')),
                          infile, args, batch_opts, dry_run=args.dry_run)
        sys.exit(0)

    logger = Logger.get(os.path.basename(__file__),None,logging.INFO)
    logger.info('Starting.')

    for infiles, outfile in zip(input_files,output_files):

        if args.data_type == 'agn':    
            acc = AGNAccumulator()

        for f in infiles:
            print('process',f)
            acc.process(f)

        print('write',outfile)
        acc.write(outfile)
        
    logger.info('Done.')

if __name__ == "__main__":
    main()
