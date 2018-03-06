# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import yaml
import argparse
import tempfile
import logging
import re
import shutil
import pprint
from fermipy.utils import mkdir
from fermipy.batch import submit_jobs, add_lsf_args
from fermipy.logger import Logger
from fermipy.gtanalysis import run_gtapp
from fermipy.validate.tools import *


def make_outpath(f, outdir):

    filename = os.path.splitext(os.path.basename(f))[0] + '_hist.fits'

    if outdir is None:
        outdir = os.path.abspath(os.path.dirname(f))

    return os.path.join(outdir, filename)


def main():

    usage = "usage: %(prog)s [options] "
    description = "Run validation analysis"
    parser = argparse.ArgumentParser(usage=usage, description=description)

    add_lsf_args(parser)

    parser.add_argument('--config', default=None, type=str, required=True,
                        help='Configuration file.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='Key name of data set to analyze.  If None then all data '
                        'sets will be analyzed.')
    parser.add_argument('--outdir', default=None, type=str,
                        help='Path to output directory used when merge=False.')
    parser.add_argument('--outfile', default=None, type=str,
                        help='Path to output file used when merge=True.')
    parser.add_argument('--dry_run', default=False, action='store_true')
    parser.add_argument('--mode', default='fill', type=str)
    parser.add_argument('--overwrite', default=False, action='store_true')

    args = parser.parse_args()

    # if args.outdir is not None:
    #    args.outdir = os.path.abspath(args.outdir)
    #    mkdir(args.outdir)

    # if args.mode == 'fill':
    #    input_files = [[os.path.abspath(x)] for x in args.files]
    #    output_files = [make_outpath(x,args.outdir) for x in args.files]
    # elif args.mode == 'collect':
    #    input_files = [[os.path.abspath(x) for x in args.files]]
    #    output_files = [args.outfile]

    # print(input_files)
    # print(output_files)

    config = yaml.load(open(args.config))

    if args.batch:

        input_files = [[]] * len(config.keys())
        output_files = [v['outfile'] for k, v in config.items()]

        opts = []
        for k, v in config['datasets'].items():
            o = vars(args).copy()
            del o['batch']
            o['dataset'] = k
            opts += [o]

        submit_jobs('fermipy-validate',
                    input_files, opts, output_files, overwrite=args.overwrite,
                    dry_run=args.dry_run)
        sys.exit(0)

    logger = Logger.get(os.path.basename(__file__), None, logging.INFO)
    logger.info('Starting.')

    for k, v in config['datasets'].items():

        if args.dataset is not None and k != args.dataset:
            continue

        if v['data_type'] == 'agn':
            val = AGNValidator(config['scfile'], 100.)
        elif v['data_type'] == 'psr':
            val = PSRValidator(config['scfile'], 100.)
        elif v['data_type'] == 'ridge':
            val = GRValidator(config['scfile'], 100.)
        else:
            raise Exception('Unknown data type {}'.format(v['data_type']))

        infiles = glob.glob(v['files'])

        for f in infiles:
            print('processing', f)
            val.process(f)

        val.calc_eff()
        if v['data_type'] in ['agn', 'psr']:
            val.calc_containment()

        print('write', v['outfile'])
        val.write(v['outfile'])

    logger.info('Done.')


if __name__ == "__main__":
    main()
