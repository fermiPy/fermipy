# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
import tempfile
import re
import shutil
import logging
from fermipy.batch import dispatch_jobs, add_lsf_args
from fermipy.utils import mkdir
from fermipy.logger import Logger

def getEntries(inFile):

    import ROOT

    FP = ROOT.TFile.Open(inFile)
    tree = FP.Get('MeritTuple')
    return tree.GetEntries()


def skimMerit(inFile, outfilename, selection,
              nentries, firstentry, enableB=None, disableB=None):

    import ROOT
    print('Preparing merit chunk from %s' % inFile)
    print('Opening input file %s' % inFile)
    oldFP = ROOT.TFile.Open(inFile)
    oldTree = oldFP.Get('MeritTuple')
    oldTree.SetBranchStatus('*', 1)
    oldTree.SetBranchStatus('Pulsar_Phase', 0)

#    for branch in enableB:
#      oldTree.SetBranchStatus(branch, 1)
#    for branch in disableB:
#      oldTree.SetBranchStatus(branch, 0)

    newFP = ROOT.TFile(outfilename, "recreate")
    newTree = oldTree.CopyTree(selection, "fast", nentries, firstentry)
    newTree.AutoSave()
    nevents = newTree.GetEntries()
    print('Skimmed events ', nevents)
    newFP.Close()
    print('Closing output file %s' % outfilename)
    oldFP.Close()
    return nevents


def phase_ft1(ft1file, outfile, logFile, ft2file, ephemfile, dry_run=False):
    cmd = '$TEMPO2ROOT/bin/tempo2 '
    cmd += ' -gr fermi -ft1 %s ' % (ft1file)
    cmd += ' -ft2 %s ' % (ft2file)
    cmd += ' -f %s -phase ' % (ephemfile)

    print(cmd)
    if not dry_run:
        os.system(cmd)

    print('cp %s %s' % (ft1file, outfile))
    if not dry_run:
        os.system('cp %s %s' % (ft1file, outfile))


def phase_merit(meritFile, outfile, logFile, ft2file, ephemfile, dry_run=False):
    nevent_chunk = 30000  # number of events to process per chunk
    mergeChain = ROOT.TChain('MeritTuple')

    skimmedEvents = getEntries(meritFile)

    for firstEvent in range(0, skimmedEvents, nevent_chunk):

        filename = os.path.splitext(os.path.basename(meritFile))[0]
        meritChunk = filename + '_%s.root' % firstEvent
        nevts = skimMerit(meritFile, meritChunk,
                          '', nevent_chunk, firstEvent)

        cmd = 'tempo2 -gr root -inFile %s -ft2 %s -f %s -graph 0 -nobs 32000 -npsr 1 -addFriend -phase' % (
            meritChunk, ft2file, ephemfile)

        print(cmd)
        os.system(cmd + ' >> %s 2>> %s' % (logFile, logFile))
        mergeChain.Add(meritChunk)

    mergeFile = ROOT.TFile('merged.root', 'RECREATE')
    if mergeChain.GetEntries() > 0:
        mergeChain.CopyTree('')

    mergeFile.Write()
    print('merged events %s' % mergeChain.GetEntries())
    mergeFile.Close()

    os.system('mv merged.root %s' % (outfile))


def main():
    
    usage = "usage: %(prog)s [options] "
    description = "Run tempo2 application on one or more FT1 files."
    parser = argparse.ArgumentParser(usage=usage, description=description)

    add_lsf_args(parser)
    
    parser.add_argument('--par_file', default=None, type=str, required=True,
                        help='Ephemeris file')

    parser.add_argument('--scfile', default=None, type=str, required=True,
                        help='FT2 file')

    parser.add_argument('--outdir', default=None, type=str, help='')
    
    parser.add_argument('--phase_colname', default='PULSE_PHASE',
                        type=str, help='Set the name of the phase column.')
    
    parser.add_argument('--dry_run', default=False, action='store_true')
    parser.add_argument('--overwrite', default=False, action='store_true')

    parser.add_argument('files', nargs='+', default=None,
                        help='List of directories in which the analysis will '
                             'be run.')
    
    args = parser.parse_args()

    if args.outdir is None:
        outdirs = [os.path.dirname(os.path.abspath(x)) for x in args.files]
    else:
        outdir = os.path.abspath(args.outdir)
        mkdir(args.outdir)
        outdirs = [outdir for x in args.files]

    input_files = [os.path.abspath(x) for x in args.files]
    output_files = [os.path.join(y,os.path.basename(x))
                    for x, y in zip(args.files,outdirs)]
    
    if args.batch:

        batch_opts = {'W' : args.time, 'R' : args.resources,
                      'oo' : 'batch.log' }
        args.batch=False
        for infile, outfile in zip(input_files,output_files):
            
            if os.path.isfile(outfile) and not args.overwrite:
                print('Output file exists, skipping.',outfile)
                continue
            
            batch_opts['oo'] = os.path.join(outdir,
                                            os.path.splitext(outfile)[0] +
                                            '_tempo2.log')            
            dispatch_jobs('python ' + os.path.abspath(__file__.rstrip('cd')),
                          [infile], args, batch_opts, dry_run=args.dry_run)
        sys.exit(0)

    logger = Logger.get(__file__,None,logging.INFO)
        
    par_file = os.path.abspath(args.par_file)
    ft2_file = os.path.abspath(args.scfile)
    
    cwd = os.getcwd()
    user = os.environ['USER']
    tmpdir = tempfile.mkdtemp(prefix=user + '.', dir='/scratch')

    logger.info('tmpdir %s',tmpdir)
    os.chdir(tmpdir)

    for infile, outfile in zip(input_files,output_files):

        staged_infile = os.path.join(tmpdir,os.path.basename(x))
        logFile = os.path.splitext(x)[0] + '_tempo2.log'

        print('cp %s %s' % (infile, staged_infile))
        os.system('cp %s %s' % (infile, staged_infile))

        if not re.search('\.root?', x) is None:
            phase_merit(staged_infile, outfile, logFile, ft2_file, par_file, args.dry_run)
        elif not re.search('\.fits?', x) is None:
            phase_ft1(staged_infile, outfile, logFile, ft2_file, par_file, args.dry_run)
        else:
            print('Unrecognized file extension: ', x)

    os.chdir(cwd)
    shutil.rmtree(tmpdir)


if __name__ == "__main__":
    main()
