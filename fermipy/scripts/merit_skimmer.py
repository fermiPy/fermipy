# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import os
import copy
import re
import yaml
import sys
import mimetypes
import tempfile
import string
import random
from os.path import splitext, basename
import xml.etree.cElementTree as ElementTree
import argparse
import numpy as np
import ROOT

from ..batch import submit_jobs, add_lsf_args
from ..validate.utils import get_cuts_from_xml, get_files, set_event_list
from ..validate.utils import load_chain, load_aliases


def get_branches(aliases):
    """Get unique branch names from an alias dictionary."""

    ignore = ['pow', 'log10', 'sqrt', 'max']
    branches = []
    for k, v in aliases.items():

        tokens = re.sub('[\(\)\+\*\/\,\=\<\>\&\!\-\|]', ' ', v).split()

        for t in tokens:

            if bool(re.search(r'^\d', t)) or len(t) <= 3:
                continue

            if bool(re.search(r'[a-zA-Z]', t)) and t not in ignore:
                branches += [t]

    return list(set(branches))


class MeritSkimmer:
    """
    Adapted from Luca B.'s fermiMusic code...
    """

    def __init__(self, chain):
        print('Loading input chain...')
        self.InputChain = chain
        self.OutputChain = None
        # Sometimes chain will have event list defined
        print('Done, %d event(s) found.' %
              self.InputChain.Draw("", "", "goff"))

    def skim(self, outputFilePath, cut="", branches=None):

        if isinstance(outputFilePath, ROOT.TFile):
            outputFile = outputFilePath
        else:
            outputFile = ROOT.TFile(outputFilePath, 'RECREATE')
        if branches is not None and branches:
            self.InputChain.SetBranchStatus('*', False)
            for b in branches:
                self.InputChain.SetBranchStatus(b, True)

        outputChain = self.InputChain.CopyTree(cut)
        self.outputChain = copy.copy(outputChain)
        outputChain.Write()
        print('Done, %d event(s) written to file.' %
              self.outputChain.GetEntries())
        if not isinstance(outputFilePath, ROOT.TFile):
            outputFile.Close()
        return self.outputChain


def create_file_list(files):

    fd, tmppath = tempfile.mkstemp(
        prefix='fermipy-meritskimmer.', dir='.', suffix='.txt')
    os.close(fd)
    with open(tmppath, 'w') as tmpfile:
        tmpfile.write("\n".join(files))
    return tmppath


def has_object(filelist, key):

    if re.search('\.root?', filelist) is None:
        meritfile = np.loadtxt(filelist, unpack=True, dtype='str')[0]
    else:
        meritfile = filelist

    # Need to use TFile::Open to work on both local and xrootd paths
    f = ROOT.TFile.Open(meritfile)

    ret = f.GetListOfKeys().Contains(key)
    f.Close()
    return ret


def load_friend_chains(chain, friend_chains, txt, nfiles=None):
    """Load a list of trees from a file and add them as friends to the
    chain."""

    if re.search('.root?', txt) is not None:
        c = ROOT.TChain(chain.GetName())
        c.SetDirectory(0)
        c.Add(txt)
        friend_chains.append(c)
        chain.AddFriend(c, rand_str())
        return

    files = np.loadtxt(txt, unpack=True, dtype='str')
    if files.ndim == 0:
        files = np.array([files])
    if nfiles is not None:
        files = files[:nfiles]
    print("Loading %i files..." % len(files))

    c = ROOT.TChain(chain.GetName())
    c.SetDirectory(0)
    for f in files:
        c.Add(f)
    friend_chains.append(c)

    chain.AddFriend(c, rand_str())
    return


def main():

    usage = "Usage: %(prog)s  [files or file lists] [options]"
    description = """Skim a list of merit files.  If run with the
--batch option, the skimming task will be split across N job where N
is the number of input files divided by the value of the files_per_job
option.  Each job generates an output file with the name of the file
defined in the output option appended with the job number
(e.g. [output]_000.root).")
"""
    parser = argparse.ArgumentParser(usage=usage, description=description)

    add_lsf_args(parser)

    parser.add_argument("--friends", default=None, type=str,
                        help="Load friend chains from xrootd")

    parser.add_argument("--nfiles", default=None, type=str,
                        help="Set the range of files to load from the file "
                        "list.  If set to a single integer N the script will "
                        "load the first N files.  Setting this option to a pair "
                        "of integers separated by a / (e.g. 10/20) sets both an "
                        "upper and lower bound on the files that will be loaded.")

    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite existing merit files.")

    parser.add_argument('--selection', default=None,
                        type=str,
                        help='Choose a selection that will be applied to the '
                        'input chain.  Only events passing this selection '
                        'will be written to the output file.')

    parser.add_argument('--output', default=None,
                        type=str, help='Set the name of the output skim '
                        'file.')

    parser.add_argument('--extra_trees', default='jobinfo',
                        type=str,
                        help='Define a list of extra trees that will be copied '
                        'to the output file.')

    parser.add_argument('--fraction', default=1.0,
                        type=float,
                        help='Set the fraction of events in the input chain '
                        'to write to the output file.')

    parser.add_argument('--start_fraction', default=None,
                        type=float, help='')

    parser.add_argument('--files_per_job', default=100,
                        type=int, help='Set the number of files to assign to '
                        'each batch job.')

    parser.add_argument('--branches', default=[], action='append',
                        type=str,
                        help='Set an YAML alias file or XML class definition file '
                        'that will be used to set the output branches.  Only variables '
                        'in this list will be copied to the output skimmed tree.')

    parser.add_argument('--aliases', default=[], action='append',
                        type=str, help='Set a yaml file that contains a '
                        'set of Merit aliases defined as key/value pairs. Also accepts '
                        'event class XML files.  Note '
                        'that the selection string can use any aliases defined '
                        'in this file.')

    parser.add_argument('files', nargs='+', default=None,
                        help='MERIT files or MERIT file lists.')

    args = parser.parse_args()

    ROOT.TFormula.SetMaxima(2000, 2000, 2000)

    # Assemble list of root files
    merit_files = get_files(args.files)

    if args.batch:

        nfiles = len(merit_files)
        njob = int(np.ceil(nfiles / float(args.files_per_job)))
        infiles = []
        outfiles = []
        job_opts = []

        for ijob, i in enumerate(range(0, nfiles, args.files_per_job)):
            file_range = '%i/%i' % (i, min(nfiles, i + args.files_per_job))

            output_dir = os.path.dirname(args.output)
            output_file = splitext(basename(args.output))[0]
            output_file += '_%03i.root' % (ijob)
            output_file = os.path.join(output_dir, output_file)

            infiles += [args.files]
            outfiles += [output_file]
            opts = vars(args).copy()
            opts['nfiles'] = file_range
            opts['output'] = output_file
            del opts['files']
            del opts['batch']
            job_opts += [opts]

        print('Submitting %i jobs...' % (njob))
        submit_jobs('fermipy-merit-skimmer', infiles, job_opts, outfiles,
                    overwrite=args.overwrite)
        sys.exit(0)

    nfiles = args.nfiles
    if not nfiles is None:
        nfiles = [int(t) for t in nfiles.split('/')]

    branches = []
    for f in args.branches:

        if f.endswith('.xml'):
            cuts = get_cuts_from_xml(f)
            branches += get_branches(cuts)
        elif f.endswith('.yaml'):
            cuts = yaml.load(open(f, 'r'))
            if isinstance(cuts, dict):
                branches += get_branches(cuts)
            elif isinstance(cuts, list):
                branches += cuts
        else:
            raise Exception('Invalid file type for branches option.')

    print('Branches: ', branches)

    chain = ROOT.TChain("MeritTuple")
    load_chain(chain, merit_files, nfiles)

    friendChains = []
    if args.friends is not None:
        print("load friends")
        for f in args.friends.split(","):
            load_friend_chains(chain, friendChains, f)

    aliases = load_aliases(args.aliases)
    for k, v in sorted(aliases.items()):
        chain.SetAlias(k, v)

    # If you want to look at the prefilters, need to change these
    print('Selection Cut:', args.selection)
    set_event_list(chain, args.selection, args.fraction, args.start_fraction)

    skimmer = MeritSkimmer(chain)
    outfile = splitext(basename(args.files[0]))[0] + '-skim.root'
    if args.output is not None:
        outfile = args.output
    else:
        outfile = splitext(basename(args.files[0]))[0] + '-skim.root'

    outputFile = ROOT.TFile(outfile, 'RECREATE')

    extra_trees = ''
    if not args.extra_trees is None:
        extra_trees = args.extra_trees

    for t in extra_trees.split(','):

        if len(t) == 0:
            continue

        c = ROOT.TChain(t)
        if not has_object(merit_files[0], t):
            print('Tree ', t, ' not found in input Merit file.')
            continue

        load_chain(c, merit_files, nfiles)
        tree = c.CopyTree('')
        outputFile.cd()
        tree.Write()

    outputFile.cd()
    copy = skimmer.skim(outputFile, branches=branches)

    outputFile.Close()


if __name__ == "__main__":
    main()
