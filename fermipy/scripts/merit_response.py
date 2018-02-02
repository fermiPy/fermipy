# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import os
import copy
import re
import yaml
import sys
import array
import string
import random
from os.path import splitext, basename
import argparse
import numpy as np
import ROOT

from ..batch import submit_jobs, add_lsf_args
from ..validate.utils import get_cuts_from_xml, load_aliases, load_chain
from ..validate.utils import get_files

from astropy.table import Table, Column
from astropy.io import fits


def calc_aeff(chain, cut, ngen, logebins, cthbins):

    zdir = get_vector(chain, '-McZDir', cut)
    loge = get_vector(chain, 'McLogEnergy', cut)

    nevent = np.histogram2d(loge, zdir, bins=[logebins, cthbins])[0]

    ngen = ngen / nevent.size
    eff = nevent / ngen
    eff_var = eff * (1 - eff) / ngen
    return eff * 6.0, eff_var**0.5 * 6.0


def get_vector(chain, var, cut=None, nentries=None, first_entry=0):

    if cut is None:
        cut = ''
    chain.SetEstimate(chain.GetEntries())
    if nentries is None:
        nentries = chain.GetEntries()
    ncut = chain.Draw('%s' % (var), cut, 'goff', nentries, first_entry)
    return copy.deepcopy(np.frombuffer(chain.GetV1(),
                                       count=ncut, dtype='double'))


def getGeneratedEvents(chain):
    NGen_sum = 0
    vref = {}

    vref['trigger'] = array.array('i', [0])
    vref['generated'] = array.array('i', [0])
    vref['version'] = array.array('f', [0.0])
    vref['revision'] = array.array('f', [0.0])
    vref['patch'] = array.array('f', [0.0])
    chain.SetBranchAddress('trigger', vref['trigger'])
    chain.SetBranchAddress('generated', vref['generated'])
    chain.SetBranchAddress('version', vref['version'])

    if chain.GetListOfBranches().Contains('revision'):
        chain.SetBranchAddress('revision', vref['revision'])

    if chain.GetListOfBranches().Contains('patch'):
        chain.SetBranchAddress('patch', vref['patch'])

    for i in range(chain.GetEntries()):
        chain.GetEntry(i)
        ver = int(vref['version'][0])
        rev = int(vref['revision'][0])
        patch = int(vref['patch'][0])
        NGen = vref['generated'][0]
        NGen_sum += NGen

    return NGen_sum


def main():

    usage = "Usage: %(prog)s  [files or file lists] [options]"
    description = """Compute the instrument response diagnostics from
a set of MC merit files.
"""
    parser = argparse.ArgumentParser(usage=usage, description=description)

    parser.add_argument('--aliases', default=[], action='append',
                        type=str, help='Set a yaml file that contains a '
                        'set of Merit aliases defined as key/value pairs. Also accepts '
                        'event class XML files.  Note '
                        'that the selection string can use any aliases defined '
                        'in this file.')

    parser.add_argument('--output', default='output.fits', type=str,
                        help='Output filename.')

    parser.add_argument('--selection', default=None, type=str, action='append',
                        help='Output filename.')

    parser.add_argument('files', nargs='+', default=None,
                        help='MERIT files or MERIT file lists.')

    args = parser.parse_args()

    ROOT.TFormula.SetMaxima(2000, 2000, 2000)

    # Assemble list of root files
    merit_files = get_files(args.files)

    chain = ROOT.TChain("MeritTuple")
    load_chain(chain, merit_files)

    aliases = load_aliases(args.aliases)
    for k, v in sorted(aliases.items()):
        chain.SetAlias(k, v)

    chain_job = ROOT.TChain('jobinfo')
    load_chain(chain_job, merit_files)

    cthmin = 0.0
    cthmax = 1.0
    cthdelta = 0.1
    cthnbin = int((cthmax - cthmin) / cthdelta)

    logemin = 1.25
    logemax = 5.75
    logedelta = 0.125
    logenbin = int((logemax - logemin) / logedelta)

    logebins = np.linspace(logemin, logemax, logenbin + 1)
    cthbins = np.linspace(cthmin, cthmax, cthnbin + 1)

    ectr = 10**(0.5 * (logebins[1:] + logebins[:-1]))

    cols = [Column(name='name', dtype='S32'),
            Column(name='ectr', dtype='f8', unit='MeV', shape=(36,)),
            Column(name='aeff', dtype='f8', unit='m^2', shape=(36, 10)),
            Column(name='acceptance', dtype='f8', unit='m^2 sr', shape=(36))]

    tab = Table(cols)
    ngen = getGeneratedEvents(chain_job)

    classes = ['SOURCE']
    if args.selection is not None:
        classes = args.selection

    for c in sorted(classes):
        print(c)

        aeff, aeff_err = calc_aeff(chain, c, ngen, logebins, cthbins)
        acc = np.sum(aeff, axis=1) * 2 * np.pi / cthnbin
        tab.add_row([c, ectr, aeff, acc])

    hdulist = fits.HDUList()
    hdulist.append(fits.table_to_hdu(tab))
    hdulist.writeto(args.output, overwrite=True)


if __name__ == "__main__":
    main()
