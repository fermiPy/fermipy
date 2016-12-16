# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Prepare data for diffuse all-sky analysis
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
import math

import yaml

from fermipy.jobs.chain import Chain
from fermipy.jobs.gtlink import Gtlink
from fermipy.jobs.scatter_gather import ConfigMaker
from fermipy.jobs.lsf_impl import build_sg_from_link
from fermipy.diffuse.name_policy import NameFactory


PSF_TYPE_DICT = dict(PSF0=4, PSF1=8, PSF2=16, PSF3=32)
NAME_FACTORY = NameFactory()


def readlines(arg):
    """Read lines from a file into a list.
    
    Removes whitespace and lines that start with '#'
    """
    fin = open(arg)
    lines_in = fin.readlines()
    fin.close()
    lines_out = []
    for line in lines_in:
        line = line.strip()
        if len(line) == 0 or line[0] == '#':
            continue
        lines_out.append(line)
    return lines_out


def create_inputlist(arglist):
    """Read lines from a file and makes a list of file names.

    Removes whitespace and lines that start with '#'
    Recursively read all files with the extension '.lst'
    """
    lines = []
    for arg in arglist:
        if os.path.splitext(arg)[1] == '.lst':
            lines += readlines(arg)
        else:
            lines.append(arg)
    return lines


class SplitAndBin(Chain):
    """Small class to split and bin data according to some user-provided specification
    """

    def __init__(self, linkname, comp_dict=None):
        """C'tor
        """
        self.comp_dict = comp_dict
        Chain.__init__(self, linkname,
                       links=[],
                       appname='fermipy-split-and-bin',
                       argmapper=self._map_arguments,
                       parser=SplitAndBin._make_parser())
        if comp_dict is not None:
            self.update_links(comp_dict)

    def update_links(self, comp_dict):
        """Build the links in this chain from the binning specification
        """
        self.comp_dict = comp_dict
        links_to_add = []
        links_to_add += SplitAndBin._make_energy_select_links(self.comp_dict)
        links_to_add += SplitAndBin._make_PSF_select_and_bin_links(self.comp_dict)
        for link in links_to_add:
            self.add_link(link)
        self._rm_keys = SplitAndBin._make_rm_keys(self.comp_dict)
        self._gz_keys = SplitAndBin._make_gz_keys(self.comp_dict)

    @staticmethod
    def _make_parser():
        """Make an argument parser for this chain """
        usage = "fermipy-split-and-bin [options]"
        description = "Run gtselect and gtbin together"

        parser = argparse.ArgumentParser(usage=usage, description=description)
        parser.add_argument('--comp', type=str, default=None,
                            help='component yaml file')
        parser.add_argument('--evclass', type=int, default=128,
                            help='Event class bit mask')
        parser.add_argument('--coordsys', type=str, default='GAL',
                            help='Coordinate system')
        parser.add_argument('--ft1file', type=str, default=None,
                            help='Input FT1 file')
        parser.add_argument('--output', type=str, default=None,
                            help='Base name for output files')
        parser.add_argument('--pfiles', type=str, default=None,
                            help='Directory for .par files')
        parser.add_argument('--dry_run', action='store_true', default=False,
                            help='Print commands but do not run them')
        return parser

    @staticmethod
    def _make_energy_select_links(comp_dict):
        """Make the links to run gtselect for each energy bin """
        links = []
        for key, comp in sorted(comp_dict.items()):
            link = Gtlink('gtselect_%s' % key,
                          appname='gtselect',
                          mapping={'infile': 'ft1file',
                                   'outfile': 'selectfile_%s' % key},
                          defaults={'emin': math.pow(10., comp['log_emin']),
                                    'emax': math.pow(10., comp['log_emax']),
                                    'zmax': comp['zmax'],
                                    'evclass': None,
                                    'pfiles': None},
                          input_file_args=['infile'],
                          output_file_args=['outfile'])
            links.append(link)
        return links

    @staticmethod
    def _make_PSF_select_and_bin_links(comp_dict):
        """Make the links to run gtselect and gtbin for each psf type"""
        links = []
        for key_e, comp_e in sorted(comp_dict.items()):
            emin = math.pow(10., comp_e['log_emin'])
            emax = math.pow(10., comp_e['log_emax'])
            enumbins = comp_e['enumbins']
            zmax = comp_e['zmax']
            for psf_type, psf_dict in sorted(comp_e['psf_types'].items()):
                key = "%s_%s" % (key_e, psf_type)
                select_link = Gtlink('gtselect_%s' % key,
                                     appname='gtselect',
                                     mapping={'infile': 'selectfile_%s' % key_e,
                                              'outfile': 'selectfile_%s' % key},
                                     defaults={'evtype': PSF_TYPE_DICT[psf_type],
                                               'zmax': zmax,
                                               'emin': emin,
                                               'emax': emax,
                                               'evclass': None,
                                               'pfiles': None},
                                     input_file_args=['infile'],
                                     output_file_args=['outfile'])
                bin_link = Gtlink('gtbin_%s' % key,
                                  appname='gtbin',
                                  mapping={'evfile': 'selectfile_%s' % key,
                                           'outfile': 'binfile_%s' % key},
                                  defaults={'algorithm': 'HEALPIX',
                                            'coordsys': 'GAL',
                                            'hpx_order': psf_dict['hpx_order'],
                                            'emin': emin,
                                            'emax': emax,
                                            'enumbins': enumbins,
                                            'pfiles': None},
                                  input_file_args=['evfile'],
                                  output_file_args=['outfile'])
                links += [select_link, bin_link]
        return links

    @staticmethod
    def _make_rm_keys(comp_dict):
        """ Make the list of arguments corresponding to files to be removed """
        lout = []
        for key_e, comp_e in sorted(comp_dict.items()):
            lout.append('selectfile_%s' % key_e)
            for psf_type in sorted(comp_e['psf_types'].keys()):
                key = "%s_%s" % (key_e, psf_type)
                lout.append('selectfile_%s' % key)
        return lout

    @staticmethod
    def _make_gz_keys(comp_dict):
        """Make the list of arguments corresponding to files to be compressed """
        lout = []
        for key_e, comp_e in sorted(comp_dict.items()):
            for psf_type in sorted(comp_e['psf_types'].keys()):
                key = "%s_%s" % (key_e, psf_type)
                lout.append('binfile_%s' % key)
        return lout

    def _map_arguments(self, input_dict):
        """Map from the top-level arguments to the arguments provided to
        the indiviudal links """
        if self.comp_dict is None:
            return None
        outbase = input_dict.get('output')
        if outbase is None:
            return None
        binnedfile = "%s" % (outbase)
        selectfile = binnedfile.replace('ccube', 'select')

        output_dict = input_dict.copy()
        for key_e, comp_e in sorted(self.comp_dict.items()):
            suffix = "zmax%i_%s" % (comp_e['zmax'], key_e)
            output_dict['selectfile_%s' % key_e] = selectfile.replace('_comp_', suffix)
            for psf_type in sorted(comp_e['psf_types'].keys()):
                key = "%s_%s" % (key_e, psf_type)
                suffix = "zmax%i_%s" % (comp_e['zmax'], key)
                output_dict['selectfile_%s' % key] = selectfile.replace('_comp_', suffix)
                output_dict['binfile_%s' % key] = binnedfile.replace('_comp_', suffix)
        return output_dict

    def run_argparser(self, argv):
        """Initialize a link with a set of arguments using argparser
        """
        if self._parser is None:
            raise ValueError('SplitAndBin was not given a parser on initialization')
        args = self._parser.parse_args(argv)
        self.update_links(yaml.safe_load(open(args.comp)))
        self.update_links_from_single_dict(args.__dict__)
        return args


class ConfigMaker_SplitAndBin(ConfigMaker):
    """Small class to generate configurations for SplitAndBin
    """

    def __init__(self, chain):
        """C'tor
        """
        ConfigMaker.__init__(self)
        self.chain = chain

    def add_arguments(self, parser, action):
        """Hook to add arguments to the command line argparser

        Parameters:
        ----------------
        parser : `argparse.ArgumentParser'
            Object we are filling

        action : str
            String specifing what we want to do

        This adds the following arguments:

        --comp     : binning component definition yaml file
        --data     : datset definition yaml file
        --coordsys : Coordinate system ['GAL' | 'CEL']
        input      : Input File List
        """
        parser.add_argument('--comp', type=str, default=None,
                            help='binning component definition yaml file')
        parser.add_argument('--data', type=str, default=None,
                            help='datset definition yaml file')
        parser.add_argument('--coordsys', type=str, default='GAL', help='Coordinate system')
        parser.add_argument('input', nargs='+', default=None, help='Input File List')

    def make_base_config(self, args):
        """Hook to build a baseline job configuration

        Parameters:
        ----------------
        args : `argparse.Namespace'
            Command line arguments, see add_arguments
        """
        self.chain.update_links(yaml.safe_load(open(args.comp)))
        self.chain.update_links_from_single_dict(args.__dict__)
        return self.chain.options

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        input_config = {}
        job_configs = {}

        NAME_FACTORY.update_base_dict(args.data)

        inputfiles = create_inputlist(args.input)
        outdir_base = NAME_FACTORY.base_dict['basedir']

        for idx, infile in enumerate(inputfiles):
            key = "%06i" % idx
            output_dir = os.path.join(outdir_base, key)
            try:
                os.mkdir(output_dir)
            except OSError:
                pass
            ccube_name =\
                os.path.basename(NAME_FACTORY.ccube(component='_comp_',
                                                    coordsys='%s_%s' % (args.coordsys, key)))
            binnedfile = os.path.join(output_dir, ccube_name)
            binnedfile_gzip = binnedfile + '.gz'
            selectfile = binnedfile.replace('ccube', 'select')
            logfile = os.path.join(output_dir, 'scatter_%s.log' % key)
            outfiles = [selectfile, binnedfile_gzip]
            job_configs[key] = dict(ft1file=infile,
                                    output=binnedfile,
                                    logfile=logfile,
                                    outfiles=outfiles,
                                    pfiles=output_dir)

        output_config = {}

        return input_config, job_configs, output_config


def build_scatter_gather():
    """Build and return a ScatterGather object that can invoke this script"""
    chain = SplitAndBin('split-and-bin')

    lsf_args = {'W': 1500,
                'R': 'rhel60'}

    usage = "fermipy-split-and-bin-sg [options] input"
    description = "Prepare data for diffuse all-sky analysis"

    config_maker = ConfigMaker_SplitAndBin(chain)
    lsf_sg = build_sg_from_link(chain, config_maker,
                                lsf_args=lsf_args,
                                usage=usage,
                                description=description)
    return lsf_sg


def main_single():
    """Entry point for command line use for single job """
    chain = SplitAndBin('SplitAndBin')
    args = chain.run_argparser(sys.argv[1:])
    chain.run_chain(sys.stdout, args.dry_run)
    chain.finalize(args.dry_run)


def main_batch():
    """Entry point for command line use for dispatching batch jobs """
    lsf_sg = build_scatter_gather()
    lsf_sg(sys.argv)

if __name__ == "__main__":
    main_single()
