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

from fermipy.jobs.file_archive import FileFlags
from fermipy.jobs.chain import Chain
from fermipy.jobs.gtlink import Gtlink
from fermipy.jobs.scatter_gather import ConfigMaker
from fermipy.jobs.lsf_impl import build_sg_from_link
from fermipy.diffuse.name_policy import NameFactory
from fermipy.diffuse.gt_coadd_split import CoaddSplit
from fermipy.diffuse import defaults as diffuse_defaults
from fermipy.diffuse.binning import EVT_TYPE_DICT 


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
    if isinstance(arglist, list):
        for arg in arglist:
            if os.path.splitext(arg)[1] == '.lst':
                lines += readlines(arg)
            else:
                lines.append(arg)
    else:
        if os.path.splitext(arglist)[1] == '.lst':
            lines += readlines(arglist)
        else:
            lines.append(arglist)
    return lines


class SplitAndBin(Chain):
    """Small class to split and bin data according to some user-provided specification
    """

    def __init__(self, linkname, comp_dict=None):
        """C'tor
        """
        self.comp_dict = comp_dict
        Chain.__init__(self, linkname,
                       appname='fermipy-split-and-bin',
                       links=[],
                       options=dict(comp=diffuse_defaults.diffuse['binning_yaml'],
                                    coordsys=diffuse_defaults.diffuse['coordsys'],
                                    hpx_order_max=diffuse_defaults.diffuse['hpx_order_ccube'],
                                    ft1file=(None, 'Input FT1 file', str),
                                    evclass=(128, 'Event class bit mask', int),
                                    output=(None, 'Base name for output files', str),
                                    pfiles=(None, 'Directory for .par files', str),
                                    scratch=(None, 'Scratch area', str),
                                    dry_run=(False, 'Print commands but do not run them', bool)),
                       argmapper=self._map_arguments,
                       parser=SplitAndBin._make_parser())
        if comp_dict is not None:
            self.update_links(comp_dict)

    def update_links(self, comp_dict):
        """Build the links in this chain from the binning specification
        """
        self.comp_dict = comp_dict
        links_to_add = []
        links_to_add += self._make_energy_select_links()
        links_to_add += self._make_PSF_select_and_bin_links()
        for link in links_to_add:
            self.add_link(link)

    @staticmethod
    def _make_parser():
        """Make an argument parser for this chain """
        usage = "fermipy-split-and-bin [options]"
        description = "Run gtselect and gtbin together"

        parser = argparse.ArgumentParser(usage=usage, description=description)
        return parser

    def _make_energy_select_links(self):
        """Make the links to run gtselect for each energy bin """
        links = []
        for key, comp in sorted(self.comp_dict.items()):
            outfilekey = 'selectfile_%s' % key
            self.files.file_args[outfilekey] = FileFlags.rm_mask
            link = Gtlink('gtselect_%s' % key,
                          appname='gtselect',
                          mapping={'infile': 'ft1file',
                                   'outfile': outfilekey},
                          options={'emin': (math.pow(10., comp['log_emin']), "Minimum energy",
                                            float),
                                   'emax': (math.pow(10., comp['log_emax']), "Maximum energy",
                                            float),
                                   'infile': (None, 'Input FT1 File', str),
                                   'outfile': (None, 'Output FT1 File', str),
                                   'zmax': (comp['zmax'], "Zenith angle cut", float),
                                   'evclass': (None, "Event Class", int),
                                   'pfiles': (None, "PFILES directory", str)},
                          file_args=dict(infile=FileFlags.in_stage_mask,
                                         outfile=FileFlags.out_stage_mask))
            links.append(link)
        return links

    def _make_PSF_select_and_bin_links(self):
        """Make the links to run gtselect and gtbin for each psf type"""
        links = []
        for key_e, comp_e in sorted(self.comp_dict.items()):
            emin = math.pow(10., comp_e['log_emin'])
            emax = math.pow(10., comp_e['log_emax'])
            enumbins = comp_e['enumbins']
            zmax = comp_e['zmax']
            for psf_type, psf_dict in sorted(comp_e['psf_types'].items()):
                key = "%s_%s" % (key_e, psf_type)
                selectkey_in = 'selectfile_%s' % key_e
                selectkey_out = 'selectfile_%s' % key
                binkey = 'binfile_%s' % key
                self.files.file_args[selectkey_in] = FileFlags.rm_mask
                self.files.file_args[selectkey_out] = FileFlags.rm_mask
                self.files.file_args[binkey] = FileFlags.gz_mask | FileFlags.internal_mask
                select_link = Gtlink('gtselect_%s' % key,
                                     appname='gtselect',
                                     mapping={'infile': selectkey_in,
                                              'outfile': selectkey_out},
                                     options={'evtype': (EVT_TYPE_DICT[psf_type], "PSF type", int),
                                              'zmax': (zmax, "Zenith angle cut", float),
                                              'emin': (emin, "Minimum energy", float),
                                              'emax': (emax, "Maximum energy", float),
                                              'infile': (None, 'Input FT1 File', str),
                                              'outfile': (None, 'Output FT1 File', str),
                                              'evclass': (None, "Event class", int),
                                              'pfiles': (None, "PFILES directory", str)},
                                     file_args=dict(infile=FileFlags.in_stage_mask,
                                                    outfile=FileFlags.out_stage_mask))
                bin_link = Gtlink('gtbin_%s' % key,
                                  appname='gtbin',
                                  mapping={'evfile': selectkey_out,
                                           'outfile': binkey},
                                  options={'algorithm': ('HEALPIX', "Binning alogrithm", str),
                                           'coordsys': ('GAL', "Coordinate system", str),
                                           'hpx_order': (psf_dict['hpx_order'], "HEALPIX ORDER", int),
                                           'evfile': (None, 'Input FT1 File', str),
                                           'outfile': (None, 'Output binned data File', str),
                                           'emin': (emin, "Minimum energy", float),
                                           'emax': (emax, "Maximum energy", float),
                                           'enumbins': (enumbins, "Number of energy bins", int),
                                           'pfiles': (None, "PFILES directory", str)},
                                  file_args=dict(evfile=FileFlags.in_stage_mask,
                                                 outfile=FileFlags.out_stage_mask))
                links += [select_link, bin_link]
        return links


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
        self.update_args(args.__dict__)
        return args


class ConfigMaker_SplitAndBin(ConfigMaker):
    """Small class to generate configurations for SplitAndBin
    """
    default_options = dict(comp=diffuse_defaults.diffuse['binning_yaml'],
                           data=diffuse_defaults.diffuse['dataset_yaml'],
                           coordsys=diffuse_defaults.diffuse['coordsys'],
                           hpx_order_max=diffuse_defaults.diffuse['hpx_order_ccube'],
                           inputlist=(None, 'Input FT1 file', str),
                           scratch=(None, 'Path to scratch area', str))

    def __init__(self, chain, gather, **kwargs):
        """C'tor
        """
        ConfigMaker.__init__(self, chain,
                             options=kwargs.get('options', self.default_options.copy()))
        self.gather = gather

    def make_base_config(self, args):
        """Hook to build a baseline job configuration

        Parameters
        ----------

        args : dict
            Command line arguments, see add_arguments
        """
        comp_file = args.get('comp', None)
        if comp_file is not None:
            comp_dict = yaml.safe_load(open(comp_file))
            self.link.update_links(comp_dict)
            self.gather.update_links(comp_dict)
        self.link.update_args(args)
        self.gather.update_args(args)
        return self.link.args

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        input_config = {}
        job_configs = {}

        NAME_FACTORY.update_base_dict(args['data'])

        inputfiles = create_inputlist(args['inputlist'])
        outdir_base = os.path.join(NAME_FACTORY.base_dict['basedir'], 'counts_cubes')

        nfiles = len(inputfiles)
        for idx, infile in enumerate(inputfiles):
            key = "%06i" % idx
            output_dir = os.path.join(outdir_base, key)
            try:
                os.mkdir(output_dir)
            except OSError:
                pass
            ccube_name =\
                os.path.basename(NAME_FACTORY.ccube(component='_comp_',
                                                    coordsys='%s' % args['coordsys']))
            binnedfile = os.path.join(output_dir, ccube_name).replace('.fits', '_%s.fits' % key)
            binnedfile_gzip = binnedfile + '.gz'
            selectfile = binnedfile.replace('ccube', 'select')
            logfile = os.path.join(output_dir, 'scatter_%s.log' % key)
            outfiles = [selectfile, binnedfile_gzip]
            job_configs[key] = dict(ft1file=infile,
                                    comp=args['comp'],
                                    hpx_order_max=args['hpx_order_max'],
                                    output=binnedfile,
                                    logfile=logfile,
                                    outfiles=outfiles,
                                    pfiles=output_dir)

        output_config = dict(comp=args['comp'],
                             data=args['data'],
                             coordsys=args['coordsys'],
                             nfiles=nfiles,
                             logfile=os.path.join(outdir_base, 'gather.log'),
                             dry_run=args['dry_run'])

        return input_config, job_configs, output_config

def create_chain_split_and_bin(**kwargs):
    """Make a `fermipy.jobs.SplitAndBin` """
    chain = SplitAndBin(linkname=kwargs.pop('linkname', 'SplitAndBin'),
                        comp_dict=kwargs.get('comp_dict', None))
    return chain

def create_sg_split_and_bin(**kwargs):
    """Build and return a `fermipy.jobs.ScatterGather` object that can invoke this script"""
    linkname = kwargs.pop('linkname', 'split-and-bin')
    chain = SplitAndBin('%s.split'%linkname)
    gather = CoaddSplit('%s.coadd'%linkname)
    appname = kwargs.pop('appname', 'fermipy-split-and-bin-sg')

    lsf_args = {'W': 1500,
                'R': 'rhel60'}

    usage = "%s [options]"%(appname)
    description = "Prepare data for diffuse all-sky analysis"

    config_maker = ConfigMaker_SplitAndBin(chain, gather)
    lsf_sg = build_sg_from_link(chain, config_maker,
                                lsf_args=lsf_args,
                                usage=usage,
                                description=description,
                                gather=gather,
                                linkname=linkname,
                                appname=appname,
                                **kwargs)
    return lsf_sg

def main_single():
    """Entry point for command line use for single job """
    chain = SplitAndBin('SplitAndBin')
    args = chain.run_argparser(sys.argv[1:])
    chain.run_chain(sys.stdout, args.dry_run)
    chain.finalize(args.dry_run)


def main_batch():
    """Entry point for command line use for dispatching batch jobs """
    lsf_sg = create_sg_split_and_bin()
    lsf_sg(sys.argv)

if __name__ == "__main__":
    main_single()
