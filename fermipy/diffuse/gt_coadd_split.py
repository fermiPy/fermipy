#!/usr/bin/env python
#

"""
Prepare data for diffuse all-sky analysis
"""

import sys
import os
import argparse

import yaml

from fermipy.jobs.chain import Chain, Link
from fermipy.diffuse.name_policy import NameFactory

NAME_FACTORY = NameFactory()


class CoaddSplit(Chain):
    """Small class to merge counts cubes for a series of binning components
    """

    def __init__(self, linkname, comp_dict=None):
        """C'tor
        """
        self.comp_dict = comp_dict
        Chain.__init__(self, linkname,
                       links=[],
                       appname='fermipy-coadd-split',
                       argmapper=self._map_arguments,
                       parser=CoaddSplit._make_parser())
        if comp_dict is not None:
            self.update_links(comp_dict)

    def update_links(self, comp_dict):
        """Build the links in this chain from the binning specification
        """
        self.comp_dict = comp_dict
        links_to_add = []
        links_to_add += CoaddSplit._make_coadd_links(self.comp_dict)
        for link in links_to_add:
            self.add_link(link)
        self._gz_keys = CoaddSplit._make_gz_keys(self.comp_dict)

    @staticmethod
    def _make_parser():
        """Make an argument parser for this chain """
        usage = "fermipy-coadd-split [options]"
        description = "Merge a set of counts cube files"

        parser = argparse.ArgumentParser(usage=usage, description=description)
        parser.add_argument('--comp', type=str, default=None,
                            help='component yaml file')
        parser.add_argument('--data', type=str, default=None,
                            help='datset yaml file')
        parser.add_argument('--coordsys', type=str, default='GAL',
                            help='Coordinate system')
        parser.add_argument('--nfiles', type=int, default=96,
                            help='Number of input files')
        parser.add_argument('--dry_run', action='store_true', default=False,
                            help='Print commands but do not run them')
        return parser

    @staticmethod
    def _make_coadd_links(comp_dict):
        """Make the links to run fermipy-coadd for each energy bin X psf type
        """
        links = []
        for key_e, comp_e in sorted(comp_dict.items()):
            for psf_type in sorted(comp_e['psf_types'].keys()):
                key = "%s_%s" % (key_e, psf_type)
                link = Link('coadd_%s' % key,
                            appname='fermipy-coadd',
                            mapping={'args': 'args_%s' % key,
                                     'output': 'binfile_%s' % key},
                            output_file_args=['output'])
                links.append(link)
        return links

    @staticmethod
    def _make_gz_keys(comp_dict):
        """Make the list of arguments corresponding to files to be compressed """
        lout = []
        for key_e, comp_e in sorted(comp_dict.items()):
            for psf_type in sorted(comp_e['psf_types'].keys()):
                key = "%s_%s" % (key_e, psf_type)
                lout.append('binfile_%s' % key)
        return lout

    @staticmethod
    def _make_input_file_list(binnedfile, num_files):
        """Make the list of input files for a particular energy bin X psf type """
        outdir_base = os.path.dirname(binnedfile)
        outbasename = os.path.basename(binnedfile)
        filelist = ""
        for i in range(num_files):
            split_key = "%06i" % i
            output_dir = os.path.join(outdir_base, split_key)
            filepath = os.path.join(output_dir,
                                    outbasename.replace('.fits', '_%s.fits.gz' % split_key))
            filelist += ' %s' % filepath
        return filelist

    def _map_arguments(self, input_dict):
        """Map from the top-level arguments to the arguments provided to
        the indiviudal links """
        datafile = input_dict.get('data')
        if datafile is None:
            return None
        NAME_FACTORY.update_base_dict(input_dict['data'])
        outdir_base = NAME_FACTORY.base_dict['basedir']
        coordsys = input_dict.get('coordsys', 'GAL')

        num_files = input_dict.get('nfiles')
        output_dict = input_dict.copy()
        for key_e, comp_e in sorted(self.comp_dict.items()):
            for psf_type in sorted(comp_e['psf_types'].keys()):
                key = "%s_%s" % (key_e, psf_type)
                suffix = "zmax%i_%s" % (comp_e['zmax'], key)
                ccube_name =\
                    os.path.basename(NAME_FACTORY.ccube(component='_comp_',
                                                        coordsys='%s_%s' % (coordsys, key)))
                binnedfile = os.path.join(outdir_base, ccube_name)
                output_dict['binfile_%s' % key] = binnedfile.replace('_comp_', suffix)
                output_dict['args_%s' % key] = CoaddSplit._make_input_file_list(
                    binnedfile, num_files)
        return output_dict

    def run_argparser(self, argv):
        """  Initialize a link with a set of arguments using argparser
        """
        if self._parser is None:
            raise ValueError('CoaddSplit was not given a parser on initialization')
        args = self._parser.parse_args(argv)
        self.update_links(yaml.safe_load(open(args.comp)))
        self.update_links_from_single_dict(args.__dict__)
        return args


def main():
    """Entry point for command line use """
    chain = CoaddSplit('coadd-split')
    args = chain.run_argparser(sys.argv[1:])
    chain.run_chain(sys.stdout, args.dry_run)
    chain.finalize(args.dry_run)


if __name__ == "__main__":
    main()
