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
from fermipy.jobs.file_archive import FileFlags
from fermipy.diffuse.name_policy import NameFactory
from fermipy.diffuse import defaults as diffuse_defaults

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
                       options=dict(comp=diffuse_defaults.diffuse['binning_yaml'],
                                    data=diffuse_defaults.diffuse['dataset_yaml'],
                                    coordsys=diffuse_defaults.diffuse['coordsys'],
                                    nfiles=(96, 'Number of input files', int),
                                    dry_run=(False, 'Print commands but do not run them', bool)),
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
        links_to_add += self._make_coadd_links()
        for link in links_to_add:
            self.add_link(link)

    @staticmethod
    def _make_parser():
        """Make an argument parser for this chain """
        usage = "fermipy-coadd-split [options]"
        description = "Merge a set of counts cube files"

        parser = argparse.ArgumentParser(usage=usage, description=description)
        return parser

    def _make_coadd_links(self):
        """Make the links to run fermipy-coadd for each energy bin X psf type
        """
        links = []
        for key_e, comp_e in sorted(self.comp_dict.items()):
            for psf_type in sorted(comp_e['psf_types'].keys()):
                key = "%s_%s" % (key_e, psf_type)
                binkey = 'binfile_%s' % key
                argkey = 'args_%s' % key
                self.files.file_args[argkey] = FileFlags.gz_mask
                link = Link('coadd_%s' % key,
                            appname='fermipy-coadd',
                            options=dict(args=([], "List of input files", list),
                                         output=(None, "Output file", str)),
                            mapping={'args': argkey,
                                     'output': binkey},
                            file_args=dict(args=FileFlags.input_mask,
                                           output=FileFlags.output_mask))
                links.append(link)
        return links

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
        outdir_base = os.path.join(NAME_FACTORY.base_dict['basedir'], 'counts_cubes')
        coordsys = input_dict.get('coordsys', 'GAL')

        num_files = input_dict.get('nfiles', 96)
        output_dict = input_dict.copy()
        for key_e, comp_e in sorted(self.comp_dict.items()):
            for psf_type in sorted(comp_e['psf_types'].keys()):
                key = "%s_%s" % (key_e, psf_type)
                suffix = "zmax%i_%s" % (comp_e['zmax'], key)
                ccube_name =\
                    os.path.basename(NAME_FACTORY.ccube(component=suffix,
                                                        coordsys=coordsys))
                binnedfile = os.path.join(outdir_base, ccube_name)
                output_dict['binfile_%s' % key] = os.path.join(outdir_base, ccube_name)
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
        self.update_args(args.__dict__)
        return args


def create_chain_coadd_split(**kwargs):
    """Build and return a `CoaddSplit` object"""
    chain = CoaddSplit(linkname=kwargs.pop('linkname', 'coadd-split'),
                       comp_dict=kwargs.get('comp_dict', None))
    return chain

def main():
    """Entry point for command line use """
    chain = CoaddSplit('coadd-split')
    args = chain.run_argparser(sys.argv[1:])
    chain.run_chain(sys.stdout, args.dry_run)
    chain.finalize(args.dry_run)


if __name__ == "__main__":
    main()
