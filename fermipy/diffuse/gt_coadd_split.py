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
from fermipy.jobs.gtlink import Gtlink
from fermipy.jobs.file_archive import FileFlags
from fermipy.diffuse.name_policy import NameFactory
from fermipy.diffuse import defaults as diffuse_defaults

NAME_FACTORY = NameFactory()


class CoaddSplit(Chain):
    """Small class to merge counts cubes for a series of binning components
    """
    default_options = dict(comp=diffuse_defaults.diffuse['binning_yaml'],
                           data=diffuse_defaults.diffuse['dataset_yaml'],
                           coordsys=diffuse_defaults.diffuse['coordsys'],
                           do_ltsum=(False, 'Sum livetime cube files', bool),
                           nfiles=(96, 'Number of input files', int),                                    
                           dry_run=(False, 'Print commands but do not run them', bool))

    def __init__(self, linkname, **kwargs):
        """C'tor
        """
        comp_file = kwargs.get('comp', None)
        if comp_file:
            self.comp_dict = yaml.safe_load(open(comp_file))
        else:
            self.comp_dict = None
        job_archive = kwargs.get('job_archive', None)
        parser = argparse.ArgumentParser(usage="fermipy-coadd-split [options]",
                                         description="Merge a set of counts cube files")
        Chain.__init__(self, linkname,
                       links=[],
                       options=CoaddSplit.default_options.copy(),
                       appname='fermipy-coadd-split',
                       argmapper=self._map_arguments,
                       parser=parser,
                       **kwargs)

        if self.comp_dict is not None:
            self.update_links(self.comp_dict)
        self.set_links_job_archive()

    def update_links(self, comp_dict, do_ltsum=False):
        """Build the links in this chain from the binning specification
        """
        self.comp_dict = comp_dict
        links_to_add = []
        links_to_add += self._make_coadd_links(do_ltsum)
        for link in links_to_add:
            self.add_link(link)

    def _make_coadd_links(self, do_ltsum):
        """Make the links to run fermipy-coadd for each energy bin X psf type
        """
        links = []
        for key_e, comp_e in sorted(self.comp_dict.items()):
            
            for mktimekey in comp_e['mktimefilters']:

                if do_ltsum:
                    ltsum_listfile = 'ltsumlist_%s_%s' % (key_e, mktimekey)
                    ltsum_outfile = 'ltsum_%s_%s' % (key_e, mktimekey)
                    link_ltsum = Gtlink('ltsum_%s_%s' % (key_e, mktimekey),
                                        appname='gtltsum',
                                        mapping={'infile1':ltsum_listfile,
                                                 'outfile':ltsum_outfile},
                                        options=dict(infile1=(None, "Livetime cube 1 or list of files", str),
                                                     infile2=("none", "Livetime cube 2", str),
                                                     outfile=(None, "Output file", str)),
                                        file_args=dict(infile1=FileFlags.input_mask,
                                                       outfile=FileFlags.output_mask)) 
                    links.append(link_ltsum)
                for evtclass in comp_e['evtclasses']:
                    for psf_type in sorted(comp_e['psf_types'].keys()):
                        key = "%s_%s_%s_%s" % (key_e, mktimekey, evtclass, psf_type)
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

    @staticmethod
    def _make_ltcube_file_list(ltsumfile, num_files):
        """Make the list of input files for a particular energy bin X psf type """
        outdir_base = os.path.dirname(ltsumfile)
        outbasename = os.path.basename(ltsumfile)
        lt_list_file = ltsumfile.replace('fits', 'lst')
        outfile = open(lt_list_file, 'w!')
        filelist = ""
        for i in range(num_files):
            split_key = "%06i" % i
            output_dir = os.path.join(NAME_FACTORY.base_dict['basedir'], 'counts_cubes', split_key)
            filepath = os.path.join(output_dir, outbasename.replace('.fits', '_%s.fits' % split_key))
            outfile.write(filepath)
            outfile.write("\n")
        outfile.close()
        return '@'+lt_list_file

    def _map_arguments(self, input_dict):
        """Map from the top-level arguments to the arguments provided to
        the indiviudal links """
        datafile = input_dict.get('data')
        if datafile is None or datafile == 'None':
            return None
        NAME_FACTORY.update_base_dict(datafile)
        outdir_base = os.path.join(NAME_FACTORY.base_dict['basedir'], 'counts_cubes')
        coordsys = input_dict.get('coordsys', 'GAL')

        num_files = input_dict.get('nfiles', 96)
        output_dict = input_dict.copy()
        if self.comp_dict is None:
            return output_dict

        for key_e, comp_e in sorted(self.comp_dict.items()):
            for mktimekey in comp_e['mktimefilters']:
                zcut = "zmax%i" % comp_e['zmax']
                kwargs_mktime = dict(zcut=zcut,
                                     ebin=key_e,
                                     psftype='ALL',
                                     coordsys=coordsys,
                                     mktime=mktimekey)
                if self.args['do_ltsum']:                    
                    ltsumname = os.path.join(NAME_FACTORY.base_dict['basedir'], 
                                             NAME_FACTORY.ltcube(**kwargs_mktime))
                    output_dict['ltsum_%s_%s' % (key_e, mktimekey)] = ltsumname
                    output_dict['ltsumlist_%s_%s' % (key_e, mktimekey)] = CoaddSplit._make_ltcube_file_list(
                        ltsumname, num_files)
                for evtclass in comp_e['evtclasses']:
                    for psf_type in sorted(comp_e['psf_types'].keys()):
                        kwargs_bin = kwargs_mktime.copy()
                        kwargs_bin['psftype'] = psf_type
                        kwargs_bin['evclass'] = evtclass
                        key = "%s_%s_%s_%s" % (key_e, mktimekey, evtclass, psf_type)
                        ccube_name =\
                            os.path.basename(NAME_FACTORY.ccube(**kwargs_bin))
                        binnedfile = os.path.join(outdir_base, ccube_name)
                        output_dict['binfile_%s' % key] = binnedfile
                        output_dict['args_%s' % key] = CoaddSplit._make_input_file_list(
                            binnedfile, num_files)

        return output_dict

    def run_argparser(self, argv):
        """  Initialize a link with a set of arguments using argparser
        """
        if self._parser is None:
            raise ValueError('CoaddSplit was not given a parser on initialization')
        args = self._parser.parse_args(argv)
        self.update_links(yaml.safe_load(open(args.comp)), args.do_ltsum)
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
