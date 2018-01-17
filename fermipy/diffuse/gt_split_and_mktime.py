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
from fermipy.diffuse.name_policy import NameFactory, EVCLASS_MASK_DICTIONARY
from fermipy.diffuse.gt_coadd_split import CoaddSplit
from fermipy.diffuse import defaults as diffuse_defaults
from fermipy.diffuse.binning import EVT_TYPE_DICT 
from fermipy.diffuse.timefilter import  MktimeFilterDict


NAME_FACTORY = NameFactory()
try: 
    MKTIME_DICT = MktimeFilterDict.build_from_yamlfile('config/mktime_filters.yaml')
except:
    MKTIME_DICT = MktimeFilterDict(aliases=dict(quality='lat_config==1&&data_qual>0'),
                                   selections=dict(standard='{quality}'))


def make_full_path(basedir, outkey, origname):
    """Make a full file path"""
    return os.path.join(basedir, outkey, os.path.basename(origname).replace('.fits','_%s.fits'%outkey))



class SplitAndMktime(Chain):
    """Small class to split, apply mktime and bin data according to some user-provided specification
    """
    default_options = dict(comp=diffuse_defaults.residual_cr['comp'],
                           data=diffuse_defaults.residual_cr['dataset_yaml'],
                           coordsys=diffuse_defaults.residual_cr['coordsys'],
                           hpx_order_max=diffuse_defaults.residual_cr['hpx_order_binning'],
                           ft1file=diffuse_defaults.residual_cr['ft1file'],
                           scfile=diffuse_defaults.residual_cr['ft2file'],
                           evclass=(128, 'Event class bit mask', int),
                           outdir=('counts_cubes', 'Output directory', str),
                           outkey=(None, 'Key for this particular output file', str),
                           pfiles=(None, 'Directory for .par files', str),
                           do_ltsum=(False, 'Sum livetime cube files', bool),                           
                           scratch=(None, 'Scratch area', str),
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
        parser = argparse.ArgumentParser(usage="fermipy-split-and-mktime [options]",
                                         description="Run gtselect, gtmktime and gtbin together")

        Chain.__init__(self, linkname,
                       appname='fermipy-split-and-mktime',
                       links=[],
                       options=SplitAndMktime.default_options.copy(),
                       argmapper=self._map_arguments,
                       parser=parser,
                       **kwargs)

        if self.comp_dict is not None:
            self.update_links(self.comp_dict)
        self.set_links_job_archive()


    def update_links(self, comp_dict):
        """Build the links in this chain from the binning specification
        """
        self.comp_dict = comp_dict
        links_to_add = []
        links_to_add += self._make_energy_select_links()
        links_to_add += self._make_PSF_select_and_bin_links()
        for link in links_to_add:
            self.add_link(link)

    def _make_energy_select_links(self):
        """Make the links to run gtselect for each energy bin """
        links = []
        for key, comp in sorted(self.comp_dict.items()):
            select_filekey = 'selectfile_%s' % key
            self.files.file_args[select_filekey] = FileFlags.rm_mask
            zmax = comp['zmax']
            link_sel = Gtlink('gtselect_%s' % key,
                              appname='gtselect',
                              mapping={'infile': 'ft1file',
                                       'outfile': select_filekey},
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
            
            links.append(link_sel)

            for mktimekey in comp['mktimefilters']:
                mktime_filekey = 'mktime_%s_%s' % (key, mktimekey)
                ltcube_filekey = 'ltcube_%s_%s' % (key, mktimekey)
                filterstring = MKTIME_DICT[mktimekey]
                self.files.file_args[mktime_filekey] = FileFlags.rm_mask
                link_mktime = Gtlink('gtmktime_%s_%s' % (key, mktimekey),
                                     appname='gtmktime',
                                     mapping={'evfile': select_filekey,
                                              'outfile': mktime_filekey},
                                     options={'evfile': (None, 'Input FT1 File', str),
                                              'outfile': (None, 'Output FT1 File', str),
                                              'scfile': (None, 'Input FT2 file', str),
                                              'roicut': (False, 'Apply ROI-based zenith angle cut', bool),
                                              'filter': (filterstring, 'Filter expression', str),
                                              'pfiles': (None, "PFILES directory", str)},
                                     file_args=dict(evfile=FileFlags.in_stage_mask,
                                                    scfile=FileFlags.in_stage_mask,
                                                    outfile=FileFlags.out_stage_mask))

                link_ltcube = Gtlink('gtltcube_%s_%s' % (key, mktimekey),
                                     appname='gtltcube',
                                     mapping={'evfile': mktime_filekey,
                                              'outfile': ltcube_filekey},
                                     options={'evfile': (None, 'Input FT1 File', str),
                                              'scfile': (None, 'Input FT2 file', str),
                                              'outfile': (None, 'Output Livetime cube File', str),
                                              'dcostheta': (0.025, 'Step size in cos(theta)', float),
                                              'binsz' : (1., 'Pixel size (degrees)', float),
                                              'phibins' : (0, 'Number of phi bins', int),
                                              'zmin' : (0, 'Minimum zenith angle', float),
                                              'zmax' : (zmax, 'Maximum zenith angle', float),
                                              'pfiles': (None, "PFILES directory", str)},
                                     file_args=dict(evfile=FileFlags.in_stage_mask,
                                                    scfile=FileFlags.in_stage_mask,
                                                    outfile=FileFlags.out_stage_mask))
                links.append(link_mktime)
                links.append(link_ltcube)

        return links

    def _make_PSF_select_and_bin_links(self):
        """Make the links to run gtselect and gtbin for each psf type"""
        links = []
        for key_e, comp_e in sorted(self.comp_dict.items()):
            emin = math.pow(10., comp_e['log_emin'])
            emax = math.pow(10., comp_e['log_emax'])
            enumbins = comp_e['enumbins']
            zmax = comp_e['zmax']

            for mktimekey in comp_e['mktimefilters']:
                mktime_filekey = 'mktime_%s_%s' % (key_e, mktimekey)
                ltcube_filekey = 'ltcube_%s_%s' % (key_e, mktimekey)

                for evtclass in comp_e['evtclasses']:
                    evtclassint = EVCLASS_MASK_DICTIONARY[evtclass]
                    for psf_type, psf_dict in sorted(comp_e['psf_types'].items()):
                        key = "%s_%s_%s_%s" % (key_e, mktimekey, evtclass, psf_type)
                        selectkey_out = 'selectfile_%s' % key
                        binkey = 'binfile_%s' % key
                        hpxorder_key = 'hpxorder_%s' % key
                        self.files.file_args[mktime_filekey] = FileFlags.rm_mask
                        self.files.file_args[selectkey_out] = FileFlags.rm_mask
                        self.files.file_args[binkey] = FileFlags.gz_mask | FileFlags.internal_mask
                        select_link = Gtlink('gtselect_%s' % key,
                                             appname='gtselect',
                                             mapping={'infile': mktime_filekey,
                                                      'outfile': selectkey_out},
                                             options={'evtype': (EVT_TYPE_DICT[psf_type], "PSF type", int),
                                                      'zmax': (zmax, "Zenith angle cut", float),
                                                      'emin': (emin, "Minimum energy", float),
                                                      'emax': (emax, "Maximum energy", float),
                                                      'infile': (None, 'Input FT1 File', str),
                                                      'outfile': (None, 'Output FT1 File', str),
                                                      'evclass': (evtclassint, "Event class", int),
                                                      'pfiles': (None, "PFILES directory", str)},
                                             file_args=dict(infile=FileFlags.in_stage_mask,
                                                            outfile=FileFlags.out_stage_mask))
                        bin_link = Gtlink('gtbin_%s' % key,
                                          appname='gtbin',
                                          mapping={'evfile': selectkey_out,
                                                   'outfile': binkey,
                                                   'hpx_order' : hpxorder_key},
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

        NAME_FACTORY.update_base_dict(input_dict['data'])

        coordsys = input_dict.get('coordsys')
        outdir = input_dict.get('outdir')
        outkey = input_dict.get('outkey')
        if outdir is None or outkey is None:
            return None
        
        output_dict = input_dict.copy()
        output_dict['filter'] = input_dict.get('mktimefilter')
        output_dict.pop('evclass')

        for key_e, comp_e in sorted(self.comp_dict.items()):
            zcut = "zmax%i"%comp_e['zmax']
            kwargs_select = dict(zcut=zcut,
                                 ebin=key_e,
                                 psftype='ALL',
                                 coordsys=coordsys)
            selectfile = make_full_path(outdir, outkey, NAME_FACTORY.select(**kwargs_select) )
            output_dict['selectfile_%s' % key_e] = selectfile
            for mktimekey in comp_e['mktimefilters']:
                kwargs_mktime = kwargs_select.copy()
                kwargs_mktime['mktime'] = mktimekey
                output_dict['mktime_%s_%s' % (key_e, mktimekey)] = make_full_path(outdir, outkey, NAME_FACTORY.mktime(**kwargs_mktime))
                output_dict['ltcube_%s_%s' % (key_e, mktimekey)] = make_full_path(outdir, outkey, NAME_FACTORY.ltcube(**kwargs_mktime))

                for evtclass in comp_e['evtclasses']:
                    for psf_type, psf_dict in sorted(comp_e['psf_types'].items()):
                        key = "%s_%s_%s_%s"%(key_e, mktimekey, evtclass, psf_type)
                        kwargs_bin = kwargs_mktime.copy()
                        kwargs_bin['psftype'] = psf_type
                        kwargs_bin['coordsys'] = coordsys
                        kwargs_bin['evclass'] = evtclass
                        output_dict['selectfile_%s' % key] = make_full_path(outdir, outkey, NAME_FACTORY.select(**kwargs_bin))
                        output_dict['binfile_%s' % key] = make_full_path(outdir, outkey, NAME_FACTORY.ccube(**kwargs_bin))
                        output_dict['hpxorder_%s' % key] = min(input_dict['hpx_order_max'], psf_dict['hpx_order'])

        return output_dict

    def run_argparser(self, argv):
        """Initialize a link with a set of arguments using argparser
        """
        if self._parser is None:
            raise ValueError('SplitAndMktime was not given a parser on initialization')
        args = self._parser.parse_args(argv)
        self.update_links(yaml.safe_load(open(args.comp)))
        self.update_args(args.__dict__)
        return args


class ConfigMaker_SplitAndMktime(ConfigMaker):
    """Small class to generate configurations for SplitAndMktime
    """
    default_options = dict(comp=diffuse_defaults.residual_cr['comp'],
                           data=diffuse_defaults.residual_cr['dataset_yaml'],
                           coordsys=diffuse_defaults.diffuse['coordsys'],
                           hpx_order_max=diffuse_defaults.diffuse['hpx_order_ccube'],
                           ft1file=diffuse_defaults.residual_cr['ft1file'],
                           ft2file=diffuse_defaults.residual_cr['ft2file'],
                           evclass=(128, 'Event class bit mask', int),
                           pfiles=(None, 'Directory for .par files', str),
                           do_ltsum=(False, 'Sum livetime cube files', bool),                           
                           scratch=(None, 'Path to scratch area', str),
                           dry_run=(False, 'Print commands but do not run them', bool))

    def __init__(self, chain, **kwargs):
        """C'tor
        """
        ConfigMaker.__init__(self, chain,
                             options=kwargs.get('options', self.default_options.copy()))

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
        self.link.update_args(args)
        return self.link.args

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        datafile = args['data']
        if datafile is None or datafile == 'None':
            return job_configs
        NAME_FACTORY.update_base_dict(args['data'])

        inputfiles = create_inputlist(args['ft1file'])
        outdir_base = os.path.join(NAME_FACTORY.base_dict['basedir'], 'counts_cubes')
        data_ver = NAME_FACTORY.base_dict['data_ver']

        nfiles = len(inputfiles)
        for idx, infile in enumerate(inputfiles):
            key = "%06i" % idx
            key_scfile = "%03i" % (idx+1)
            output_dir = os.path.join(outdir_base, key)
            try:
                os.mkdir(output_dir)
            except OSError:
                pass
            scfile = args['ft2file'].replace('.lst', '_%s.fits' % key_scfile)
            logfile = os.path.join(output_dir, 'scatter_mk_%s_%s.log' % (data_ver, key))

            job_configs[key] = dict(ft1file=infile,
                                    scfile=scfile,
                                    comp=args['comp'],
                                    hpx_order_max=args['hpx_order_max'],
                                    outdir=outdir_base,
                                    outkey=key,
                                    logfile=logfile,
                                    pfiles=output_dir)

        return job_configs

def create_chain_split_and_mktime(**kwargs):
    """Build and return a `Link` object that can invoke split-and-mktime"""
    linkname = kwargs.pop('linkname', 'split-and-mktime')
    chain = SplitAndMktime(**kwargs)
    return chain

def create_sg_split_and_mktime(**kwargs):
    """Build and return a `fermipy.jobs.ScatterGather` object that can invoke this script"""
    linkname = kwargs.pop('linkname', 'split-and-mktime')
    chain = SplitAndMktime(linkname, **kwargs)
    appname = kwargs.pop('appname', 'fermipy-split-and-mktime-sg')

    lsf_args = {'W': 1500,
                'R': '\"select[rhel60 && !fell]\"'}

    usage = "%s [options]"%(appname)
    description = "Prepare data for diffuse all-sky analysis"

    config_maker = ConfigMaker_SplitAndMktime(chain)
    lsf_sg = build_sg_from_link(chain, config_maker,
                                lsf_args=lsf_args,
                                usage=usage,
                                description=description,
                                linkname=linkname,
                                appname=appname,
                                **kwargs)
    return lsf_sg

def main_single(): 
    """Entry point for command line use for single job """
    chain = SplitAndMktime('SplitAndMktime')
    args = chain.run_argparser(sys.argv[1:])
    chain.run_chain(sys.stdout, args.dry_run)
    chain.finalize(args.dry_run)


def main_batch():
    """Entry point for command line use for dispatching batch jobs """
    lsf_sg = create_sg_split_and_mktime()
    lsf_sg(sys.argv)

if __name__ == "__main__":
    main_single()
