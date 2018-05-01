#!/usr/bin/env python
#

"""
Prepare data for diffuse all-sky analysis
"""

import sys
import os
import argparse
import copy

import yaml

from collections import OrderedDict

from fermipy.jobs.utils import is_null, is_not_null
from fermipy.jobs.link import Link 
from fermipy.jobs.chain import Chain, insert_app_config
from fermipy.jobs.scatter_gather import ConfigMaker
from fermipy.jobs.gtlink import Gtlink
from fermipy.jobs.file_archive import FileFlags
from fermipy.jobs.slac_impl import make_nfs_path

from fermipy.diffuse.utils import create_inputlist
from fermipy.diffuse.name_policy import NameFactory
from fermipy.diffuse.binning import Component
from fermipy.diffuse import defaults as diffuse_defaults

NAME_FACTORY = NameFactory()

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


class CoaddSplit(Chain):
    """Small class to merge counts cubes for a series of binning components
    """
    appname = 'fermipy-coadd-split'
    linkname_default = 'coadd-split'
    usage = '%s [options]' %(appname)
    description='Merge a set of counts cube files'

    default_options = dict(comp=diffuse_defaults.diffuse['comp'],
                           data=diffuse_defaults.diffuse['data'],
                           do_ltsum=(False, 'Sum livetime cube files', bool),
                           nfiles=(96, 'Number of input files', int),                                    
                           dry_run=(False, 'Print commands but do not run them', bool))

    def __init__(self, **kwargs):
        """C'tor
        """
        linkname, init_dict = self._init_dict(**kwargs)
        super(CoaddSplit, self).__init__(linkname, **init_dict)
        self.comp_dict = None

    def _register_link_classes(self):    
        from fermipy.diffuse.job_library import register_classes as register_library
        register_library()

    def _map_arguments(self, input_dict):
        """Map from the top-level arguments to the arguments provided to
        the indiviudal links """
        comp_file = input_dict.get('comp', None)
        datafile = input_dict.get('data', None)
        do_ltsum = input_dict.get('do_ltsum', False)
        o_dict = OrderedDict()
        if is_null(comp_file):
            return o_dict
        if is_null(datafile):
            return o_dict

        NAME_FACTORY.update_base_dict(datafile)
        outdir_base = os.path.join(NAME_FACTORY.base_dict['basedir'], 'counts_cubes')
        num_files = input_dict.get('nfiles', 96)
     
        self.comp_dict = yaml.safe_load(open(comp_file))
        coordsys = self.comp_dict.pop('coordsys')

        for key_e, comp_e in sorted(self.comp_dict.items()):
            
            if comp_e.has_key('mktimefilters'):
                mktimelist = comp_e['mktimefilters']
            else:
                mktimelist = ['none']

            if comp_e.has_key('evtclasses'):
                evtclasslist_keys = comp_e['evtclasses']
                evtclasslist_vals = comp_e['evtclasses']
                evtclasslist = comp_e['evtclasses']
            else:
                evtclasslist_keys = ['default']
                evtclasslist_vals = [NAME_FACTORY.base_dict['evclass']]
                evtclasslist = ['default']

            for mktimekey in mktimelist:
                zcut = "zmax%i" % comp_e['zmax']
                kwargs_mktime = dict(zcut=zcut,
                                     ebin=key_e,
                                     psftype='ALL',
                                     coordsys=coordsys,
                                     mktime=mktimekey)
                
                if do_ltsum:
                    ltsum_listfile = 'ltsumlist_%s_%s' % (key_e, mktimekey)
                    ltsum_outfile = 'ltsum_%s_%s' % (key_e, mktimekey)
                    insert_app_config(o_dict, 'gtltsum',
                                      'gtltsum',
                                      infile1=ltsum_listfile,
                                      infile2=None,
                                      outfile=ltsum_outfile)

                for evtclasskey, evtclassval in zip(evtclasslist_keys, evtclasslist_vals):
                    for psf_type in sorted(comp_e['psf_types'].keys()):
                        kwargs_bin = kwargs_mktime.copy()
                        kwargs_bin['psftype'] = psf_type
                        kwargs_bin['evclass'] = NAME_FACTORY.evclassmask(evtclassval)
                        key = "%s_%s_%s_%s" % (key_e, mktimekey, evtclasskey, psf_type)
                        ccube_name =\
                            os.path.basename(NAME_FACTORY.ccube(**kwargs_bin))
                        outputfile = os.path.join(outdir_base, ccube_name)
                        args = _make_input_file_list(ccube_name, num_files)
                        insert_app_config(o_dict, 'coadd',
                                          'fermipy-coadd',
                                          args=args,
                                          output=outputfile)

        return o_dict


class CoaddSplit_SG(ConfigMaker):
    """Small class to generate configurations for fermipy-coadd

    This takes the following arguments:
    --comp     : binning component definition yaml file
    --data     : datset definition yaml file
    --ft1file  : Input list of ft1 files
    """
    appname = 'fermipy-coadd-split-sg'
    usage = "%s [options]" % (appname)
    description = "Submit fermipy-coadd-split- jobs in parallel"
    clientclass = CoaddSplit

    job_time = 300

    default_options = dict(comp=diffuse_defaults.diffuse['comp'],
                           data=diffuse_defaults.diffuse['data'],
                           ft1file=(None, 'Input FT1 file', str))

    def __init__(self, link, **kwargs):
        """C'tor
        """
        super(CoaddSplit_SG, self).__init__(link,
                                            options=kwargs.get('options', self.default_options.copy()))

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        components = Component.build_from_yamlfile(args['comp'])

        datafile = args['data']
        if datafile is None or datafile == 'None':
            return job_configs
        NAME_FACTORY.update_base_dict(args['data'])
        outdir_base = os.path.join(NAME_FACTORY.base_dict['basedir'], 'counts_cubes')

        inputfiles = create_inputlist(args['ft1file'])
        num_files = len(inputfiles)

        for comp in components:
            zcut = "zmax%i" % comp.zmax
            
            mktimelist = copy.copy(comp.mktimefilters)
            if len(mktimelist) == 0:
                mktimelist.append('none')
            evtclasslist_keys = copy.copy(comp.evtclasses)
            if len(evtclasslist_keys) == 0:
                evtclasslist_keys.append('default')
                evtclasslist_vals = [NAME_FACTORY.base_dict['evclass']]
            else:
                evtclasslist_vals = copy.copy(evtclasslist_keys)

            for mktimekey in mktimelist:
                for evtclasskey, evtclassval in zip(evtclasslist_keys, evtclasslist_vals):       
                    fullkey = comp.make_key('%s_%s_{ebin_name}_%s_{evtype_name}'%(evtclassval, zcut, mktimekey))

                    name_keys = dict(zcut=zcut,
                                     ebin=comp.ebin_name,
                                     psftype=comp.evtype_name,
                                     coordsys=comp.coordsys,
                                     irf_ver=NAME_FACTORY.irf_ver(),
                                     mktime=mktimekey,
                                     evclass=NAME_FACTORY.evclassmask(evtclassval),
                                     fullpath=True)

                    ccube_name = os.path.basename(NAME_FACTORY.ccube(**name_keys))
                    outfile = os.path.join(outdir_base, ccube_name)
                    infiles = _make_input_file_list(outfile, num_files)
                    logfile = make_nfs_path(outfile.replace('.fits', '.log'))
                    job_configs[fullkey] = dict(args=infiles,
                                                output=outfile,
                                                logfile=logfile)

        return job_configs


def register_classes():
    CoaddSplit.register_class()
    CoaddSplit_SG.register_class()

