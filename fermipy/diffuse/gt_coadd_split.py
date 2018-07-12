#!/usr/bin/env python
#

"""
Prepare data for diffuse all-sky analysis
"""

import os
import copy
from collections import OrderedDict

import yaml

from fermipy.jobs.utils import is_null
from fermipy.jobs.link import Link
from fermipy.jobs.chain import Chain
from fermipy.jobs.scatter_gather import ScatterGather
from fermipy.jobs.slac_impl import make_nfs_path

from fermipy.diffuse.utils import create_inputlist
from fermipy.diffuse.name_policy import NameFactory
from fermipy.diffuse.binning import Component
from fermipy.diffuse import defaults as diffuse_defaults

from fermipy.diffuse.job_library import Gtlink_ltsum, Link_FermipyCoadd


NAME_FACTORY = NameFactory()


def _make_input_file_list(binnedfile, num_files):
    """Make the list of input files for a particular energy bin X psf type """
    outdir_base = os.path.abspath(os.path.dirname(binnedfile))
    outbasename = os.path.basename(binnedfile)
    filelist = ""
    for i in range(num_files):
        split_key = "%06i" % i
        output_dir = os.path.join(outdir_base, split_key)
        filepath = os.path.join(output_dir,
                                outbasename.replace('.fits', '_%s.fits' % split_key))
        filelist += ' %s' % filepath
    return filelist


class CoaddSplit(Chain):
    """Small class to merge counts cubes for a series of binning components

    This chain consists multiple `Link` objects:

    coadd-EBIN-ZCUT-FILTER-EVTYPE : `_Link_FermipyCoadd`
        Link to coadd data of a particular type.

    """
    appname = 'fermipy-coadd-split'
    linkname_default = 'coadd-split'
    usage = '%s [options]' % (appname)
    description = 'Merge a set of counts cube files'

    default_options = dict(comp=diffuse_defaults.diffuse['comp'],
                           data=diffuse_defaults.diffuse['data'],
                           do_ltsum=(False, 'Sum livetime cube files', bool),
                           nfiles=(96, 'Number of input files', int),
                           dry_run=(False, 'Print commands but do not run them', bool))

    __doc__ += Link.construct_docstring(default_options)

    def __init__(self, **kwargs):
        """C'tor
        """
        super(CoaddSplit, self).__init__(**kwargs)
        self.comp_dict = None

    def _map_arguments(self, args):
        """Map from the top-level arguments to the arguments provided to
        the indiviudal links """
        comp_file = args.get('comp', None)
        datafile = args.get('data', None)
        do_ltsum = args.get('do_ltsum', False)

        NAME_FACTORY.update_base_dict(datafile)
        outdir_base = os.path.join(NAME_FACTORY.base_dict['basedir'], 'counts_cubes')
        num_files = args.get('nfiles', 96)

        self.comp_dict = yaml.safe_load(open(comp_file))
        coordsys = self.comp_dict.pop('coordsys')

        for key_e, comp_e in sorted(self.comp_dict.items()):

            if 'mktimefilters' in comp_e:
                mktimelist = comp_e['mktimefilters']
            else:
                mktimelist = ['none']

            if 'evtclasses' in comp_e:
                evtclasslist_vals = comp_e['evtclasses']
            else:
                evtclasslist_vals = [NAME_FACTORY.base_dict['evclass']]

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
                    linkname = 'ltsum_%s_%s' % (key_e, mktimekey)
                    self._set_link(likname, Gtlink_ltsum,
                                   infile1=ltsum_listfile,
                                   infile2=None,
                                   outfile=ltsum_outfile,
                                   logfile=os.path.join(outdir_base, "%s.log" % linkname))

                for evtclassval in evtclasslist_vals:
                    for psf_type in sorted(comp_e['psf_types'].keys()):
                        fullkey = "%s_%s_%s_%s"%(key_e, mktimekey, evtclassval, psf_type)
                        linkname = 'coadd_%s' % (fullkey)
                        kwargs_bin = kwargs_mktime.copy()
                        kwargs_bin['psftype'] = psf_type
                        kwargs_bin['evclass'] = evtclassval
                        ccube_name =\
                            os.path.basename(NAME_FACTORY.ccube(**kwargs_bin))
                        outputfile = os.path.join(outdir_base, ccube_name)
                        args = _make_input_file_list(outputfile, num_files)
                        
                        self._set_link(linkname,
                                       Link_FermipyCoadd,
                                       args=args,
                                       output=outputfile,
                                       logfile=os.path.join(outdir_base, "%s.log" % linkname))

                                                 
class CoaddSplit_SG(ScatterGather):
    """Small class to generate configurations for fermipy-coadd

   """
    appname = 'fermipy-coadd-split-sg'
    usage = "%s [options]" % (appname)
    description = "Submit fermipy-coadd-split- jobs in parallel"
    clientclass = Link_FermipyCoadd

    job_time = 300

    default_options = dict(comp=diffuse_defaults.diffuse['comp'],
                           data=diffuse_defaults.diffuse['data'],
                           ft1file=(None, 'Input FT1 file', str))

    __doc__ += Link.construct_docstring(default_options)
                                               
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
            if not mktimelist:
                mktimelist.append('none')
            evtclasslist_keys = copy.copy(comp.evtclasses)
            if not evtclasslist_keys:
                evtclasslist_vals = [NAME_FACTORY.base_dict['evclass']]
            else:
                evtclasslist_vals = copy.copy(evtclasslist_keys)

            for mktimekey in mktimelist:
                for evtclassval in evtclasslist_vals:
                    fullkey = comp.make_key(
                        '%s_%s_{ebin_name}_%s_{evtype_name}' %
                        (evtclassval, zcut, mktimekey))

                    name_keys = dict(zcut=zcut,
                                     ebin=comp.ebin_name,
                                     psftype=comp.evtype_name,
                                     coordsys=comp.coordsys,
                                     irf_ver=NAME_FACTORY.irf_ver(),
                                     mktime=mktimekey,
                                     evclass=evtclassval,
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
    """Register these classes with the `LinkFactory` """
    CoaddSplit.register_class()
    CoaddSplit_SG.register_class()
