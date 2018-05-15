# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Prepare data for diffuse all-sky analysis
"""
from __future__ import absolute_import, division, print_function

import os
import math

import yaml

from fermipy.jobs.utils import is_null
from fermipy.jobs.link import Link
from fermipy.jobs.chain import Chain
from fermipy.jobs.scatter_gather import ScatterGather
from fermipy.jobs.slac_impl import make_nfs_path

from fermipy.diffuse.utils import create_inputlist
from fermipy.diffuse.name_policy import NameFactory
from fermipy.diffuse import defaults as diffuse_defaults
from fermipy.diffuse.binning import EVT_TYPE_DICT

from fermipy.diffuse.job_library import Gtlink_select,\
    Gtlink_bin, Gtexpcube2_SG

from fermipy.diffuse.gt_coadd_split import CoaddSplit_SG


NAME_FACTORY = NameFactory()


def make_full_path(basedir, outkey, origname):
    """Make a full file path"""
    return os.path.join(basedir, outkey, os.path.basename(
        origname).replace('.fits', '_%s.fits' % outkey))


class SplitAndBin(Chain):
    """Small class to split and bin data according to some user-provided specification

    This chain consists multiple `Link` objects:

    select-energy-EBIN-ZCUT : `Gtlink_select`
        Initial splitting by energy bin and zenith angle cut

    select-type-EBIN-ZCUT-FILTER-TYPE : `Gtlink_select`
        Refinement of selection from event types

    bin-EBIN-ZCUT-FILTER-TYPE : `Gtlink_bin`
        Final binning of the data for each event type

    """
    appname = 'fermipy-split-and-bin'
    linkname_default = 'split-and-bin'
    usage = '%s [options]' % (appname)
    description = 'Run gtselect and gtbin together'

    default_options = dict(data=diffuse_defaults.diffuse['data'],
                           comp=diffuse_defaults.diffuse['comp'],
                           hpx_order_max=diffuse_defaults.diffuse['hpx_order_ccube'],
                           ft1file=(None, 'Input FT1 file', str),
                           evclass=(128, 'Event class bit mask', int),
                           outdir=('counts_cubes_cr', 'Base name for output files', str),
                           outkey=(None, 'Key for this particular output file', str),
                           pfiles=(None, 'Directory for .par files', str),
                           scratch=(None, 'Scratch area', str),
                           dry_run=(False, 'Print commands but do not run them', bool))

    __doc__ += Link.construct_docstring(default_options)

    def __init__(self, **kwargs):
        """C'tor
        """
        super(SplitAndBin, self).__init__(**kwargs)
        self.comp_dict = None

    def _map_arguments(self, args):
        """Map from the top-level arguments to the arguments provided to
        the indiviudal links """
        comp_file = args.get('comp', None)
        datafile = args.get('data', None)
        if is_null(comp_file):
            return
        if is_null(datafile):
            return

        NAME_FACTORY.update_base_dict(datafile)

        outdir = args.get('outdir', None)
        outkey = args.get('outkey', None)
        ft1file = args['ft1file']

        if is_null(outdir) or is_null(outkey):
            return
        pfiles = os.path.join(outdir, outkey)

        self.comp_dict = yaml.safe_load(open(comp_file))
        coordsys = self.comp_dict.pop('coordsys')
        full_out_dir = make_nfs_path(os.path.join(outdir, outkey))

        for key_e, comp_e in sorted(self.comp_dict.items()):
            emin = math.pow(10., comp_e['log_emin'])
            emax = math.pow(10., comp_e['log_emax'])
            enumbins = comp_e['enumbins']
            zmax = comp_e['zmax']
            zcut = "zmax%i" % comp_e['zmax']
            evclassstr = NAME_FACTORY.base_dict['evclass']
            kwargs_select = dict(zcut=zcut,
                                 ebin=key_e,
                                 psftype='ALL',
                                 coordsys=coordsys,
                                 mktime='none')
            selectfile_energy = make_full_path(outdir, outkey, NAME_FACTORY.select(**kwargs_select))
            linkname = 'select-energy-%s-%s' % (key_e, zcut)
            self._set_link(linkname, Gtlink_select,
                           infile=ft1file,
                           outfile=selectfile_energy,
                           zmax=zmax,
                           emin=emin,
                           emax=emax,
                           evclass=NAME_FACTORY.evclassmask(evclassstr),
                           pfiles=pfiles,
                           logfile=os.path.join(full_out_dir, "%s.log" % linkname))

            if 'evtclasses' in comp_e:
                evtclasslist_vals = comp_e['evtclasses']
            else:
                evtclasslist_vals = [NAME_FACTORY.base_dict['evclass']]

            for evtclassval in evtclasslist_vals:
                for psf_type, psf_dict in sorted(comp_e['psf_types'].items()):
                    linkname_select = 'select-type-%s-%s-%s-%s' % (
                        key_e, zcut, evtclassval, psf_type)
                    linkname_bin = 'bin-%s-%s-%s-%s' % (key_e, zcut, evtclassval, psf_type)
                    hpx_order = psf_dict['hpx_order']
                    kwargs_bin = kwargs_select.copy()
                    kwargs_bin['psftype'] = psf_type
                    selectfile_psf = make_full_path(
                        outdir, outkey, NAME_FACTORY.select(**kwargs_bin))
                    binfile = make_full_path(outdir, outkey, NAME_FACTORY.ccube(**kwargs_bin))
                    self._set_link(linkname_select, Gtlink_select,
                                   infile=selectfile_energy,
                                   outfile=selectfile_psf,
                                   zmax=zmax,
                                   emin=emin,
                                   emax=emax,
                                   evtype=EVT_TYPE_DICT[psf_type],
                                   evclass=NAME_FACTORY.evclassmask(evtclassval),
                                   pfiles=pfiles,
                                   logfile=os.path.join(full_out_dir, "%s.log" % linkname_select))

                    self._set_link(linkname_bin, Gtlink_bin,
                                   coordsys=coordsys,
                                   hpx_order=hpx_order,
                                   evfile=selectfile_psf,
                                   outfile=binfile,
                                   emin=emin,
                                   emax=emax,
                                   enumbins=enumbins,
                                   pfiles=pfiles,
                                   logfile=os.path.join(full_out_dir, "%s.log" % linkname_bin))


class SplitAndBin_SG(ScatterGather):
    """Small class to generate configurations for SplitAndBin
    """
    appname = 'fermipy-split-and-bin-sg'
    usage = "%s [options]" % (appname)
    description = "Prepare data for diffuse all-sky analysis"
    clientclass = SplitAndBin

    job_time = 1500

    default_options = dict(comp=diffuse_defaults.diffuse['comp'],
                           data=diffuse_defaults.diffuse['data'],
                           hpx_order_max=diffuse_defaults.diffuse['hpx_order_ccube'],
                           ft1file=(None, 'Input FT1 file', str),
                           scratch=(None, 'Path to scratch area', str))

    __doc__ += Link.construct_docstring(default_options)

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        job_configs = {}

        comp_file = args.get('comp', None)
        if comp_file is not None:
            comp_dict = yaml.safe_load(open(comp_file))
            coordsys = comp_dict.pop('coordsys')
            for v in comp_dict.values():
                v['coordsys'] = coordsys
        else:
            return job_configs

        NAME_FACTORY.update_base_dict(args['data'])

        inputfiles = create_inputlist(args['ft1file'])
        outdir_base = os.path.join(NAME_FACTORY.base_dict['basedir'], 'counts_cubes')

        for idx, infile in enumerate(inputfiles):
            key = "%06i" % idx
            output_dir = os.path.join(outdir_base, key)
            try:
                os.mkdir(output_dir)
            except OSError:
                pass
            logfile = make_nfs_path(os.path.join(output_dir, 'scatter_%s.log' % key))
            job_configs[key] = comp_dict.copy()
            job_configs[key].update(dict(ft1file=infile,
                                         comp=args['comp'],
                                         hpx_order_max=args['hpx_order_max'],
                                         outdir=outdir_base,
                                         outkey=key,
                                         logfile=logfile,
                                         pfiles=output_dir))

        return job_configs


class SplitAndBinChain(Chain):
    """Chain to run split and bin and then make exposure cubes

    This chain consists of:

    split-and-bin : `SplitAndBin_SG`
        Chain to make the binned counts maps for each input file
    
    coadd-split : `CoaddSplit_SG`
        Link to co-add the binnec counts maps files

    expcube2 : `Gtexpcube2_SG`
        Link to make the corresponding binned exposure maps

    """
    appname = 'fermipy-split-and-bin-chain'
    linkname_default = 'split-and-bin-chain'
    usage = '%s [options]' % (appname)
    description = 'Run split-and-bin, coadd-split and exposure'

    default_options = dict(data=diffuse_defaults.diffuse['data'],
                           comp=diffuse_defaults.diffuse['comp'],
                           ft1file=diffuse_defaults.diffuse['ft1file'],
                           hpx_order_ccube=diffuse_defaults.diffuse['hpx_order_ccube'],
                           hpx_order_expcube=diffuse_defaults.diffuse['hpx_order_expcube'],
                           scratch=diffuse_defaults.diffuse['scratch'],
                           dry_run=diffuse_defaults.diffuse['dry_run'])

    __doc__ += Link.construct_docstring(default_options)

    def _map_arguments(self, args):
        """Map from the top-level arguments to the arguments provided to
        the indiviudal links """
        data = args.get('data')
        comp = args.get('comp')
        ft1file = args.get('ft1file')
        scratch = args.get('scratch', None)
        dry_run = args.get('dry_run', None)

        self._set_link('split-and-bin', SplitAndBin_SG,
                       comp=comp, data=data,
                       hpx_order_max=args.get('hpx_order_ccube', 9),
                       ft1file=ft1file,
                       scratch=scratch,
                       dry_run=dry_run)

        self._set_link('coadd-split', CoaddSplit_SG,
                       comp=comp, data=data,
                       ft1file=ft1file)

        self._set_link('expcube2', Gtexpcube2_SG,
                       comp=comp, data=data,
                       hpx_order_max=args.get('hpx_order_expcube', 5),
                       dry_run=dry_run)

def register_classes():
    """Register these classes with the `LinkFactory` """
    SplitAndBin.register_class()
    SplitAndBin_SG.register_class()
    SplitAndBinChain.register_class()
