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

from collections import OrderedDict

from fermipy.jobs.file_archive import FileFlags
from fermipy.jobs.utils import is_null, is_not_null
from fermipy.jobs.chain import Chain, insert_app_config, purge_dict
from fermipy.jobs.gtlink import Gtlink
from fermipy.jobs.scatter_gather import ConfigMaker, build_sg_from_link
from fermipy.jobs.slac_impl import make_nfs_path, get_slac_default_args, Slac_Interface

from fermipy.diffuse.utils import create_inputlist
from fermipy.diffuse.name_policy import NameFactory
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
    appname = 'fermipy-split-and-mktime'
    linkname_default = 'split-and-mktime'
    usage = '%s [options]' %(appname)
    description='Run gtselect and gtbin together'

    default_options = dict(comp=diffuse_defaults.diffuse['comp'],
                           data=diffuse_defaults.diffuse['data'],
                           hpx_order_max=diffuse_defaults.diffuse['hpx_order_ccube'],
                           ft1file=diffuse_defaults.diffuse['ft1file'],
                           ft2file=diffuse_defaults.diffuse['ft2file'],
                           evclass=(128, 'Event class bit mask', int),
                           outdir=('counts_cubes', 'Output directory', str),
                           outkey=(None, 'Key for this particular output file', str),
                           pfiles=(None, 'Directory for .par files', str),
                           do_ltsum=(False, 'Sum livetime cube files', bool),                           
                           scratch=(None, 'Scratch area', str),
                           dry_run=(False, 'Print commands but do not run them', bool))

    def __init__(self, **kwargs):
        """C'tor
        """
        linkname, init_dict = self._init_dict(**kwargs)
        super(SplitAndMktime, self).__init__(linkname, **init_dict)
        self.comp_dict = None

    def _register_link_classes(self):    
         from fermipy.diffuse.job_library import register_classes as register_library
         register_library()

    def _map_arguments(self, input_dict):
        """Map from the top-level arguments to the arguments provided to
        the indiviudal links """
        print (input_dict)
        comp_file = input_dict.get('comp', None)
        datafile = input_dict.get('data', None)
        do_ltsum = input_dict.get('do_ltsum', False)
        o_dict = OrderedDict()
        if is_null(comp_file):
            return o_dict
        if is_null(datafile):
            return o_dict

        NAME_FACTORY.update_base_dict(input_dict['data'])

        outdir = input_dict.get('outdir')
        outkey = input_dict.get('outkey')
        ft1file = input_dict['ft1file']
        ft2file = input_dict['ft2file']
        if is_null(outdir) or is_null(outkey):
            return o_dict
        pfiles = os.path.join(outdir, outkey) 
        
        self.comp_dict = yaml.safe_load(open(comp_file))
        coordsys = self.comp_dict.pop('coordsys')

        for key_e, comp_e in sorted(self.comp_dict.items()):
            emin = math.pow(10., comp_e['log_emin'])
            emax = math.pow(10., comp_e['log_emax'])
            enumbins = comp_e['enumbins']
            zmax = comp_e['zmax']
            zcut = "zmax%i"%comp_e['zmax']
            evclassstr = NAME_FACTORY.base_dict['evclass']    
            
            kwargs_select = dict(zcut=zcut,
                                 ebin=key_e,
                                 psftype='ALL',
                                 coordsys=coordsys)
            linkname = 'select-energy-%s-%s'%(key_e, zcut)
            selectfile_energy = make_full_path(outdir, outkey, NAME_FACTORY.select(**kwargs_select) )
            insert_app_config(o_dict, linkname,
                              'gtselect',
                              linkname=linkname,
                              infile=ft1file,
                              outfile=selectfile_energy,
                              zmax=zmax,
                              emin=emin,
                              emax=emax,
                              evclass=NAME_FACTORY.evclassmask(evclassstr))

            if comp_e.has_key('mktimefilters'):
                mktimefilters = comp_e['mktimefilters']
            else:
                mktimefilters = ['none']

            for mktimekey in mktimefilters:
                kwargs_mktime = kwargs_select.copy()
                kwargs_mktime['mktime'] = mktimekey
                filterstring = MKTIME_DICT[mktimekey]
                mktime_file = make_full_path(outdir, outkey, NAME_FACTORY.mktime(**kwargs_mktime))
                ltcube_file = make_full_path(outdir, outkey, NAME_FACTORY.ltcube(**kwargs_mktime))
                linkname_mktime = 'mktime-%s-%s-%s'%(key_e, zcut, mktimekey)
                linkname_ltcube = 'mktime-%s-%s-%s'%(key_e, zcut, mktimekey)

                insert_app_config(o_dict, linkname_mktime,
                                  'gtmktime',
                                  linkname=linkname_mktime,
                                  evfile=selectfile_energy,
                                  outfile=mktime_file,
                                  scfile=ft2file,
                                  filter=filterstring,
                                  pfiles=pfiles)
                insert_app_config(o_dict, linkname_ltcube,
                                  'gtltcube',
                                  linkname=linkname_ltcube,
                                  evfile=mktime_file,
                                  outfile=ltcube_file,
                                  scfile=ft2file,
                                  zmax=zmax,
                                  pfiles=pfiles)
                
                if comp_e.has_key('evtclasses'):
                    evtclasslist_keys = comp_e['evtclasses']
                    evtclasslist_vals = comp_e['evtclasses']
                    evtclasslist = comp_e['evtclasses']
                else:
                    evtclasslist_keys = ['default']
                    evtclasslist_vals = [NAME_FACTORY.base_dict['evclass']]
                    evtclasslist = ['default']

                for evtclasskey, evtclassval in zip(evtclasslist_keys, evtclasslist_vals):
                    for psf_type, psf_dict in sorted(comp_e['psf_types'].items()):
                        linkname_select = 'select-type-%s-%s-%s-%s-%s'%(key_e, zcut, mktimekey, evtclassval, psf_type)
                        linkname_bin = 'bin-%s-%s-%s-%s-%s'%(key_e, zcut, mktimekey, evtclassval, psf_type)
                        hpx_order = psf_dict['hpx_order']
                        kwargs_bin = kwargs_mktime.copy()
                        kwargs_bin['psftype'] = psf_type
                        kwargs_bin['coordsys'] = coordsys
                        kwargs_bin['evclass'] = evtclassval
                        selectfile_psf = make_full_path(outdir, outkey, NAME_FACTORY.select(**kwargs_bin))
                        binfile_psf = make_full_path(outdir, outkey, NAME_FACTORY.ccube(**kwargs_bin))
                        hpxorder_psf = min(input_dict['hpx_order_max'], psf_dict['hpx_order'])
                        linkname_select = 'select-type-%s-%s-%s-%s'%(key_e, zcut, mktimekey, psf_type)
                        linkname_bin = 'bin-%s-%s-%s-%s'%(key_e, zcut, mktimekey, psf_type)

                        insert_app_config(o_dict, linkname_select,
                                          'gtselect',
                                          linkname=linkname_select,
                                          infile=selectfile_energy,
                                          outfile=selectfile_psf,
                                          zmax=zmax,
                                          emin=emin,
                                          emax=emax,
                                          evtype=EVT_TYPE_DICT[psf_type],
                                          evclass=NAME_FACTORY.evclassmask(evtclassval),
                                          pfiles=pfiles)
                        insert_app_config(o_dict, linkname_bin,
                                          'gtbin',
                                          linkname=linkname_bin,
                                          coordsys=coordsys,
                                          hpx_order=hpx_order,
                                          evfile=selectfile_psf,
                                          outfile=binfile_psf,
                                          emin=emin,
                                          emax=emax,
                                          enumbins=enumbins,
                                          pfiles=pfiles)

        return o_dict


class SplitAndMktime_SG(ConfigMaker):
    """Small class to generate configurations for SplitAndMktime
    """
    appname = 'fermipy-split-and-mktime-sg'
    usage = "%s [options]" % (appname)
    description = "Prepare data for diffuse all-sky analysis"
    clientclass = SplitAndMktime

    batch_args = get_slac_default_args()    
    batch_interface = Slac_Interface(**batch_args)

    default_options = dict(comp=diffuse_defaults.diffuse['comp'],
                           data=diffuse_defaults.diffuse['data'],
                           hpx_order_max=diffuse_defaults.diffuse['hpx_order_ccube'],
                           ft1file=diffuse_defaults.diffuse['ft1file'],
                           ft2file=diffuse_defaults.diffuse['ft2file'],
                           do_ltsum=diffuse_defaults.diffuse['do_ltsum'],    
                           scratch=diffuse_defaults.diffuse['scratch'],   
                           dry_run=diffuse_defaults.diffuse['dry_run'])

    def __init__(self, chain, **kwargs):
        """C'tor
        """
        super(SplitAndMktime_SG, self).__init__(chain,
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




  
class SplitAndMktimeChain(Chain):
    """Small class to split, apply mktime and bin data according to some user-provided specification
    """
    appname = 'fermipy-split-and-mktime-chain'
    linkname_default = 'split-and-mktime-chain'
    usage = '%s [options]' %(appname)
    description='Run split-and-mktime, coadd-split and exposure'

    default_options = dict(data=diffuse_defaults.diffuse['data'],
                           comp=diffuse_defaults.diffuse['comp'],
                           ft1file=diffuse_defaults.diffuse['ft1file'],
                           ft2file=diffuse_defaults.diffuse['ft2file'],
                           hpx_order_ccube=diffuse_defaults.diffuse['hpx_order_ccube'],
                           hpx_order_expcube=diffuse_defaults.diffuse['hpx_order_expcube'],
                           do_ltsum=diffuse_defaults.diffuse['do_ltsum'],
                           scratch=diffuse_defaults.diffuse['scratch'],
                           dry_run=diffuse_defaults.diffuse['dry_run'])
   
    def __init__(self, **kwargs):
        """C'tor
        """
        linkname, init_dict = self._init_dict(**kwargs)
        super(SplitAndMktimeChain, self).__init__(linkname, **init_dict)
        self.comp_dict = None     

    def _register_link_classes(self):    
        from fermipy.diffuse.job_library import register_classes as register_library
        from fermipy.diffuse.gt_coadd_split import CoaddSplit_SG
        register_library()
        SplitAndMktime_SG.register_class()
        CoaddSplit_SG.register_class()

    def _map_arguments(self, input_dict):
        """Map from the top-level arguments to the arguments provided to
        the indiviudal links """
        o_dict = OrderedDict()

        data = input_dict.get('data')
        comp = input_dict.get('comp')
        ft1file = input_dict.get('ft1file')
        ft2file = input_dict.get('ft2file')
        evclass = input_dict.get('evclass')
        scratch=input_dict.get('scratch', None)
        dry_run=input_dict.get('dry_run', None)

        insert_app_config(o_dict, 'split-and-mktime',
                          'fermipy-split-and-mktime-sg',
                          comp=comp, data=data,
                          hpx_order_max=input_dict.get('hpx_order_ccube', 9),
                          ft1file=ft1file,
                          ft2file=ft2file,
                          do_ltsum=input_dict.get('do_ltsum', False),
                          scratch=scratch,
                          dry_run=dry_run)

        insert_app_config(o_dict, 'coadd-split',
                          'fermipy-coadd-split-sg',
                          comp=comp, data=data,
                          ft1file=ft1file)
 
        insert_app_config(o_dict, 'ltsum',
                          'fermipy-gtltsum-sg',
                          comp=comp, data=data, 
                          ft1file=input_dict['ft1file'],
                          dry_run=dry_run)

        insert_app_config(o_dict, 'expcube2',
                          'fermipy-gtltsum-sg',
                          comp=comp, data=data, 
                          hpx_order_max=input_dict.get('hpx_order_expcube', 5),
                          dry_run=dry_run)
     
        return o_dict


def register_classes():
    SplitAndMktime.register_class()
    SplitAndMktime_SG.register_class()
    SplitAndMktimeChain.register_class()
 

