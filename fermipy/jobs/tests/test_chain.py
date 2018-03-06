# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

from fermipy.jobs.file_archive import FileFlags
from fermipy.jobs.chain import Link, Chain


def test_comlink():
    kwargs = dict(appname='dummy',
                  options=dict(arg_float=(4.0, 'a float', float), 
                               arg_in=('test.in', 'an input file', str),  
                               arg_out=('test.out', 'an output file', str)),
                  mapping=dict(arg_in='alias'),
                  file_args=dict(arg_in=FileFlags.input_mask,
                                 arg_out=FileFlags.output_mask))
    link = Link('link', **kwargs)


def test_chain():
    kwargs = dict(appname='dummy',
                  options=dict(arg_float=(4.0, 'a float', float),
                               arg_in=('test.in', 'an input file', str),
                               arg_out=('test.out', 'an output file', str)),
                  mapping=dict(arg_in='alias'),
                  file_args=dict(arg_in=FileFlags.input_mask,
                                 arg_out=FileFlags.output_mask))
    link = Link('link', **kwargs)

    kwargs = dict(options=dict(irfs=('CALDB', 'IRF version', str),
                               expcube=(None, 'Livetime cube file', str),
                               bexpmap=(None, 'Binned exposure map', str),
                               cmap=(None, 'Binned counts map', str),
                               srcmdl=(None, 'Input source model xml file',  str),
                               outfile=(None, 'Output file', str)),
                  file_args=dict(expcube=FileFlags.input_mask, 
                                 cmap=FileFlags.input_mask, 
                                 bexpmap=FileFlags.input_mask, 
                                 srcmdl=FileFlags.input_mask,
                                 outfile=FileFlags.output_mask))
    # This should be a Gtlink, but we only really wanna test the chain functionality here
    link2 = Link('gtsrcmaps', **kwargs)

    def argmapper(args):
        basename = args['basename']
        ret_dict = dict(expcube="%s_ltcube.fits"%basename,
                        cmap="%s_ccube.fits"%basename,
                        bexpmap="%s_bexpmap.fits"%basename,
                        srcmdl="%s_srcmdl.xml"%basename)
        return ret_dict

    chain = Chain('chain', 
                  links=[link, link2], 
                  options=dict(basename=('dummy', 'Base file name', str)),
                  argmapper=argmapper)
    

