# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

from fermipy.jobs.chain import Link, Chain


def test_comlink():
    defaults = dict(arg_float=4.0, arg_in='test.in', arg_out='test.out')
    mapping = dict(arg_in='alias')
    kwargs = dict(appname='dummy',
                  defaults=defaults,
                  mapping=mapping,
                  input_file_args=['arg_in'],
                  output_file_args=['arg_out'])
    link = Link('link', **kwargs)

def test_chain():
    defaults = dict(arg_float=4.0, arg_in='test.in', arg_out='test.out')
    mapping = dict(arg_in='alias')
    kwargs = dict(appname='dummy',
                  defaults=defaults,
                  mapping=mapping,
                  input_file_args=['arg_in'],
                  output_file_args=['arg_out'])
    link = Link('link', **kwargs)
    
    options=dict(irfs='CALDB', expcube=None,
                 bexpmap=None, cmap=None,
                 srcmdl=None, outfile=None)
    kwargs = dict(options=options,
                  flags=['gzip'],
                  input_file_args=['expcube', 'cmap', 'bexpmap', 'srcmdl'],
                  output_file_args=['outfile'])
    # This should be a Gtlink, but we only really wanna test the chain functionality here
    link2 = Link('gtsrcmaps', **kwargs)

    def argmapper(args):
        basename = args.basename
        ret_dict = dict(expcube="%s_ltcube.fits"%basename,
                        cmap="%s_ccube.fits"%basename,
                        bexpmap="%s_bexpmap.fits"%basename,
                        srcmdl="%s_srcmdl.xml"%basename)
        return ret_dict

    chain = Chain('chain', 
                  links=[link, link2], 
                  options=dict(basename=None))
    
