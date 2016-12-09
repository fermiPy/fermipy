# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

from fermipy.jobs.gt_chain import comlink, gtlink, gtchain


def test_comlink():
    defaults = dict(arg_float=4.0, arg_in='test.in', arg_out='test.out')
    mapping = dict(arg_in='alias')
    kwargs = dict(appname='dummy',
                  defaults=defaults,
                  mapping=mapping,
                  input_file_args=['arg_in'],
                  output_file_args=['arg_out'])
    link = comlink('link', **kwargs)
    return link


def test_gtlink():
    pass

def test_gtchain():
    pass
