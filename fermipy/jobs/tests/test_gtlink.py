# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

from fermipy.tests.utils import requires_dependency

try:
    from fermipy.jobs.gtlink import Gtlink
except ImportError:
    pass

# Skip tests in this file if Fermi ST aren't available
pytestmark = requires_dependency('Fermi ST')

def test_gtlink():
    options=dict(irfs='CALDB', expcube=None,
                 bexpmap=None, cmap=None,
                 srcmdl=None, outfile=None)
    kwargs = dict(options=options,
                  flags=['gzip'],
                  input_file_args=['expcube', 'cmap', 'bexpmap', 'srcmdl'],
                  output_file_args=['outfile'])
    gtlink = Gtlink('gtsrcmaps', **kwargs)
