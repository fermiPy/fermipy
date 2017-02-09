# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import os
import re
from astropy.tests.helper import pytest
from fermipy import get_st_version

__all__ = ['requires_dependency', 'requires_st_version', 'requires_file']


def requires_file(filepath):

    skip_it = True if not os.path.isfile(filepath) else False
    return pytest.mark.skipif(skip_it,
                              reason='File %s does not exist.' % filepath)


def version_str_to_int(version_str):

    m = re.search('(\d\d)-(\d\d)-(\d\d)', version_str)
    if m is None:
        return 0
    else:
        return (int(m.group(1)) * 10000 +
                int(m.group(2)) * 100 + int(m.group(3)))


def requires_st_version(version_str):
    """Decorator to declare minimum ST version needed for tests.
    """

    version = version_str_to_int(version_str)
    st_version = version_str_to_int(get_st_version())

    if st_version >= version:
        skip_it = False
    else:
        skip_it = True

    reason = 'Requires ST Version >=: {}'.format(version_str)
    return pytest.mark.skipif(skip_it, reason=reason)


def requires_dependency(name):
    """Decorator to declare required dependencies for tests.

    Examples
    --------

    ::

        from fermipy.tests.utils import requires_dependency

        @requires_dependency('scipy')
        def test_using_scipy():
            import scipy
            ...

        @requires_dependency('Fermi ST')
        def test_using_fermi_science_tools():
            import pyLikelihood
            ...
    """
    if name == 'Fermi ST':
        name = 'pyLikelihood'

    try:
        __import__(name)
        skip_it = False
    except ImportError:
        skip_it = True

    reason = 'Missing dependency: {}'.format(name)
    return pytest.mark.skipif(skip_it, reason=reason)
