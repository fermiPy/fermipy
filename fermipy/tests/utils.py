# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
from astropy.tests.helper import pytest

__all__ = ['requires_dependency']


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
