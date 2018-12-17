# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utility functions for the `fermipy.jobs` module.
"""
from __future__ import absolute_import, division, print_function

def is_null(val):
    """Check if a value is null,
    This is needed b/c we are parsing
    command line arguements and 'None' and 'none'
    can be used.
    """
    return val in [None, 'none', 'None']


def is_not_null(val):
    """Check if a value is not null,
    This is needed b/c we are parsing
    command line arguements and 'None' and 'none'
    can be used.
    """
    return val not in [None, 'none', 'None']
