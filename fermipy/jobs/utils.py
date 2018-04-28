# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Prepare data for diffuse all-sky analysis
"""
from __future__ import absolute_import, division, print_function

import os

def is_null(val):
    if val in [None, 'none', 'None']:
        return True
    else:
        return False

def is_not_null(val):
    if val in [None, 'none', 'None']:
        return False
    else:
        return True

