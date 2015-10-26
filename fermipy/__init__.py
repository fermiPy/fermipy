#!/usr/bin/env python

import os

#__version__ = "0.3.1"
#from _version import __version__

try:
    import _version
    __version__ = _version.__version__
except:
    from version import get_git_version
    __version__ = get_git_version()

__author__ = "Matthew Wood"

PACKAGE_ROOT = os.path.abspath(os.path.dirname(__file__))
