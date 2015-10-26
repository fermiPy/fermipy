#!/usr/bin/env python

import os

__version__ = "unknown"

try:
    from version import get_git_version
    __version__ = get_git_version()
except Exception, message:
    print message

__author__ = "Matthew Wood"

PACKAGE_ROOT = os.path.abspath(os.path.dirname(__file__))
