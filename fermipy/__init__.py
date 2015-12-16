#!/usr/bin/env python

import os

__version__ = "unknown"

try:
    from version import get_git_version
    __version__ = get_git_version()
except Exception as message:
    print(message)

__author__ = "Matthew Wood"

PACKAGE_ROOT = os.path.abspath(os.path.dirname(__file__))
PACKAGE_DATA = os.path.join(PACKAGE_ROOT,'data')
os.environ['FERMIPY_ROOT'] = PACKAGE_ROOT
os.environ['FERMIPY_DATA_DIR'] = PACKAGE_DATA
