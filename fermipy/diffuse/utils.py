# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Prepare data for diffuse all-sky analysis
"""
from __future__ import absolute_import, division, print_function

import os
from fermipy.jobs.utils import is_null


def readlines(arg):
    """Read lines from a file into a list.

    Removes whitespace and lines that start with '#'
    """
    fin = open(arg)
    lines_in = fin.readlines()
    fin.close()
    lines_out = []
    for line in lines_in:
        line = line.strip()
        if not line or line[0] == '#':
            continue
        lines_out.append(line)
    return lines_out


def create_inputlist(arglist):
    """Read lines from a file and makes a list of file names.

    Removes whitespace and lines that start with '#'
    Recursively read all files with the extension '.lst'
    """
    lines = []
    if isinstance(arglist, list):
        for arg in arglist:
            if os.path.splitext(arg)[1] == '.lst':
                lines += readlines(arg)
            else:
                lines.append(arg)
    elif is_null(arglist):
        pass
    else:
        if os.path.splitext(arglist)[1] == '.lst':
            lines += readlines(arglist)
        else:
            lines.append(arglist)
    return lines
