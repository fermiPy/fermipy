# -*- coding: utf-8 -*-
# Author: Douglas Creager <dcreager@dcreager.net>
# This file is placed into the public domain.

# Calculates the current version number.  If possible, this is the
# output of “git describe”, modified to conform to the versioning
# scheme that setuptools uses.  If “git describe” returns an error
# (most likely because we're in an unpacked copy of a release tarball,
# rather than in a git working copy), then we fall back on reading the
# contents of the RELEASE-VERSION file.
#
# To use this script, simply import it your setup.py file, and use the
# results of get_git_version() as your package version:
#
# from version import *
#
# setup(
#     version=get_git_version(),
#     .
#     .
#     .
# )
#
# This will automatically update the RELEASE-VERSION file, if
# necessary.  Note that the RELEASE-VERSION file should *not* be
# checked into git; please add it to your top-level .gitignore file.
#
# You'll probably want to distribute the RELEASE-VERSION file in your
# sdist tarballs; to do this, just create a MANIFEST.in file that
# contains the following line:
#
#   include RELEASE-VERSION

import os
import subprocess
from subprocess import check_output

__all__ = ("get_git_version")

_refname = '$Format: %D$'
_tree_hash = '$Format: %t$'
_commit_info = '$Format:%cd by %aN$'
_commit_hash = '$Format: %h$'


def capture_output(cmd, dirname):

    p = subprocess.Popen(cmd,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         cwd=dirname)
    p.stderr.close()

    output = p.stdout.readlines()

    if not output:
        return None
    else:
        return output[0].strip()


def render_pep440(vcs):
    """Convert git release tag into a form that is PEP440 compliant."""

    if vcs is None:
        return None

    tags = vcs.split('-')

    # Bare version number
    if len(tags) == 1:
        return tags[0]
    else:
        return tags[0] + '+' + '.'.join(tags[1:])


def call_git_describe(abbrev=4):

    dirname = os.path.abspath(os.path.dirname(__file__))

    try:
        has_git_tree = capture_output(['git', 'rev-parse',
                                       '--is-inside-work-tree'], dirname)
    except:
        return None

    if not has_git_tree:
        return None

    try:
        line = check_output(['git', 'describe', '--abbrev=%d' % abbrev,
                             '--dirty', '--tags'], cwd=dirname)

        return line.strip().decode('utf-8')

    except:
        return None


def read_release_keywords(keyword):

    refnames = keyword.strip()
    if refnames.startswith("$Format"):
        return None

    refs = set([r.strip() for r in refnames.strip("()").split(",")])
    TAG = "tag: "
    tags = set([r[len(TAG):] for r in refs if r.startswith(TAG)])
    if not tags:
        return None
    return sorted(tags)[-1]


def read_release_version():
    """Read the release version from ``_version.py``."""
    import re
    dirname = os.path.abspath(os.path.dirname(__file__))

    try:
        f = open(os.path.join(dirname, "_version.py"), "rt")
        for line in f.readlines():

            m = re.match("__version__ = '([^']+)'", line)
            if m:
                ver = m.group(1)
                return ver

    except:
        return None

    return None


def write_release_version(version):
    """Write the release version to ``_version.py``."""
    dirname = os.path.abspath(os.path.dirname(__file__))
    f = open(os.path.join(dirname, "_version.py"), "wt")
    f.write("__version__ = '%s'\n" % version)
    f.close()


def get_git_version(abbrev=4):

    # Read in the version that's currently in _version.py.
    release_version = read_release_version()

    # First try to get the current version using “git describe”.
    git_version = call_git_describe(abbrev)
    git_version = render_pep440(git_version)

    # Try to deduce the version from keyword expansion
    keyword_version = read_release_keywords(_refname)
    keyword_version = render_pep440(keyword_version)

    # If that doesn't work, fall back on the value that's in
    # _version.py.
    if git_version is not None:
        version = git_version
    elif release_version is not None:
        version = release_version
    elif keyword_version is not None:
        version = keyword_version
    else:
        version = 'unknown'

    # If we still don't have anything, that's an error.
    if version is None:
        raise ValueError("Cannot find the version number!")

    # If the current version is different from what's in the
    # _version.py file, update the file to be current.
    if version != release_version and version != 'unknown':
        write_release_version(version)

    # Finally, return the current version.
    return version


if __name__ == "__main__":
    print(get_git_version())
