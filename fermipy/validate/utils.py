# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function


import os
import copy
import re
import yaml
import sys
import mimetypes
import tempfile
import string
import random
from os.path import splitext, basename
import xml.etree.cElementTree as ElementTree
import argparse
import numpy as np


def rand_str(size=7):
    chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for x in range(size))


def replace_aliases(cut_dict, aliases):
    """Substitute aliases in a cut dictionary."""
    for k, v in cut_dict.items():
        for k0, v0 in aliases.items():
            cut_dict[k] = cut_dict[k].replace(k0, '(%s)' % v0)


def strip(input_str):
    """Strip newlines and whitespace from a string."""
    return str(input_str.replace('\n', '').replace(' ', ''))


def get_files(files, extnames=['.root']):
    """Extract a list of file paths from a list containing both paths
    and file lists with one path per line."""

    files_out = []
    for f in files:

        mime = mimetypes.guess_type(f)
        if os.path.splitext(f)[1] in extnames:
            files_out += [f]
        elif mime[0] == 'text/plain':
            files_out += list(np.loadtxt(f, unpack=True, dtype='str'))
        else:
            raise Exception('Unrecognized input type.')

    return files_out


def load_chain(chain, files, nfiles=None):

    if isinstance(nfiles, list) and len(nfiles) == 1:
        files = files[:nfiles[0]]
    elif isinstance(nfiles, list) and len(nfiles) >= 2:
        files = files[nfiles[0]:nfiles[1]]
    elif nfiles is not None:
        files = files[:nfiles]

    print("Loading %i files..." % len(files))
    for f in files:
        chain.Add(f)
    return chain


def load_aliases(alias_files):

    aliases = {}
    for f in alias_files:
        if f.endswith('.xml'):
            aliases.update(get_cuts_from_xml(f))
        elif f.endswith('.yaml'):
            aliases.update(yaml.load(open(f, 'r')))
        else:
            raise Exception('Invalid file type for aliases option.')

    return aliases


def get_cuts_from_xml(xmlfile):
    """Extract event selection strings from the XML file."""

    root = ElementTree.ElementTree(file=xmlfile).getroot()
    event_maps = root.findall('EventMap')
    alias_maps = root.findall('AliasDict')[0]

    event_classes = {}
    event_types = {}
    event_aliases = {}

    for m in event_maps:
        if m.attrib['altName'] == 'EVENT_CLASS':
            for c in m.findall('EventCategory'):
                event_classes[c.attrib['name']] = strip(
                    c.find('ShortCut').text)
        elif m.attrib['altName'] == 'EVENT_TYPE':
            for c in m.findall('EventCategory'):
                event_types[c.attrib['name']] = strip(c.find('ShortCut').text)

    for m in alias_maps.findall('Alias'):
        event_aliases[m.attrib['name']] = strip(m.text)

    replace_aliases(event_aliases, event_aliases.copy())
    replace_aliases(event_aliases, event_aliases.copy())
    replace_aliases(event_classes, event_aliases)
    replace_aliases(event_types, event_aliases)

    event_selections = {}
    event_selections.update(event_classes)
    event_selections.update(event_types)
    event_selections.update(event_aliases)

    return event_selections


def set_event_list(tree, selection=None, fraction=None, start_fraction=None):
    """
    Set the event list for a tree or chain.

    Parameters
    ----------    
    tree : `ROOT.TTree`
        Input tree/chain.
    selection : str
        Cut string defining the event list.
    fraction : float
        Fraction of the total file to include in the event list
        starting from the *end* of the file.

    """
    import ROOT

    elist = rand_str()

    if selection is None:
        cuts = ''
    else:
        cuts = selection

    if fraction is None or fraction >= 1.0:
        n = tree.Draw(">>%s" % elist, cuts, "goff")
        tree.SetEventList(ROOT.gDirectory.Get(elist))
    elif start_fraction is None:
        nentries = int(tree.GetEntries())
        first_entry = min(int((1.0 - fraction) * nentries), nentries)
        n = tree.Draw(">>%s" % elist, cuts, "goff", nentries, first_entry)
        tree.SetEventList(ROOT.gDirectory.Get(elist))
    else:
        nentries = int(tree.GetEntries())
        first_entry = min(int(start_fraction * nentries), nentries)
        n = first_entry + int(nentries * fraction)
        n = tree.Draw(">>%s" % elist, cuts, "goff",
                      n - first_entry, first_entry)
        tree.SetEventList(ROOT.gDirectory.Get(elist))

    return n
