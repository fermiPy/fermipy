# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import os
import argparse
import yaml
import numpy as np
from astropy.table import Table, Column, vstack
from fermipy.roi_model import ROIModel


def read_sources_from_numpy_file(npfile):
    """ Open a numpy pickle file and read all the new sources into a dictionary

    Parameters
    ----------
    npfile : file name
       The input numpy pickle file

    Returns
    -------
    tab : `~astropy.table.Table`
    """
    srcs = np.load(npfile).flat[0]['sources']

    roi = ROIModel()
    roi.load_sources(srcs.values())
    return roi.create_table()


def read_sources_from_yaml_file(yamlfile):
    """ Open a yaml file and read all the new sources into a dictionary

    Parameters
    ----------
    yaml : file name
       The input yaml file

    Returns
    -------
    tab : `~astropy.table.Table`
    """
    f = open(yamlfile)
    dd = yaml.load(f)
    srcs = dd['sources']
    f.close()
    roi = ROIModel()
    roi.load_sources(srcs.values())
    return roi.create_table()


def merge_source_tables(src_tab, tab, all_sources=False, prefix="", suffix="",
                        roi_idx=None):
    """Append the sources in a table into another table.

    Parameters
    ----------
    src_tab : `~astropy.table.Table`    
       Master source table that will be appended with the sources in
       ``tab``.

    tab : `~astropy.table.Table`
       Table to be merged into ``src_tab``. 

    all_sources : bool
       If true, then all the sources get added to the table.  
       if false, then only the sources that start with 'PS' get added

    prefix : str
       Prepended to all source names

    suffix : str
       Appended to all source names

    Returns
    -------
    tab : `~astropy.table.Table`

    """
    if roi_idx is not None and 'roi' not in tab.columns:
        tab.add_column(Column(name='roi', data=len(tab) * [roi_idx]))

    remove_rows = []
    for i, row in enumerate(tab):

        if not all_sources and row['name'].find("PS") != 0:
            remove_rows += [i]
            continue
        sname = "%s%s%s" % (prefix, row['name'], suffix)
        row['name'] = sname

    tab.remove_rows(remove_rows)

    if src_tab is None:
        src_tab = tab
    else:
        src_tab = vstack([src_tab, tab], join_type='outer')

    return src_tab


def main():
    usage = "usage: %(prog)s [input]"
    description = "Collect multiple source files into a single FITS file."

    parser = argparse.ArgumentParser(usage=usage, description=description)
    parser.add_argument('--output', type=argparse.FileType('w'), help='Output FITS file.',
                        required=True)
    parser.add_argument('--clobber', action='store_true',
                        help='Overwrite output file.')
    parser.add_argument('--all_sources', action='store_true',
                        help='Include all sources.')
    parser.add_argument('input', nargs="*", help='Input npy or yaml files.')

    args = parser.parse_args()

    src_tab = None
    print('Collect sources')
    for idx, filepath in enumerate(args.input):

        print('Processing', filepath)
        path, ext = os.path.splitext(filepath)

        if ext == '.fits':
            tab = Table.read(filepath)
        elif ext == '.npy':
            tab = read_sources_from_numpy_file(filepath)
        elif ext == '.yaml':
            tab = read_sources_from_yaml_file(filepath)
        else:
            raise Exception("Can't read source from file type %s" % (ext))

        src_tab = merge_source_tables(src_tab, tab, prefix="roi_%05i_" % idx,
                                      all_sources=args.all_sources,
                                      roi_idx=idx)

    print('Collected', len(src_tab), 'sources')
    src_tab.write(args.output, format='fits', overwrite=args.clobber)


if __name__ == "__main__":
    main()
