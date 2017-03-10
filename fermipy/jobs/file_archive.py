# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Classes and utilites to keep track of files associated to an analysis.

The main class is `FileArchive`, which keep track of all the files associated to an analysis.

The `FileHandle` helper class encapsulates information on a particular file.
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import time
import tempfile

from collections import OrderedDict

import numpy as np
#from numpy.core import defchararray
from astropy.table import Table, Column

from fermipy.fits_utils import write_tables_to_fits


def get_timestamp():
    """Get the current time as an integer"""
    return int(time.time())


def get_unique_match(table, colname, value):
    """Get the row matching value for a particular column.
    If exactly one row matchs, return index of that row,
    Otherwise raise KeyError.
    """
    # FIXME, This is here for python 3.5, where astropy is now returning bytes
    # instead of str
    if table[colname].dtype.kind in ['S', 'U']:
        mask = table[colname].astype(str) == value
    else:
        mask = table[colname] == value

    if mask.sum() != 1:
        raise KeyError("%i rows in column %s match value %s" %
                       (mask.sum(), colname, value))
    return np.argmax(mask)


#@unique
# class FileStatus(Enum):
class FileStatus(object):
    """Enumeration of file status types"""
    no_file = 0  # File is not in system
    expected = 1  # File will be created by a scheduled job
    exists = 2  # File exists
    missing = 3  # File should exist, but does not
    superseded = 4  # File exists, but has been superseded
    temp_removed = 5  # File was temporary and has been removed


# class FileFlags(Enum):
class FileFlags(object):
    """Bit masks to indicate file types"""
    no_flags = 0  # No flags are set for this file
    input_mask = 1  # File is input to job
    output_mask = 2  # File is output from job
    rm_mask = 4  # File is removed by job
    gz_mask = 8  # File is compressed by job
    internal_mask = 16  # File is internal to job
    stageable = 32  # File can be staged to / from scratch area
    in_ch_mask = input_mask | output_mask | rm_mask | internal_mask
    out_ch_mask = output_mask | rm_mask | internal_mask
    in_stage_mask = input_mask | stageable
    out_stage_mask = output_mask | stageable
    rmint_mask = rm_mask | internal_mask


class FileDict(object):
    """Small class to keep track of files used & createed by a link.

    Parameters
    ----------

    file_args : dict
        Dictionary mapping argument [str] to `FileFlags` enum

    file_dict : dict
        Dictionary mapping file path [str] to `FileFlags` enum
    """

    def __init__(self, **kwargs):
        """C'tor"""
        self.file_args = kwargs.get('file_args', {})
        self.file_dict = {}

    def latch_file_info(self, args):
        """Extract the file paths from a set of arguments
        """
        self.file_dict.clear()
        for key, val in self.file_args.items():
            try:
                file_path = args[key]
                if file_path is None:
                    continue
                # 'args' is special
                if key[0:4] == 'args':
                    if isinstance(file_path, list):
                        tokens = file_path
                    elif isinstance(file_path, str):
                        tokens = file_path.split()
                    else:
                        raise TypeError(
                            "Args has type %s, expect list or str" % type(file_path))
                    for token in tokens:
                        self.file_dict[token.replace('.gz', '')] = val
                else:
                    self.file_dict[file_path.replace('.gz', '')] = val
            except KeyError:
                pass

    def update(self, file_dict):
        """Update self with values from a dictionary
        mapping file path [str] to `FileFlags` enum """
        for key, val in file_dict.items():
            if self.file_dict.has_key(key):
                self.file_dict[key] |= val
            else:
                self.file_dict[key] = val

    def items(self):
        """Return iterator over self.file_dict"""
        return self.file_dict.items()

    @property
    def input_files(self):
        """Return a list of the input files needed by this link.

        For `Link` sub-classes this will return the union
        of all the input files of each internal `Link`.

        That is to say this will include files produced by one
        `Link` in a `Chain` and used as input to another `Link` in the `Chain`
        """
        ret_list = []
        for key, val in self.file_dict.items():
            # For input files we only want files that were marked as input
            if val & FileFlags.input_mask:
                ret_list.append(key)
        return ret_list

    @property
    def output_files(self):
        """Return a list of the output files produced by this link.

        For `Link` sub-classes this will return the union
        of all the output files of each internal `Link`.

        That is to say this will include files produced by one
        `Link` in a `Chain` and used as input to another `Link` in the `Chain`
        """
        ret_list = []
        for key, val in self.file_dict.items():
            # For output files we only want files that were marked as output
            if val & FileFlags.output_mask:
                ret_list.append(key)
        return ret_list

    @property
    def chain_input_files(self):
        """Return a list of the input files needed by this chain.

        For `Link` sub-classes this will return only those files
        that were not created by any internal `Link`
        """
        ret_list = []
        for key, val in self.file_dict.items():
            # For chain input files we only want files that were not marked as output
            # (I.e., not produced by some other step in the chain)
            if val & FileFlags.in_ch_mask == FileFlags.input_mask:
                ret_list.append(key)
        return ret_list

    @property
    def chain_output_files(self):
        """Return a list of the all the output files produced by this link.

        For `Link` sub-classes this will return only those files
        that were not marked as internal files or marked for removal.
        """
        ret_list = []
        for key, val in self.file_dict.items():
            # For pure input files we only want output files that were not
            # marked as internal or temp
            if val & FileFlags.out_ch_mask == FileFlags.output_mask:
                ret_list.append(key)
        return ret_list

    @property
    def input_files_to_stage(self):
        """Return a list of the input files needed by this link.

        For `Link` sub-classes this will return the union
        of all the input files of each internal `Link`.

        That is to say this will include files produced by one
        `Link` in a `Chain` and used as input to another `Link` in the `Chain`
        """
        ret_list = []
        for key, val in self.file_dict.items():
            # For input files we only want files that were marked as input
            if val & FileFlags.in_stage_mask == FileFlags.in_stage_mask:
                ret_list.append(key)
        return ret_list

    @property
    def output_files_to_stage(self):
        """Return a list of the input files needed by this link.

        For `Link` sub-classes this will return the union
        of all the input files of each internal `Link`.

        That is to say this will include files produced by one
        `Link` in a `Chain` and used as input to another `Link` in the `Chain`
        """
        ret_list = []
        for key, val in self.file_dict.items():
            # For input files we only want files that were marked as input
            if val & FileFlags.out_stage_mask == FileFlags.out_stage_mask:
                ret_list.append(key)
        return ret_list

    @property
    def internal_files(self):
        """Return a list of the intermediate files produced by this link.

        This returns all files that were explicitly marked as internal files.
        """
        ret_list = []
        for key, val in self.file_dict.items():
            # For internal files we only want files that were marked as
            # internal
            if val & FileFlags.internal_mask:
                ret_list.append(key)
        return ret_list

    @property
    def temp_files(self):
        """Return a list of the temporary files produced by this link.

        This returns all files that were explicitly marked for removal.
        """
        ret_list = []
        for key, val in self.file_dict.items():
            # For temp files we only want files that were marked for removal
            if val & FileFlags.rm_mask:
                ret_list.append(key)
        return ret_list

    @property
    def gzip_files(self):
        """Return a list of the files compressed by this link.

        This returns all files that were explicitly marked for compression.
        """
        ret_list = []
        for key, val in self.file_dict.items():
            # For temp files we only want files that were marked for removal
            if val & FileFlags.gz_mask:
                ret_list.append(key)
        return ret_list

    def print_summary(self, stream=sys.stdout, indent=""):
        """Print a summary of the files in this file dict.

        This version explictly counts the union of all input and output files.
        """
        stream.write("%sTotal files      : %i\n" %
                     (indent, len(self.file_dict)))
        stream.write("%s  Input files    : %i\n" %
                     (indent, len(self.input_files)))
        stream.write("%s  Output files   : %i\n" %
                     (indent, len(self.output_files)))
        stream.write("%s  Internal files : %i\n" %
                     (indent, len(self.internal_files)))
        stream.write("%s  Temp files     : %i\n" %
                     (indent, len(self.temp_files)))

    def print_chain_summary(self, stream=sys.stdout, indent=""):
        """Print a summary of the files in this file dict.

        This version uses chain_input_files and chain_output_files to
        count the input and output files.
        """
        stream.write("%sTotal files      : %i\n" %
                     (indent, len(self.file_dict)))
        stream.write("%s  Input files    : %i\n" %
                     (indent, len(self.chain_input_files)))
        stream.write("%s  Output files   : %i\n" %
                     (indent, len(self.chain_output_files)))
        stream.write("%s  Internal files : %i\n" %
                     (indent, len(self.internal_files)))
        stream.write("%s  Temp files     : %i\n" %
                     (indent, len(self.temp_files)))


class FileStageManager(object):
    """Small class to deal with staging files to and from a scratch area """

    def __init__(self, scratchdir, workdir):
        """C'tor """
        try:
            self.scratchdir = tempfile.mkdtemp(prefix=os.environ['USER'] + '.',
                                               dir=scratchdir)
        except OSError:
            self.scratchdir = os.path.join(
                scratchdir, os.environ['USER'], 'dummy')
        self.workdir = os.path.abspath(workdir)

    def split_local_path(self, local_file):
        """Split the local path into a directory name and a file name

        If local_file is in self.workdir or a subdirectory of it,
        the directory will consist of the relative path from workdir.

        If local_file is not in self.workdir, directory will be empty.

        Returns (dirname, basename)
        """
        abspath = os.path.abspath(local_file)
        if abspath.find(self.workdir) >= 0:
            relpath = abspath.replace(self.workdir, '')[1:]
            basename = os.path.basename(relpath)
            dirname = os.path.dirname(relpath)
        else:
            basename = os.path.basename(local_file)
            dirname = ''
        return (dirname, basename)

    def construct_scratch_path(self, dirname, basename):
        """Construct and return a path in the scratch area.

        This will be  <self.scratchdir>/<dirname>/<basename>
        """
        return os.path.join(self.scratchdir, dirname, basename)

    def get_scratch_path(self, local_file):
        """Construct and return a path in the scratch area from a local file.
        """
        (local_dirname, local_basename) = self.split_local_path(local_file)
        return self.construct_scratch_path(local_dirname, local_basename)

    def map_files(self, local_files):
        """Build a dictionary mapping local paths to scratch paths.

        Parameters
        ----------

        local_files : list
            List of filenames to be mapped to scratch area

        Returns dict
            Mapping local_file : fullpath of scratch file
        """
        ret_dict = {}
        for local_file in local_files:
            ret_dict[local_file] = self.get_scratch_path(local_file)
        return ret_dict

    @staticmethod
    def make_scratch_dirs(file_mapping, dry_run=True):
        """Make any directories need in the scratch area"""
        scratch_dirs = {}
        for value in file_mapping.values():
            scratch_dirname = os.path.dirname(value)
            scratch_dirs[scratch_dirname] = True
        for scratch_dirname in scratch_dirs.keys():
            if dry_run:
                print ("mkdir -f %s" % (scratch_dirname))
            else:
                try:
                    os.makedirs(scratch_dirname)
                except OSError:
                    pass

    @staticmethod
    def copy_to_scratch(file_mapping, dry_run=True):
        """Copy input files to scratch area """
        for key, value in file_mapping.items():
            if not os.path.exists(key):
                continue
            if dry_run:
                print ("cp %s %s" % (key, value))
            else:
                os.system("cp %s %s" % (key, value))
        return file_mapping

    @staticmethod
    def copy_from_scratch(file_mapping, dry_run=True):
        """Copy output files from scratch area """
        for key, value in file_mapping.items():
            if dry_run:
                print ("cp %s %s" % (value, key))
            else:
                try:
                    outdir = os.path.dirname(key)
                    os.makedirs(outdir)
                except OSError:
                    pass
                os.system("cp %s %s" % (value, key))
        return file_mapping


class FileHandle(object):
    """Class to keep track of infomration about a file file.

    Parameters
    ----------

    key : int
        Unique id for this particular file

    creator : int
        Unique id for the job that created this file

    timestamp : int
        File creation time cast as an int

    status : `FileStatus`
        Enum giving current status of file

    flags : `FileFlags`
        Mask giving flags set on this file

    path : str
        Path to file
    """

    def __init__(self, **kwargs):
        """C'tor

        Take values of class members from keyword arguments.
        """
        self.key = kwargs.get('key', -1)
        self.creator = kwargs.get('creator', -1)
        self.timestamp = kwargs.get('timestamp', 0)
        self.status = kwargs.get('status', FileStatus.no_file)
        self.flags = kwargs.get('flags', FileFlags.no_flags)
        self.path = kwargs['path']

    @staticmethod
    def make_table(file_dict):
        """Build and return an `astropy.table.Table` to store `FileHandle`"""
        col_key = Column(name='key', dtype=int)
        col_path = Column(name='path', dtype='S256')
        col_creator = Column(name='creator', dtype=int)
        col_timestamp = Column(name='timestamp', dtype=int)
        col_status = Column(name='status', dtype=int)
        col_flags = Column(name='flags', dtype=int)
        columns = [col_key, col_path, col_creator,
                   col_timestamp, col_status, col_flags]
        table = Table(data=columns)
        for val in file_dict.values():
            val.append_to_table(table)
        return table

    @staticmethod
    def make_dict(table):
        """Build and return a dict of `FileHandle` from an `astropy.table.Table`

        The dictionary is keyed by FileHandle.key, which is a unique integer for each file
        """
        ret_dict = {}
        for row in table:
            file_handle = FileHandle.create_from_row(row)
        ret_dict[file_handle.key] = file_handle
        return ret_dict

    @staticmethod
    def create_from_row(table_row):
        """Build and return a `FileHandle` from an `astropy.table.row.Row` """
        kwargs = {}
        for key in table_row.colnames:
            if table_row[key].dtype.kind in ['S', 'U']:
                kwargs[key] = table_row[key].astype(str)
            else:
                kwargs[key] = table_row[key]
        try:
            return FileHandle(**kwargs)
        except KeyError:
            print (kwargs)

    def check_status(self, basepath=None):
        """Check on the status of this particular file"""
        if basepath is None:
            fullpath = self.path
        else:
            fullpath = os.path.join(basepath, self.path)
        exists = os.path.exists(fullpath)
        if not exists:
            if self.flags & FileFlags.gz_mask != 0:
                fullpath += '.gz'
                exists = os.path.exists(fullpath)
        if exists:
            if self.status == FileStatus.superseded:
                pass
            else:
                self.status = FileStatus.exists
        else:
            if self.status in [FileStatus.no_file,
                               FileStatus.expected,
                               FileStatus.missing,
                               FileStatus.temp_removed]:
                if self.flags & FileFlags.rmint_mask != 0:
                    self.status = FileStatus.temp_removed
            elif self.status == FileStatus.exists:
                self.status = FileStatus.missing
            elif self.status == FileStatus.exists:
                self.status = FileStatus.temp_removed
        return self.status

    def append_to_table(self, table):
        """Add this instance as a row on a `astropy.table.Table` """
        table.add_row(dict(path=self.path,
                           key=self.key,
                           creator=self.creator,
                           timestamp=self.timestamp,
                           status=self.status,
                           flags=self.flags))

    def update_table_row(self, table, row_idx):
        """Update the values in an `astropy.table.Table` for this instances"""
        table[row_idx]['path'] = self.path
        table[row_idx]['key'] = self.key
        table[row_idx]['creator'] = self.creator
        table[row_idx]['timestamp'] = self.timestamp
        table[row_idx]['status'] = self.status
        table[row_idx]['flags'] = self.flags


class FileArchive(object):
    """Class that keeps track of the status of files used in an analysis

    Parameters
    ----------

    table_file   : str
        Path to the file used to persist this `FileArchive`
    table        : `astropy.table.Table`
        Persistent representation of this `FileArchive`
    cache        : `OrderedDict`
        Transient representation of this `FileArchive`
    base_path    : str
        Base file path for all files in this `FileArchive`
    """
    # Singleton instance
    _archive = None

    def __init__(self, **kwargs):
        """C'tor

        Takes self.base_path from kwargs['base_path']
        Reads kwargs['file_archive_table']
        """
        self._table_file = None
        self._table = None
        self._cache = OrderedDict()
        self._base_path = kwargs['base_path']
        self._read_table_file(kwargs['file_archive_table'])

    def __getitem__(self, key):
        """ Return the `FileHandle` whose linkname is key"""
        return self._cache[key]

    @property
    def table_file(self):
        """Return the path to the file used to persist this `FileArchive` """
        return self._table_file

    @property
    def table(self):
        """Return the persistent representation of this `FileArchive` """
        return self._table

    @property
    def cache(self):
        """Return the transiet representation of this `FileArchive` """
        return self._cache

    @property
    def base_path(self):
        """Return the base file path for all files in this `FileArchive` """
        return self._base_path

    def _get_fullpath(self, filepath):
        """Return filepath with the base_path prefixed """
        if filepath[0] == '/':
            return filepath
        else:
            return os.path.join(self._base_path, filepath)

    def _get_localpath(self, filepath):
        """Return the filepath with the base_path removed """
        return filepath.replace(self._base_path, '')

    def _fill_cache(self):
        """Fill the cache from the `astropy.table.Table`"""
        for irow in range(len(self._table)):
            file_handle = self._make_file_handle(irow)
            self._cache[file_handle.path] = file_handle

    def _read_table_file(self, table_file):
        """Read an `astropy.table.Table` to set up the archive"""
        self._table_file = table_file
        if os.path.exists(self._table_file):
            self._table = Table.read(self._table_file)
        else:
            self._table = FileHandle.make_table({})
        self._fill_cache()

    def _make_file_handle(self, row_idx):
        """Build and return a `FileHandle` object from an `astropy.table.row.Row` """
        row = self._table[row_idx]
        return FileHandle.create_from_row(row)

    def get_handle(self, filepath):
        """Get the `FileHandle` object associated to a particular file """
        localpath = self._get_localpath(filepath)
        return self._cache[localpath]

    def register_file(self, filepath, creator, status=FileStatus.no_file, flags=FileFlags.no_flags):
        """Register a file in the archive.

        If the file already exists, this raises a `KeyError`

        Parameters
        ----------

        filepath : str
            The path to the file
        creatror : int
            A unique key for the job that created this file
        status   : `FileStatus`
            Enumeration giving current status of file
        flags   : `FileFlags`
            Enumeration giving flags set on this file

        Returns `FileHandle`
        """
        # check to see if the file already exists
        try:
            file_handle = self.get_handle(filepath)
            raise KeyError("File %s already exists in archive" % filepath)
        except KeyError:
            pass
        localpath = self._get_localpath(filepath)
        if status == FileStatus.exists:
            # Make sure the file really exists
            fullpath = self._get_fullpath(filepath)
            if not os.path.exists(fullpath):
                print ("register_file called on called on mising file %s" % fullpath)
                status = FileStatus.missing
                timestamp = 0
            else:
                timestamp = int(os.stat(fullpath).st_mtime)
        else:
            timestamp = 0
        key = len(self._table) + 1
        file_handle = FileHandle(path=localpath,
                                 key=key,
                                 creator=creator,
                                 timestamp=timestamp,
                                 status=status,
                                 flags=flags)
        file_handle.append_to_table(self._table)
        self._cache[localpath] = file_handle
        return file_handle

    def update_file(self, filepath, creator, status):
        """Update a file in the archive

        If the file does not exists, this raises a `KeyError`

        Parameters
        ----------

        filepath : str
            The path to the file
        creatror : int
            A unique key for the job that created this file
        status   : `FileStatus`
            Enumeration giving current status of file

        Returns `FileHandle`
        """
        file_handle = self.get_handle(filepath)
        if status in [FileStatus.exists, FileStatus.superseded]:
            # Make sure the file really exists
            fullpath = file_handle.fullpath
            if not os.path.exists(fullpath):
                raise ValueError("File %s does not exist" % fullpath)
            timestamp = int(os.stat(fullpath).st_mtime)
        else:
            timestamp = 0
        file_handle.creator = creator
        file_handle.timestamp = timestamp
        file_handle.status = status
        file_handle.update_table_row(self._table, file_handle.key - 1)
        return file_handle

    def get_file_ids(self, file_list, creator=None,
                     status=FileStatus.no_file, file_dict=None):
        """Get or create a list of file ids based on file names

        Parameters
        ----------

        file_list : list
            The paths to the file
        creatror : int
            A unique key for the job that created these files
        status   : `FileStatus`
            Enumeration giving current status of files
        file_dict : `FileDict`
            Mask giving flags set on this file

        Returns list of integers
        """
        ret_list = []
        for fname in file_list:
            if file_dict is None:
                flags = FileFlags.no_flags
            else:
                flags = file_dict.file_dict[fname]
            try:
                fhandle = self.get_handle(fname)
            except KeyError:
                if creator is None:
                    creator = -1
                    #raise KeyError("Can not register a file %s without a creator"%fname)
                fhandle = self.register_file(fname, creator, status, flags)
            ret_list.append(fhandle.key)
        return ret_list

    def get_file_paths(self, id_list):
        """Get a list of file paths based of a set of ids

        Parameters
        ----------

        id_list : list
            List of integer file keys

        Returns list of file paths
        """
        if id_list is None:
            return []
        try:
            path_array = self._table[id_list - 1]['path']
        except IndexError:
            print ("IndexError ", len(self._table), id_list)
            path_array = []
        return [path for path in path_array]

    def write_table_file(self, table_file=None):
        """Write the table to self._table_file"""
        if self._table is None:
            raise RuntimeError("No table to write")
        if table_file is not None:
            self._table_file = table_file
        if self._table_file is None:
            raise RuntimeError("No output file specified for table")           
        write_tables_to_fits(self._table_file, [self._table], clobber=True,
                             namelist=['FILE_ARCHIVE'])


    def update_file_status(self):
        """Update the status of all the files in the archive"""
        nfiles = len(self.cache.keys())
        status_vect = np.zeros((6), int)
        print ("Updating status of %i files: "%nfiles)
        for i, key in enumerate(self.cache.keys()):
            if i % 200 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
            fhandle = self.cache[key]
            fhandle.check_status(self._base_path)
            fhandle.update_table_row(self._table, fhandle.key - 1)
            status_vect[fhandle.status] += 1
            
        sys.stdout.write("!\n")
        sys.stdout.flush()
        sys.stdout.write("Summary:\n")
        sys.stdout.write("  no_file:      %i\n"%status_vect[0])
        sys.stdout.write("  expected:     %i\n"%status_vect[1])
        sys.stdout.write("  exists:       %i\n"%status_vect[2])
        sys.stdout.write("  missing:      %i\n"%status_vect[3])
        sys.stdout.write("  superseded:   %i\n"%status_vect[4])
        sys.stdout.write("  temp_removed: %i\n"%status_vect[5])

    @staticmethod
    def get_archive():
        """Return the singleton `FileArchive` instance """
        return FileArchive._archive

    @staticmethod
    def build_archive(**kwargs):
        """Return the singleton `FileArchive` instance, building it if needed"""
        if FileArchive._archive is None:
            FileArchive._archive = FileArchive(**kwargs)
        return FileArchive._archive


def main_browse():
    """Entry point for command line use for browsing a FileArchive """
    parser = argparse.ArgumentParser(usage="file_archive.py [options]",
                                     description="Browse a job archive")

    parser.add_argument('--files', action='store', dest='file_archive_table',
                        type=str, default='file_archive_temp.fits', help="File archive file")
    parser.add_argument('--base', action='store', dest='base_path',
                        type=str, default=os.path.abspath('.'), help="File archive base path")
    
    args = parser.parse_args(sys.argv[1:])    
    FileArchive.build_archive(**args.__dict__)


if __name__ == '__main__':
    main_browse()
