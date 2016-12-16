# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Classes and utilites to keep track of files associated to an analysis
"""
from __future__ import absolute_import, division, print_function

import os
import time

import numpy as np
from numpy.core import defchararray
from astropy.table import Table, Column

def get_timestamp():
    """Get the current time as an integer"""
    return int(time.time())

def get_unique_match(table, colname, value):
    """Get the row matching value for a particular column. 
    If exactly one row matchs, return indsx of that row.
    Otherwise raise KeyError
    """
    # FIXME, This is here for python 3.5, where astropy is now returning bytes instead of str
    if table[colname].dtype.kind in ['S', 'U']:
        mask = table[colname].astype(str) == value
    else:
        mask = table[colname] == value

    if mask.sum() != 1:
        raise KeyError("%i rows in column %s match value %s"%(mask.sum(), colname, value))
    return np.argmax(mask)


#@unique
#class JobStatus(Enum):
class FileStatus(object):
    """Enumeration of file status types"""
    no_file = 0       #File is not in system
    expected = 1      #File will be created by a scheduled job
    exists = 2        #File exists
    missing = 3       #File should exist, but does not
    superseded = 4    #File exists, but has been superseded
    temp_removed = 5  #File was temporaray and has been removed


class FileHandle(object): 
    """Class to keep track of a file"""
    def __init__(self, **kwargs):
        """C'tor """
        self.key = kwargs.get('key', -1)
        self.creator = kwargs.get('creator', -1)
        self.timestamp = kwargs.get('timestamp', 0)
        self.status = kwargs.get('status', FileStatus.no_file)
        self.path = kwargs['path']

    @staticmethod
    def make_table(file_dict):
        """Build an `astropy.table.Table' to store `FileHandle' objects"""
        col_key = Column(name='key', dtype=int)
        col_creator = Column(name='creator', dtype=int)
        col_timestamp = Column(name='timestamp', dtype=int)
        col_status = Column(name='status', dtype=int)
        col_path = Column(name='path', dtype='S256')
        columns = [col_key, col_creator,
                   col_timestamp, col_status, col_path]
        table = Table(data=columns)
        for val in file_dict.values():
            val._append_to_table(table)
        return table
    
    @staticmethod
    def make_dict(table):
        """Build a dict of file from an `astropy.table.Table'"""
        ret_dict = {}
        for row in table:
            file_handle = FileHandle._create_from_row(row)
        ret_dict[file_handle.key] = file_handle
        return ret_dict

    @staticmethod
    def _create_from_row(table_row):
        """Create a FileHandle object from an `astropy.table.row.Row` """
        kwargs = {}
        for key in table_row.colnames:
            if table_row[key].dtype.kind in ['S', 'U']:
                kwargs[key] = table_row[key].astype(str)
            else:            
                kwargs[key] = table_row[key]
        return FileHandle(**kwargs)


    def _append_to_table(self, table):
        """Add this instance as a row on a `astropy.Table' """
        table.add_row(dict(path=self.path,
                           key=self.key,
                           creator=self.creator,
                           timestamp=self.timestamp,
                           status=self.status))

    def _update_table_row(self, table_row):
        """Update the values in an `astropy.Table' for this instances"""
        table_row = dict(path=self.path,
                         key=self.key,
                         creator=self.creator,
                         timestamp=self.timestamp,
                         status=self.status)


class FileArchive(object):
    """Class that keeps track of the status of files used in an analysis"""
    def __init__(self, **kwargs):
        self._read_table_file(kwargs['file_archive_table'])
        self._base_path = kwargs['base_path']

    def _get_fullpath(self, filepath):
        """Prefix the base_path to the filepath if it isn't already there"""
        if filepath.find(self._base_path) != 0:
            return os.path.join(self._base_path, filepath)
        else:
            return filepath

    def _get_localpath(self, filepath):
        """Prefix the base_path to the filepath if it isn't already there"""
        return filepath.replace(self._base_path, '')

    def _read_table_file(self, table_file):
        """Read an `astropy.table.Table' to set up the archive"""
        self._table_file = table_file
        if os.path.exists(self._table_file):
            self._table = Table.read(self._table_file)
        else:
            self._table = FileHandle.make_table({})

    def _make_file_handle(self, row_idx):
        """Create a FileHandle object from an `astropy.table.row.Row` """
        row = self._table[row_idx]
        return FileHandle._create_from_row(row)

    def get_handle(self, filepath):
        """Get the FileHandle object associated to a particular file"""
        localpath = self._get_localpath(filepath)
        row_idx = get_unique_match(self._table, 'path', localpath)
        return self._make_file_handle(row_idx)

    def register_file(self, filepath, creator, status=FileStatus.no_file):
        """Register a file in the archive"""
        # check to see if the file already exists
        try: 
            file_handle = self.get_handle(filepath)
            raise KeyError("File %s already exists in archive"%filepath)
        except KeyError:
            pass
        localpath = self._get_localpath(filepath)        
        if status == FileStatus.exists:
            # Make sure the file really exists
            fullpath = self._get_fullpath(filepath)
            if not os.path.exists(fullpath):
                print ("File %s does not exist but register_file was called with FileStatus.exists"%fullpath)
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
                                 status=status)        
        file_handle._append_to_table(self._table)
        return file_handle
        
    
    def update_file(self, filepath, creator, status):
        """Update a file in the archive"""
        file_handle = self.get_handle(filepath)
        if status in [FileStatus.exists, FileStatus.superseded]:
            # Make sure the file really exists
            fullpath = file_handle.fullpath
            if not os.path.exists(fullpath):
                raise ValueError("File %s does not exist but register_file was called with FileStatus.exists"%fullpath)
            timestamp = int(os.stat(fullpath).st_mtime)
        else:
            timestamp = 0
        file_handle.creator = creator
        file_handle.timestamp = timestamp
        file_handle.status = status
        file_handle._update_table_row(self._table[file_handle.key - 1])
        return file_handle

    def get_file_ids(self, file_list, creator=None, status=FileStatus.no_file):
        """Get or create a list of file ids based on file names"""
        ret_list = []
        for fname in file_list:
            try:
                fhandle = self.get_handle(fname)
            except KeyError:
                if creator is None:
                    raise KeyError("Can not register a file %s without a creator"%fname)
                fhandle = self.register_file(fname, creator, status)
            ret_list.append(fhandle.key)
        return ret_list

    def get_file_paths(self, id_list):
        """Get a list of file paths based of a set of ids"""
        if id_list is None:
            return []
        path_array = self._table[id_list-1]['path']
        return [path for path in path_array]
    
  
  
