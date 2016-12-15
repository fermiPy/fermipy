# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Classes and utilites to keep track of files associated to an analysis
"""
from __future__ import absolute_import, division, print_function

import os
import time

#from enum import Enum

import numpy as np
from astropy.table import Table, Column

from fermipy.jobs.file_archive import FileStatus, FileArchive


def get_timestamp():
    """Get the current time as an integer"""
    return int(time.time())

def get_matches(table, colname, value):
    """Get the rows matching a value for a particular column. 
    """
    if table[colname].dtype.kind in ['S', 'U']:
        matches = table[colname].astype(str) == value
    else:
        matches = table[colname] == value
    return matches



#@unique
#class JobStatus(Enum):
class JobStatus(object):
    """Enumeration of job status types"""
    no_job = 0
    pending = 1
    running = 2
    done = 3
    failed = 4


class JobDetails(object):
    """ This is just a simple structure to keep track of the details of each
        of the sub-proccess jobs
    """

    def __init__(self, **kwargs):
        """ C'tor

        Keyword arguments:
        ---------------
        dbkey : int
            A unique key to identify this job
            
        jobname : str
            A name used to idenfity this job

        jobkey : str
            A string to identify this instance of the job

        appname : str
            The executable inovked to run the job
            
        logfile : str
            The logfile for this job, may be used to check for success/ failure

        job_config : dict
            A dictionrary with the arguments for the job

        parent_id : int
            Unique key identifying the parent job

        infile_ids : list of int
            Keys to identify input files to this job

        outfile_ids : list of int
            Keys to identify output files from this job
        
        rm_ids : list in int
            Keys to identify temporary files removed by this job

        status : int
            Current job status, one of the enums above
        """
        self.dbkey = kwargs.get('dbkey', -1)
        self.jobname = kwargs.get('jobname')
        self.jobkey = kwargs.get('jobkey')
        self.appname = kwargs.get('appname')
        self.logfile = kwargs.get('logfile')
        self.job_config = kwargs.get('job_config', {})
        if type(self.job_config) is str:
            self.job_config = eval(self.job_config)
        self.timestamp = kwargs.get('timestamp', 0)
        self.infiles = kwargs.get('infiles', None)
        self.outfiles = kwargs.get('outfiles', None)
        self.rmfiles = kwargs.get('rmfiles', None)
        self.infile_ids = kwargs.get('infile_ids', None)
        self.outfile_ids = kwargs.get('outfile_ids', None)
        self.rmfile_ids = kwargs.get('rmfile_ids', None)
        self.status = kwargs.get('status', 'Pending')

    @staticmethod
    def make_tables(job_dict):
        """Build an `astropy.table.Table' to store job details"""
        col_dbkey = Column(name='dbkey', dtype=int)
        col_jobname = Column(name='jobname', dtype='S64')
        col_jobkey = Column(name='jobkey', dtype='S64')
        col_appname = Column(name='appname', dtype='S64')
        col_logfile = Column(name='logfile', dtype='S256')
        col_job_config = Column(name='job_config', dtype='S1024')
        col_timestamp = Column(name='timestamp', dtype=int)
        col_infile_refs = Column(name='infile_refs', dtype=int, shape=(2))
        col_outfile_refs = Column(name='outfile_refs', dtype=int, shape=(2))        
        col_rmfile_refs = Column(name='rmfile_refs', dtype=int, shape=(2))        
        col_status = Column(name='status', dtype=int)
        columns = [col_dbkey, col_jobname, col_jobkey, col_appname,
                   col_logfile, col_job_config, col_timestamp,
                   col_infile_refs, col_outfile_refs, col_rmfile_refs,
                   col_status]

        table = Table(data=columns)

        col_file_ids = Column(name='file_id', dtype=int)
        table_ids = Table(data=[col_file_ids])

        for val in job_dict.values():
            val._append_to_tables(table, table_ids)
        return table, table_ids


    def get_file_ids(self, file_archive, creator=None, status=FileStatus.no_file):
        """Fill the file id arrays from the file lists"""
        if self.infile_ids is None:            
            if self.infiles is not None:
                self.infile_ids = np.zeros((len(self.infiles)), int)
                filelist = file_archive.get_file_ids(self.infiles, creator, FileStatus.exists)
                JobDetails._fill_array_from_list(filelist, self.infile_ids)
            else:
                self.infile_ids = np.zeros((0), int)
        if self.outfile_ids is None:            
            if self.outfiles is not None:
                self.outfile_ids = np.zeros((len(self.outfiles)), int)
                filelist = file_archive.get_file_ids(self.outfiles, creator, status)
                JobDetails._fill_array_from_list(filelist, self.outfile_ids)
            else:
                self.outfile_ids = np.zeros((0), int)
        if self.rmfile_ids is None:
            if self.rmfiles is not None:
                self.rmfile_ids = np.zeros((len(self.rmfiles)), int)
                filelist = file_archive.get_file_ids(self.rmfiles)
                JobDetails._fill_array_from_list(filelist, self.rmfile_ids)
            else:
                self.rmfile_ids = np.zeros((0), int)

    def get_file_paths(self, file_archive):
        """Fill the file name lists from the file id arrays"""
        if self.infiles is None:
            self.infiles = file_archive.get_file_paths(self.infile_ids)
        if self.outfiles is None:
            self.outfiles = file_archive.get_file_paths(self.outfile_ids)
        if self.rmfiles is None:
            self.rmfiles = file_archive.get_file_paths(self.rmfile_ids)

    @staticmethod
    def _fill_array_from_list(the_list, the_array):
        for i, val in enumerate(the_list):
            the_array[i] = val
        return the_array

    @staticmethod
    def _fill_list_from_array(the_array):
        return [ v for v in the_array.nonzero()[0] ]

    @staticmethod
    def make_dict(table, table_ids):
        """Build a dictionary of JobDetails from an `astropy.table.Table'"""
        ret_dict = {}
        table_id_array = table_ids['file_id'].data
        for row in table:
            job_details = JobDetails._create_from_row(row, table_id_array)
        ret_dict[job_details.dbkey] = job_details
        return ret_dict

    @staticmethod
    def _create_from_row(table_row, table_id_array):
        """Create a JobDetails object from an `astropy.table.row.Row` """
        kwargs = {}
        for key in table_row.colnames:
            if table_row[key].dtype.kind in ['S', 'U']:
                kwargs[key] = table_row[key].astype(str)
            else:            
                kwargs[key] = table_row[key]
           
        infile_refs = kwargs.pop('infile_refs')
        outfile_refs = kwargs.pop('outfile_refs')
        rmfile_refs = kwargs.pop('rmfile_refs')

        kwargs['infile_ids'] = np.arange(infile_refs[0], infile_refs[1])
        kwargs['outfile_ids'] = np.arange(outfile_refs[0], outfile_refs[1])
        kwargs['rmfile_ids'] = np.arange(rmfile_refs[0], rmfile_refs[1])
        return JobDetails(**kwargs)


    def _append_to_tables(self, table, table_ids):
        """Add this instance as a row on a `astropy.Table' """
        infile_refs = np.zeros((2), int)
        outfile_refs = np.zeros((2), int)
        rmfile_refs = np.zeros((2), int)
        f_ptr = len(table_ids['file_id'])
        infile_refs[0] = f_ptr
        if self.infile_ids is not None:
            for fid in self.infile_ids:
                table_ids.add_row(dict(file_id=fid))
                f_ptr += 1
        infile_refs[1] = f_ptr
        outfile_refs[0] = f_ptr
        if self.outfile_ids is not None:
            for fid in self.outfile_ids:
                table_ids.add_row(dict(file_id=fid))
                f_ptr += 1
        outfile_refs[1] = f_ptr
        rmfile_refs[0] = f_ptr
        if self.rmfile_ids is not None:
            for fid in self.rmfile_ids:
                table_ids.add_row(dict(file_id=fid))
                f_ptr += 1
        rmfile_refs[1] = f_ptr

        table.add_row(dict(dbkey=self.dbkey,
                           jobname=self.jobname,
                           jobkey=self.jobkey,
                           appname=self.appname,
                           logfile=self.logfile,
                           job_config=str(self.job_config),
                           timestamp=self.timestamp,
                           infile_refs=infile_refs,
                           outfile_refs=outfile_refs,
                           rmfile_refs=rmfile_refs,
                           status=self.status))
                               
    def _update_table_row(self, table_row):
        """Add this instance as a row on a `astropy.Table' """
        table_row.update(dict(timestamp=self.timestamp,
                              status=self.status))


class JobArchive(object):
    """Class that keeps track of the status of files used in an analysis"""
    def __init__(self, **kwargs):
        self._table_file = None
        self._table = None
        self._table_ids = None
        self._table_id_array = None
        self._read_table_file(kwargs['job_archive_table'])
        self._file_archive = FileArchive(**kwargs)

    def _read_table_file(self, table_file):
        """Read an `astropy.table.Table' to set up the archive"""
        self._table_file = table_file
        if os.path.exists(self._table_file):
            self._table = Table.read(self._table_file, hdu='JOB_ARCHIVE')
            self._table_ids = Table.read(self._table_file, hdu='FILE_IDS')
        else:
            self._table, self._table_ids = JobDetails.make_tables({})
        self._table_id_array = self._table_ids['file_id'].data

    def _make_job_details(self, row_idx):
        """Create a JobDetails object from an `astropy.table.row.Row` """
        row = self._table[row_idx]
        job_details = JobDetails._create_from_row(row, self._table_id_array)
        job_details.get_file_paths(self._file_archive)

      
    def get_details(self, jobname, jobkey):
        """Get the JobDetails object associated to a particular job instance"""        
        match_job = get_matches(self._table, 'jobname', jobname)
        match_key = get_matches(self._table, 'jobkey', jobkey)
        mask = match_job * match_key
        if mask.sum() != 1:
            raise KeyError("%i rows match jobname=%s, jobkey=%s"%(mask.sum(), jobname, jobkey))
        row_idx= np.argmax(mask)
        return self._make_job_details(row_idx)

    def register_job(self, job_details):
        """Register a job in the archive"""
        # check to see if the job already exists
        try:
            job_details = self.get_details(job_details.jobname,
                                           job_details.jobkey)
            raise KeyError("File %s:%s already exists in archive"%(job_details.jobname,
                                                                   job_details.jobkey))
        except KeyError:
            pass        
        job_details.dbkey = len(self._table) + 1
        job_details.get_file_ids(self._file_archive, creator=job_details.dbkey)
        job_details._append_to_tables(self._table, self._table_ids)
        self._table_id_array = self._table_ids['file_id'].data
        return job_details
        
    
    def register_job_from_link(self, link, key, **kwargs):
        """Register a job in the archive from a `Link' object """
        creator_key = len(self._table) + 1
        job_config = kwargs.get('job_config', None)
        if job_config is None:
            job_config = link.args
        status = kwargs.get('status', JobStatus.no_job)
        job_details = JobDetails(jobname=link.linkname,
                                 jobkey=key,
                                 appname=link.appname,
                                 logfile=kwargs.get('logfile'),
                                 jobconfig=job_config,
                                 timestamp=get_timestamp(),
                                 input_files=link.input_files,
                                 output_files=link.output_files,
                                 status=status)
        self.register_job(job_details)
        return job_details

    def update_job(self, job_details):
        """Update a job in the archive"""
        other = self.get_details(job_details.jobname,
                                 job_details.jobkey)
        other.timestamp = job_details.timestamp
        other.status = job_details.status
        other._update_table_row(self._table[other.dbkey -1])
        return other


    @staticmethod
    def build_temp_job_archive():
        """ """
        try:
            os.unlink('job_archive_temp.fits')
            os.unlink('file_archive_temp.fits')
        except OSError:
            pass
        job_archive = JobArchive(job_archive_table='job_archive_temp.fits',
                                 file_archive_table='file_archive_temp.fits',
                                 base_path=os.path.abspath('.'))
        return job_archive
