# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Classes and utilites to keep track the various jobs that comprise an analysis pipeline.

The main class is `JobArchive`, which keep track of all the jobs associated to an analysis.

The `JobDetails` helper class encapsulates information on a instance of running a job.
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import argparse

import copy
from collections import OrderedDict

#from enum import Enum

import numpy as np
from astropy.table import Table, Column

from fermipy.fits_utils import write_tables_to_fits
from fermipy.jobs.file_archive import get_timestamp, FileStatus, FileDict, FileArchive


def get_matches(table, colname, value):
    """Get the rows matching a value for a particular column.
    """
    if table[colname].dtype.kind in ['S', 'U']:
        matches = table[colname].astype(str) == value
    else:
        matches = table[colname] == value
    return matches


#@unique
# class JobStatus(Enum):
class JobStatus(object):
    """Enumeration of job status types"""
    no_job = 0
    pending = 1
    running = 2
    done = 3
    failed = 4


class JobDetails(object):
    """A simple structure to keep track of the details of each
    of the sub-proccess jobs.

    Parameters
    ----------

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

    rmfile_ids : list of int
        Keys to identify temporary files removed by this job

    intfile_ids : list of int
        Keys to identify internal files

    status : int
        Current job status, one of the enums above
    """

    def __init__(self, **kwargs):
        """ C'tor

        Take values of class members from keyword arguments.
        """
        self.dbkey = kwargs.get('dbkey', -1)
        self.jobname = kwargs.get('jobname')
        self.jobkey = kwargs.get('jobkey')
        self.appname = kwargs.get('appname')
        self.logfile = kwargs.get('logfile')
        self.job_config = kwargs.get('job_config', {})
        if isinstance(self.job_config, str):
            try:
                self.job_config = eval(self.job_config)
            except SyntaxError:
                self.job_config = {}
        self.timestamp = kwargs.get('timestamp', 0)
        self.file_dict = kwargs.get('file_dict', None)
        self.sub_file_dict = kwargs.get('sub_file_dict', None)
        self.infile_ids = kwargs.get('infile_ids', None)
        self.outfile_ids = kwargs.get('outfile_ids', None)
        self.rmfile_ids = kwargs.get('rmfile_ids', None)
        self.intfile_ids = kwargs.get('intfile_ids', None)
        self.status = kwargs.get('status', JobStatus.pending)

    @staticmethod
    def make_fullkey(jobkey, jobname):
        """Combine jobname and jobkey to make a unique key
        fullkey = <jobkey>@<jobname>
        """
        return "%s@%s"%(jobkey, jobname)

    @staticmethod
    def split_fullkey(fullkey):
        """Split fullkey to make extract jobname, jobkey
        fullkey = <jobkey>@<jobname>
        """
        return fullkey.split('@')
        
    @staticmethod
    def make_tables(job_dict):
        """Build and return an `astropy.table.Table' to store `JobDetails`"""
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
        col_intfile_refs = Column(name='intfile_refs', dtype=int, shape=(2))
        col_status = Column(name='status', dtype=int)
        columns = [col_dbkey, col_jobname, col_jobkey, col_appname,
                   col_logfile, col_job_config, col_timestamp,
                   col_infile_refs, col_outfile_refs,
                   col_rmfile_refs, col_intfile_refs,
                   col_status]

        table = Table(data=columns)

        col_file_ids = Column(name='file_id', dtype=int)
        table_ids = Table(data=[col_file_ids])

        for val in job_dict.values():
            val.append_to_tables(table, table_ids)
        return table, table_ids

    @property
    def fullkey(self):
        """Return the fullkey for this job
        fullkey = <jobkey>@<jobname>
        """
        return JobDetails.make_fullkey(self.jobkey, self.jobname)

    def get_file_ids(self, file_archive, creator=None, status=FileStatus.no_file):
        """Fill the file id arrays from the file lists

        Parameters
        ----------

        file_archive : `FileArchive`
            Used to look up file ids

        creator : int
            A unique key for the job that created these file

        status   : `FileStatus`
            Enumeration giving current status thse files
        """
        file_dict = copy.deepcopy(self.file_dict)
        if self.sub_file_dict is not None:
            file_dict.update(self.sub_file_dict.file_dict)

        infiles = file_dict.input_files
        outfiles = file_dict.output_files
        rmfiles = file_dict.temp_files
        int_files = file_dict.internal_files

        if self.infile_ids is None:
            if infiles is not None:
                self.infile_ids = np.zeros((len(infiles)), int)
                filelist = file_archive.get_file_ids(
                    infiles, creator, FileStatus.expected, file_dict)
                JobDetails._fill_array_from_list(filelist, self.infile_ids)
            else:
                self.infile_ids = np.zeros((0), int)
        if self.outfile_ids is None:
            if outfiles is not None:
                self.outfile_ids = np.zeros((len(outfiles)), int)
                filelist = file_archive.get_file_ids(
                    outfiles, creator, status, file_dict)
                JobDetails._fill_array_from_list(filelist, self.outfile_ids)
            else:
                self.outfile_ids = np.zeros((0), int)
        if self.rmfile_ids is None:
            if rmfiles is not None:
                self.rmfile_ids = np.zeros((len(rmfiles)), int)
                filelist = file_archive.get_file_ids(rmfiles)
                JobDetails._fill_array_from_list(filelist, self.rmfile_ids)
            else:
                self.rmfile_ids = np.zeros((0), int)
        if self.intfile_ids is None:
            if int_files is not None:
                self.intfile_ids = np.zeros((len(int_files)), int)
                filelist = file_archive.get_file_ids(
                    int_files, creator, status)
                JobDetails._fill_array_from_list(filelist, self.intfile_ids)
            else:
                self.intfile_ids = np.zeros((0), int)

    def get_file_paths(self, file_archive, file_id_array):
        """Get the full paths of the files used by this object from the the id arrays  """
        full_list = []
        status_dict = {}
        full_list += file_archive.get_file_paths(file_id_array[self.infile_ids])
        full_list += file_archive.get_file_paths(file_id_array[self.outfile_ids])
        full_list += file_archive.get_file_paths(file_id_array[self.rmfile_ids])
        full_list += file_archive.get_file_paths(file_id_array[self.intfile_ids])
        for filepath in full_list:
            handle = file_archive.get_handle(filepath)
            status_dict[filepath] = handle.status
        if self.file_dict is None:
            self.file_dict = FileDict()
        self.file_dict.update(status_dict)

    @staticmethod
    def _fill_array_from_list(the_list, the_array):
        """Fill an `array` from a `list`"""
        for i, val in enumerate(the_list):
            the_array[i] = val
        return the_array

    @staticmethod
    def _fill_list_from_array(the_array):
        """Fill a `list` from the nonzero members of an `array`"""
        return [v for v in the_array.nonzero()[0]]

    @staticmethod
    def make_dict(table, table_ids):
        """Build a dictionary map int to `JobDetails` from an `astropy.table.Table`"""
        ret_dict = {}
        table_id_array = table_ids['file_id'].data
        for row in table:
            job_details = JobDetails.create_from_row(row, table_id_array)
        ret_dict[job_details.dbkey] = job_details
        return ret_dict

    @staticmethod
    def create_from_row(table_row, table_id_array):
        """Create a `JobDetails` from an `astropy.table.row.Row` """
        kwargs = {}
        for key in table_row.colnames:
            if table_row[key].dtype.kind in ['S', 'U']:
                kwargs[key] = table_row[key].astype(str)
            else:
                kwargs[key] = table_row[key]

        infile_refs = kwargs.pop('infile_refs')
        outfile_refs = kwargs.pop('outfile_refs')
        rmfile_refs = kwargs.pop('rmfile_refs')
        intfile_refs = kwargs.pop('intfile_refs')

        kwargs['infile_ids'] = np.arange(infile_refs[0], infile_refs[1])
        kwargs['outfile_ids'] = np.arange(outfile_refs[0], outfile_refs[1])
        kwargs['rmfile_ids'] = np.arange(rmfile_refs[0], rmfile_refs[1])
        kwargs['intfile_ids'] = np.arange(intfile_refs[0], intfile_refs[1])
        return JobDetails(**kwargs)

    def append_to_tables(self, table, table_ids):
        """Add this instance as a row on a `astropy.table.Table` """
        infile_refs = np.zeros((2), int)
        outfile_refs = np.zeros((2), int)
        rmfile_refs = np.zeros((2), int)
        intfile_refs = np.zeros((2), int)
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
        intfile_refs[0] = f_ptr
        if self.intfile_ids is not None:
            for fid in self.intfile_ids:
                table_ids.add_row(dict(file_id=fid))
                f_ptr += 1
        intfile_refs[1] = f_ptr

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
                           intfile_refs=intfile_refs,
                           status=self.status))

    def update_table_row(self, table, row_idx):
        """Add this instance as a row on a `astropy.table.Table` """
        table[row_idx]['timestamp'] = self.timestamp
        table[row_idx]['status'] = status=self.status

    def check_status_logfile(self, checker_func):
        """Check on the status of this particular job using the logfile"""
        self.status = checker_func(self.logfile)
        return self.status


class JobArchive(object):
    """Class that keeps of all the jobs associated to an analysis.

    Parameters
    ----------

    table_file   : str
        Path to the file used to persist this `JobArchive`
    table        : `astropy.table.Table`
        Persistent representation of this `JobArchive`
    table_ids    : `astropy.table.Table`
        Ancillary table with information about file ids
    file_archive : `FileArchive`
        Archive with infomation about all this files used and produced by this analysis
    """

    # Singleton instance
    _archive = None

    def __init__(self, **kwargs):
        """C'tor

        Reads kwargs['job_archive_table'] and passes remain kwargs to self.file_archive
        """
        self._table_file = None
        self._table = None
        self._table_ids = None
        self._table_id_array = None
        self._cache = OrderedDict()
        self._file_archive = FileArchive.build_archive(**kwargs)
        self._read_table_file(kwargs['job_archive_table'])

    def __getitem__(self, fullkey):
        """ Return the `JobDetails` matching fullkey"""
        return self._cache[key]

    @property
    def table_file(self):
        """Return the path to the file used to persist this `JobArchive` """
        return self._table_file

    @property
    def table(self):
        """Return the persistent representation of this `JobArchive` """
        return self._table

    @property
    def table_ids(self):
        """Return the rpersistent epresentation of the ancillary info of this `JobArchive` """
        return self._table_ids

    @property
    def file_archive(self):
        """Return the `FileArchive` with infomation about all
        the files used and produced by this analysis"""
        return self._file_archive

    @property
    def cache(self):
        """Return the transiet representation of this `JobArchive` """
        return self._cache

    def _fill_cache(self):
        """Fill the cache from the `astropy.table.Table`"""
        for irow in range(len(self._table)):
            job_details = self.make_job_details(irow)
            self._cache[job_details.fullkey] = job_details

    def _read_table_file(self, table_file):
        """Read an `astropy.table.Table` from table_file to set up the `JobArchive`"""
        self._table_file = table_file
        if os.path.exists(self._table_file):
            self._table = Table.read(self._table_file, hdu='JOB_ARCHIVE')
            self._table_ids = Table.read(self._table_file, hdu='FILE_IDS')
        else:
            self._table, self._table_ids = JobDetails.make_tables({})
        self._table_id_array = self._table_ids['file_id'].data
        self._fill_cache()

    def make_job_details(self, row_idx):
        """Create a `JobDetails` from an `astropy.table.row.Row` """
        row = self._table[row_idx]
        job_details = JobDetails.create_from_row(row, self._table_id_array)
        job_details.get_file_paths(self._file_archive, self._table_id_array)
        self._cache[job_details.fullkey] = job_details
        return job_details

    def get_details(self, jobname, jobkey):
        """Get the `JobDetails` associated to a particular job instance"""
        fullkey = JobDetails.make_fullkey(jobkey, jobname)
        return self._cache[fullkey]

    def register_job(self, job_details):
        """Register a job in this `JobArchive` """
        # check to see if the job already exists
        try:
            job_details = self.get_details(job_details.jobname,
                                           job_details.jobkey)
            raise KeyError("Job %s:%s already exists in archive" % (job_details.jobname,
                                                                    job_details.jobkey))
        except KeyError:
            pass
        job_details.dbkey = len(self._table) + 1
        job_details.get_file_ids(self._file_archive, creator=job_details.dbkey)
        job_details.append_to_tables(self._table, self._table_ids)
        self._table_id_array = self._table_ids['file_id'].data
        self._cache[job_details.fullkey] = job_details
        return job_details

    def register_jobs(self, job_dict):
        """Register a bunch of jobs in this archive"""
        njobs = len(job_dict)
        sys.stdout.write("Registering %i total jobs: \n" % njobs)
        for i, job_details in enumerate(job_dict.values()):
            if i % 10 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
            self.register_job(job_details)
        sys.stdout.write('!\n')

    def register_job_from_link(self, link, key, **kwargs):
        """Register a job in the `JobArchive` from a `Link` object """
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
                                 file_dict=copy.deepcopy(link.files),
                                 sub_file_dict=copy.deepcopy(link.sub_files),
                                 status=status)
        self.register_job(job_details)
        return job_details

    def update_job(self, job_details):
        """Update a job in the `JobArchive` """
        other = self.get_details(job_details.jobname,
                                 job_details.jobkey)
        other.timestamp = job_details.timestamp
        other.status = job_details.status
        other.update_table_row(self._table, other.dbkey - 1)
        return other

    @staticmethod
    def build_temp_job_archive():
        """Build and return a `JobArchive` using defualt locations of
        persistent files. """
        try:
            os.unlink('job_archive_temp.fits')
            os.unlink('file_archive_temp.fits')
        except OSError:
            pass
        JobArchive._archive = JobArchive(job_archive_table='job_archive_temp.fits',
                                         file_archive_table='file_archive_temp.fits',
                                         base_path=os.path.abspath('.') + '/')
        return JobArchive._archive

    def write_table_file(self, job_table_file=None, file_table_file=None):
        """Write the table to self._table_file"""
        if self._table is None:
            raise RuntimeError("No table to write")
        if self._table_ids is None:
            raise RuntimeError("No ID table to write")
        if job_table_file is not None:
            self._table_file = table_file
        if self._table_file is None:
            raise RuntimeError("No output file specified for table")           
        write_tables_to_fits(self._table_file, [self._table, self._table_ids], clobber=True,
                             namelist=['JOB_ARCHIVE', 'FILE_IDS'])
        self._file_archive.write_table_file(file_table_file)

    def update_job_status(self, checker_func):
        """Update the status of all the jobs in the archive"""
        njobs = len(self.cache.keys())
        status_vect = np.zeros((5), int)
        print ("Updating status of %i jobs: "%njobs)
        for i, key in enumerate(self.cache.keys()):
            if i % 200 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
            job_details = self.cache[key]
            job_details.check_status_logfile(checker_func)
            job_details.update_table_row(self._table, job_details.dbkey - 1)
            status_vect[job_details.status] += 1
            
        sys.stdout.write("!\n")
        sys.stdout.flush()
        sys.stdout.write("Summary:\n")
        sys.stdout.write("  no_job:   %i\n"%status_vect[0])
        sys.stdout.write("  pending:  %i\n"%status_vect[1])
        sys.stdout.write("  running:  %i\n"%status_vect[2])
        sys.stdout.write("  done:     %i\n"%status_vect[3])
        sys.stdout.write("  failed:   %i\n"%status_vect[4])

    @staticmethod
    def get_archive():
        """Return the singleton `JobArchive` instance """
        return JobArchive._archive

    @staticmethod
    def build_archive(**kwargs):
        """Return the singleton `JobArchive` instance, building it if needed """
        if JobArchive._archive is None:
            JobArchive._archive = JobArchive(**kwargs)
        return JobArchive._archive


def main_browse():
    """Entry point for command line use for browsing a JobArchive """
    parser = argparse.ArgumentParser(usage="job_archive.py [options]",
                                     description="Browse a job archive")

    parser.add_argument('--jobs', action='store', dest='job_archive_table',
                        type=str, default='job_archive_temp.fits', help="Job archive file")
    parser.add_argument('--files', action='store', dest='file_archive_table',
                        type=str, default='file_archive_temp.fits', help="File archive file")
    parser.add_argument('--base', action='store', dest='base_path',
                        type=str, default=os.path.abspath('.'), help="File archive base path")
    
    args = parser.parse_args(sys.argv[1:])    
    JobArchive.build_archive(**args.__dict__)


if __name__ == '__main__':
    main_browse()
    ja = JobArchive.get_archive()

    from fermipy.jobs.lsf_impl import check_log
    ja.update_job_status(check_log)
