# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Abstract interface for interactions with system for launching jobs.
"""
from __future__ import absolute_import, division, print_function

import os
import sys

from fermipy.jobs.job_archive import JobStatus


def remove_file(filepath, dry_run=False):
    """Remove the file at filepath

    Catches exception if the file does not exist.

    If dry_run is True, print name of file to be removed, but do not remove it.
    """
    if dry_run:
        sys.stdout.write("rm %s\n" % filepath)
    else:
        try:
            os.remove(filepath)
        except OSError:
            pass


def clean_job(logfile, outfiles, dry_run=False):
    """Removes log file and files created by failed jobs.

    If dry_run is True, print name of files to be removed, but do not remove them.
    """
    remove_file(logfile, dry_run)
    for outfile in outfiles.values():
        remove_file(outfile, dry_run)


def check_log(logfile, exited='Exited with exit code',
              successful='Successfully completed'):
    """Check a log file to determine status of LSF job

    Often logfile doesn't exist because the job hasn't begun
    to run. It is unclear what you want to do in that case...

    Parameters
    ----------

    logfile : str
        String with path to logfile

    exited  : str
        Value to check for in existing logfile for exit with failure

    successful : str
        Value to check for in existing logfile for success

    Returns str, one of 'Pending', 'Running', 'Done', 'Failed'
    """
    if not os.path.exists(logfile):
        return JobStatus.ready
    if exited in open(logfile).read():
        return JobStatus.failed
    elif successful in open(logfile).read():
        return JobStatus.done
    return JobStatus.running


class SysInterface(object):
    """  Base class to handle job dispatching interface """

    string_exited = 'Exited with exit code'
    string_successful = 'Successfully completed'

    """C'tor """

    def __init__(self, **kwargs):
        self._dry_run = kwargs.get('dry_run', False)
        self._job_check_sleep = kwargs.get('job_check_sleep', None)

    @classmethod
    def check_job(cls, job_details):
        """ Check the status of a specfic job """
        return check_log(job_details.logfile, cls.string_exited, cls.string_successful)

    def dispatch_job_hook(self, link, key, job_config, logfile, stream=sys.stdout):
        """Hook to dispatch a single job"""
        raise NotImplementedError("SysInterface.dispatch_job_hook")

    def dispatch_job(self, link, key, job_archive, stream=sys.stdout):
        """Function to dispatch a single job

        Parameters
        ----------

        link : `Link`
            Link object that sendes the job

        key : str
            Key used to identify this particular job

        job_archive : `JobArchive`
            Archive used to keep track of jobs

        Returns `JobDetails` object
        """
        try:
            job_details = link.jobs[key]
        except KeyError:
            print(key, link.jobs)
        job_config = job_details.job_config
        link.update_args(job_config)
        logfile = job_config['logfile']
        try:
            self.dispatch_job_hook(link, key, job_config, logfile, stream)
            job_details.status = JobStatus.running
        except IOError:
            job_details.status = JobStatus.failed

        if job_archive is not None:
            job_archive.register_job(job_details)
        return job_details

    def submit_jobs(self, link, job_dict=None, job_archive=None, stream=sys.stdout):
        """Run the `Link` with all of the items job_dict as input.

        If job_dict is None, the job_dict will be take from link.jobs

        Returns a `JobStatus` enum
        """
        failed = False
        if job_dict is None:
            job_dict = link.jobs

        for job_key, job_details in sorted(job_dict.items()):
            job_config = job_details.job_config
            # clean failed jobs
            if job_details.status == JobStatus.failed:
                clean_job(job_details.logfile,
                          job_details.outfiles, self._dry_run)
                # clean_job(job_details.logfile, {}, self._dry_run)
            job_config['logfile'] = job_details.logfile
            new_job_details = self.dispatch_job(
                link, job_key, job_archive, stream)
            if new_job_details.status == JobStatus.failed:
                failed = True
                clean_job(new_job_details.logfile,
                          new_job_details.outfiles, self._dry_run)
            link.jobs[job_key] = new_job_details
        if failed:
            return JobStatus.failed
        return JobStatus.done

    def clean_jobs(self, link, job_dict=None, clean_all=False):
        """ Clean up all the jobs associated with this link.

        Returns a `JobStatus` enum
        """
        failed = False
        if job_dict is None:
            job_dict = link.jobs

        for job_details in job_dict.values():
            # clean failed jobs
            if job_details.status == JobStatus.failed or clean_all:
                # clean_job(job_details.logfile, job_details.outfiles, self._dry_run)
                clean_job(job_details.logfile, {}, self._dry_run)
                job_details.status = JobStatus.ready
        if failed:
            return JobStatus.failed
        return JobStatus.done
