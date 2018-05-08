# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Implementation of `ScatterGather` class for dealing with LSF batch jobs
"""
from __future__ import absolute_import, division, print_function

import sys
import os
import time

from fermipy.jobs.job_archive import JobStatus
from fermipy.jobs.sys_interface import clean_job, SysInterface


class NativeInterface(SysInterface):
    """Implmentation of ScatterGather that uses the native system"""
    string_exited = 'Exited with exit code'
    string_successful = 'Successfully completed'

    def __init__(self, **kwargs):
        """C'tor

        Keyword arguements
        ------------------

        jobs_per_cycle : int [20]
            Maximum number of jobs to submit in each cycle

        time_per_cycle : int [15]
            Time per submission cycle in seconds

        max_job_age : int [90]
            Max job age in minutes
        """
        super(NativeInterface, self).__init__(**kwargs)
        self._time_per_cycle = kwargs.pop('time_per_cycle', 15)
        self._jobs_per_cycle = kwargs.pop('jobs_per_cycle', 20)

    def dispatch_job_hook(self, link, key, job_config, logfile, stream=sys.stdout):
        """Send a single job to be executed

        Parameters
        ----------

        link : `fermipy.jobs.chain.Link`
            The link used to invoke the command we are running

        key : str
            A string that identifies this particular instance of the job

        job_config : dict
            A dictionrary with the arguments for the job.  Used with
            the self._command_template job template

        logfile : str
            The logfile for this job, may be used to check for success/ failure
        """
        full_sub_dict = job_config.copy()

        full_command = "%s >& %s" % (
            link.command_template().format(**full_sub_dict), logfile)

        logdir = os.path.dirname(logfile)

        if self._dry_run:
            sys.stdout.write("%s\n" % full_command)
        else:
            try:
                os.makedirs(logdir)
            except OSError:
                pass
            os.system(full_command)

    def submit_jobs(self, link, job_dict=None, job_archive=None, stream=sys.stdout):
        """Submit all the jobs in job_dict """
        if link is None:
            return JobStatus.no_job
        if job_dict is None:
            job_keys = link.jobs.keys()
        else:
            job_keys = sorted(job_dict.keys())

        # copy & reverse the keys b/c we will be popping item off the back of
        # the list
        unsubmitted_jobs = job_keys
        unsubmitted_jobs.reverse()

        failed = False
        while unsubmitted_jobs:
            njob_to_submit = min(self._jobs_per_cycle,
                                 len(unsubmitted_jobs))

            if self._dry_run:
                njob_to_submit = len(unsubmitted_jobs)

            for i in range(njob_to_submit):
                job_key = unsubmitted_jobs.pop()

                # job_details = job_dict[job_key]
                job_details = link.jobs[job_key]
                job_config = job_details.job_config
                if job_details.status == JobStatus.failed:
                    clean_job(job_details.logfile, {}, self._dry_run)
                    # clean_job(job_details.logfile,
                    #          job_details.outfiles, self.args['dry_run'])

                job_config['logfile'] = job_details.logfile
                new_job_details = self.dispatch_job(link, job_key, job_archive)
                if new_job_details.status == JobStatus.failed:
                    failed = True
                    clean_job(new_job_details.logfile,
                              new_job_details.outfiles, self._dry_run)
                link.jobs[job_key] = new_job_details

            if unsubmitted_jobs:
                print('Sleeping %.0f seconds between submission cycles' %
                      self._time_per_cycle)
                time.sleep(self._time_per_cycle)

        return failed


def get_native_default_args():
    """ Get the correct set of batch jobs arguments.
    """
    native_default_args = dict(max_jobs=500,
                               time_per_cycle=15,
                               jobs_per_cycle=20,
                               max_job_age=90,
                               no_batch=False)
    return native_default_args.copy()
