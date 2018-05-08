# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Implementation of `ScatterGather` interface class for dealing with LSF batch jobs at SLAC
"""
from __future__ import absolute_import, division, print_function

import sys
import os
import time
import subprocess

from fermipy.jobs.job_archive import JobStatus
from fermipy.jobs.sys_interface import clean_job, SysInterface


def make_nfs_path(path):
    """Make a nfs version of a file path.
    This just puts /nfs at the beginning instead of /gpfs"""
    if os.path.isabs(path):
        fullpath = path
    else:
        fullpath = os.path.abspath(path)
    if len(fullpath) < 6:
        return fullpath
    if fullpath[0:6] == '/gpfs/':
        fullpath = fullpath.replace('/gpfs/', '/nfs/')
    return fullpath


def make_gpfs_path(path):
    """Make a gpfs version of a file path.
    This just puts /gpfs at the beginning instead of /nfs"""
    if os.path.isabs(path):
        fullpath = os.path.abspath(path)
    else:
        fullpath = os.path.abspath(path)
    if len(fullpath) < 5:
        return fullpath
    if fullpath[0:5] == '/nfs/':
        fullpath = fullpath.replace('/nfs/', '/gpfs/')
    return fullpath


def get_lsf_status():
    """Count and print the number of jobs in various LSF states
    """
    status_count = {'RUN': 0,
                    'PEND': 0,
                    'SUSP': 0,
                    'USUSP': 0,
                    'NJOB': 0,
                    'UNKNWN': 0}

    try:
        subproc = subprocess.Popen(['bjobs'],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        subproc.stderr.close()
        output = subproc.stdout.readlines()
    except OSError:
        return status_count

    for line in output[1:]:
        line = line.strip().split()

        status_count['NJOB'] += 1

        for k in status_count:
            if line[2] == k:
                status_count[k] += 1

    return status_count


def build_bsub_command(command_template, lsf_args):
    """Build and return a lsf batch command template

    The structure will be 'bsub -s <key> <value> <command_template>'
        where <key> and <value> refer to items in lsf_args
    """
    if command_template is None:
        return ""
    full_command = 'bsub -o {logfile}'
    for key, value in lsf_args.items():
        full_command += ' -%s' % key
        if value is not None:
            full_command += ' %s' % value
    full_command += ' %s' % command_template
    return full_command


class SlacInterface(SysInterface):
    """Implmentation of ScatterGather that uses LSF"""
    string_exited = 'Exited with exit code'
    string_successful = 'Successfully completed'

    def __init__(self, **kwargs):
        """C'tor

        Keyword arguements
        ------------------
        lsf_args : dict
            Dictionary of arguments passed to LSF

        max_jobs : int [500]
            Limit on the number of running or queued jobs

        jobs_per_cycle : int [20]
            Maximum number of jobs to submit in each cycle

        time_per_cycle : int [15]
            Time per submission cycle in seconds

        max_job_age : int [90]
            Max job age in minutes
        """
        super(SlacInterface, self).__init__(**kwargs)
        self._lsf_args = kwargs.pop('lsf_args', {})
        self._max_jobs = kwargs.pop('max_jobs', 500)
        self._time_per_cycle = kwargs.pop('time_per_cycle', 15)
        self._jobs_per_cycle = kwargs.pop('jobs_per_cycle', 20)
        self._max_job_age = kwargs.pop('max_job_age', 90)
        self._no_batch = kwargs.pop('no_batch', False)

    def dispatch_job_hook(self, link, key, job_config, logfile, stream=sys.stdout):
        """Send a single job to the LSF batch

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

        if self._no_batch:
            full_command = "%s >& %s" % (
                link.command_template().format(**full_sub_dict), logfile)
        else:
            full_sub_dict['logfile'] = logfile
            full_command_template = build_bsub_command(
                link.command_template(), self._lsf_args)
            full_command = full_command_template.format(**full_sub_dict)

        logdir = os.path.dirname(logfile)

        print_bsub = True
        if self._dry_run:
            if print_bsub:
                stream.write("%s\n" % full_command)
            return 0

        try:
            os.makedirs(logdir)
        except OSError:
            pass
        proc = subprocess.Popen(full_command.split(),
                                stderr=stream,
                                stdout=stream)
        proc.communicate()
        return proc.returncode

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
        if unsubmitted_jobs:
            if stream != sys.stdout:
                sys.stdout.write('Submitting jobs (%i): ' %
                                 len(unsubmitted_jobs))
                sys.stdout.flush()
        while unsubmitted_jobs:
            status = get_lsf_status()
            njob_to_submit = min(self._max_jobs - status['NJOB'],
                                 self._jobs_per_cycle,
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
                new_job_details = self.dispatch_job(
                    link, job_key, job_archive, stream)
                if new_job_details.status == JobStatus.failed:
                    failed = True
                    clean_job(new_job_details.logfile,
                              new_job_details.outfiles, self._dry_run)
                link.jobs[job_key] = new_job_details

            if unsubmitted_jobs:
                if stream != sys.stdout:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                stream.write('Sleeping %.0f seconds between submission cycles\n' %
                             self._time_per_cycle)
                time.sleep(self._time_per_cycle)

        if failed:
            return JobStatus.failed

        if stream != sys.stdout:
            sys.stdout.write('!\n')

        return JobStatus.done


def get_slac_default_args(job_time=1500):
    """ Create a batch job interface object.

    Parameters
    ----------

    job_time : int
        Expected max length of the job, in seconds.
        This is used to select the batch queue and set the
        job_check_sleep parameter that sets how often
        we check for job completion.

    """
    slac_default_args = dict(lsf_args={'W': job_time,
                                       'R': '\"select[rhel60&&!fell]\"'},
                             max_jobs=500,
                             time_per_cycle=15,
                             jobs_per_cycle=20,
                             max_job_age=90,
                             no_batch=False)
    return slac_default_args.copy()
