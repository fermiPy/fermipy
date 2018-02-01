# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Implementation of `ScatterGather` class for dealing with LSF batch jobs
"""
from __future__ import absolute_import, division, print_function

import sys
import os
import time
import subprocess

from fermipy.jobs.job_archive import get_timestamp, JobStatus, JobDetails
from fermipy.jobs.scatter_gather import clean_job, SG_Interface

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
        fullpath = pathos.path.abspath(path)
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

        for k in status_count.keys():

            if line[2] == k:
                status_count[k] += 1

    return status_count


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
        return JobStatus.pending
    if exited in open(logfile).read():
        return JobStatus.failed
    elif successful in open(logfile).read():
        return JobStatus.done
    else:
        return JobStatus.running


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


class LSF_Interface(SG_Interface):
    """Implmentation of ScatterGather that uses LSF"""

    def __init__(self, **kwargs):
        """C'tor

        Keyword arguements
        ------------------

        lsf_exited : str ['Exited with exit code']
            String used to identify failed jobs

        lsf_successful : str ['Successfully completed']
            String used to identify completed jobs

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
        super(LSF_Interface, self).__init__(**kwargs)
        self._exited = kwargs.pop('lsf_exited', 'Exited with exit code')
        self._successful = kwargs.pop('lsf_successful', 'Successfully completed')
        self._lsf_args = kwargs.pop('lsf_args', {})
        self._max_jobs = kwargs.pop('max_jobs', 500)
        self._time_per_cycle = kwargs.pop('time_per_cycle', 15)
        self._jobs_per_cycle = kwargs.pop('jobs_per_cycle', 20)
        self._max_job_age = kwargs.pop('max_job_age', 90)
        self._no_batch =  kwargs.pop('no_batch', False)

    def check_job(self, job_details):
        """Check the status of a single job

        Returns str, one of 'Pending', 'Running', 'Done', 'Failed'
        """
        return check_log(job_details.logfile, self._exited, self._successful)

    def dispatch_job_hook(self, link, key, job_config, logfile):
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
                sys.stdout.write("%s\n" % full_command)
        else:
            try:
                os.makedirs(logdir)
            except OSError:
                pass
            os.system(full_command)

    def submit_jobs(self, link, job_dict=None, job_archive=None):
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
        while len(unsubmitted_jobs) > 0:
            status = get_lsf_status()
            njob_to_submit = min(self._max_jobs - status['NJOB'],
                                 self._jobs_per_cycle,
                                 len(unsubmitted_jobs))

            if self._dry_run:
                njob_to_submit = len(unsubmitted_jobs)

            for i in range(njob_to_submit):
                job_key = unsubmitted_jobs.pop()

                #job_details = job_dict[job_key]
                job_details = link.jobs[job_key]
                job_config = job_details.job_config
                if job_details.status == JobStatus.failed:
                    clean_job(job_details.logfile, {}, self._dry_run)
                    #clean_job(job_details.logfile,
                    #          job_details.outfiles, self.args['dry_run'])
                    
                job_config['logfile'] = job_details.logfile
                new_job_details = self.dispatch_job(link, job_key, job_archive)
                if new_job_details.status == JobStatus.failed:
                    failed = True
                    clean_job(new_job_details.logfile,
                              new_job_details.outfiles, self._dry_run)
                link.jobs[job_key] = new_job_details

            if len(unsubmitted_jobs) > 0:
                print('Sleeping %.0f seconds between submission cycles' %
                      self._time_per_cycle)
                time.sleep(self._time_per_cycle)

        return failed




def get_lsf_default_args():
    lsf_default_args = dict(lsf_exited='Exited with exit code',
                            lsf_successful='Successfully completed',
                            lsf_args={'W': 1500,
                                      'R': '\"select[rhel60 && !fell]\"'},
                            max_jobs=500,
                            time_per_cycle=15,
                            jobs_per_cycle=20,
                            max_job_age=90,
                            no_batch=False)
    return lsf_default_args.copy()
