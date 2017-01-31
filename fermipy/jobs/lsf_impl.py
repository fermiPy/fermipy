# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Implementation of `ScatterGather` class for dealing with LSF batch jobs
"""
from __future__ import absolute_import, division, print_function

import sys
import os
import time
import subprocess

from fermipy.jobs.job_archive import JobStatus, JobArchive
from fermipy.jobs.scatter_gather import clean_job, ScatterGather


def get_lsf_status():
    """Count and print the number of jobs in various LSF states
    """
    status_count = {'RUN': 0,
                    'PEND': 0,
                    'SUSP': 0,
                    'USUSP': 0,
                    'NJOB': 0,
                    'UNKNWN': 0}

    subproc = subprocess.Popen(['bjobs'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    subproc.stderr.close()

    output = subproc.stdout.readlines()

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


class LsfScatterGather(ScatterGather):
    """Implmentation of ScatterGather that uses LSF"""

    default_options = ScatterGather.default_options.copy()
    default_options.update(dict(max_jobs=(500,
                                          'Limit on the number of running or queued jobs.', int),
                                jobs_per_cycle=(
                                    20, 'Maximum number of jobs to submit in each cycle.', int),
                                time_per_cycle=(
                                    15., 'Time per submission cycle in seconds.', float),
                                max_job_age=(90., 'Max job age in minutes.', float),))

    def __init__(self, **kwargs):
        """C'tor

        Keyword arguements
        ------------------

        lsf_exited : str ['Exited with exit code']
            String used to identify failed jobs

        lsf_successful : str ['Successfully completed']
            String used to identify completed jobs
        """
        super(LsfScatterGather, self).__init__(**kwargs)
        self._exited = kwargs.pop('lsf_exited', 'Exited with exit code')
        self._successful = kwargs.pop(
            'lsf_successful', 'Successfully completed')
        self._lsf_args = kwargs.pop('lsf_args', {})

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

        if self.no_batch:
            full_command = "%s >& %s" % (
                link.command_template().format(**full_sub_dict), logfile)
        else:
            full_sub_dict['logfile'] = logfile
            full_command_template = build_bsub_command(
                link.command_template(), self._lsf_args)
            full_command = full_command_template.format(**full_sub_dict)

        logdir = os.path.dirname(logfile)

        if self.args['dry_run']:
            sys.stdout.write("%s\n" % full_command)
        else:
            try:
                os.makedirs(logdir)
            except OSError:
                pass
            os.system(full_command)

    def submit_jobs(self, link, job_dict=None):
        """Submit all the jobs in job_dict """
        if self._scatter_link is None:
            return JobStatus.no_job
        if job_dict is None:
            job_dict = link.jobs
        job_keys = sorted(job_dict.keys())

        # copy & reverse the keys b/c we will be popping item off the back of
        # the list
        unsubmitted_jobs = job_keys
        unsubmitted_jobs.reverse()

        failed = False
        while len(unsubmitted_jobs) > 0:
            status = get_lsf_status()
            njob_to_submit = min(self.args['max_jobs'] - status['NJOB'],
                                 self.args['jobs_per_cycle'],
                                 len(unsubmitted_jobs))

            if self.args['dry_run']:
                njob_to_submit = len(unsubmitted_jobs)

            for i in range(njob_to_submit):
                job_key = unsubmitted_jobs.pop()

                job_details = job_dict[job_key]
                job_config = job_details.job_config
                if job_details.status == JobStatus.failed:
                    clean_job(job_details.logfile,
                              job_details.outfiles, self.args['dry_run'])
                job_config['logfile'] = job_details.logfile
                new_job_details = self.dispatch_job(
                    self._scatter_link, job_key)
                if new_job_details.status == JobStatus.failed:
                    failed = True
                    clean_job(new_job_details.logfile,
                              new_job_details.outfiles, self.args['dry_run'])
                job_dict[job_key] = new_job_details

            print('Sleeping %.0f seconds between submission cycles' %
                  self.args['time_per_cycle'])
            time.sleep(self.args['time_per_cycle'])

        return failed


def build_sg_from_link(link, config_maker, **kwargs):
    """Build a `ScatterGather` that will run multiple instance of a single link
    """
    kwargs['config_maker'] = config_maker
    kwargs['scatter'] = link
    job_archive = kwargs.get('job_archive', None)
    if job_archive is None:
        kwargs['job_archive'] = JobArchive.build_temp_job_archive()
    lsf_sg = LsfScatterGather(**kwargs)
    return lsf_sg
