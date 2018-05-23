# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Abstract interface for parallel execution of multiple jobs.

The main class is `ScatterGather`, which can submit many instances
of a job with different configurations.
"""
from __future__ import absolute_import, division, print_function

import sys
import time

import numpy as np

from fermipy.jobs.batch import get_batch_job_interface
from fermipy.jobs.job_archive import JobStatus,\
    JobStatusVector, JobDetails, JobArchive, JOB_STATUS_STRINGS
from fermipy.jobs.link import extract_arguments, Link

from fermipy.jobs import defaults

ACTIONS = ['run', 'resubmit', 'check_status', 'config', 'skip', 'clean']


class ScatterGather(Link):
    """ Class to dispatch several jobs in parallel and
    collect and merge the results.

    Sub-classes will need to generatare configuration
    for the jobs that they launch.

    Parameters
    ----------

    clientclass : type
        Type of `Link` object managed by this class.

    job_time : int
        Estimated maximum time it takes to run a job
        This is used to manage batch farm scheduling and
        checking for completion.

    """
    appname = 'dummy-sg'
    usage = "%s [options]" % (appname)
    description = "Run multiple analyses"
    clientclass = None

    job_time = 1500
    default_prefix_logfile = 'scatter'

    default_options = dict()
    default_options_base = dict(action=defaults.jobs['action'],
                                dry_run=defaults.jobs['dry_run'],
                                job_check_sleep=defaults.jobs['job_check_sleep'],
                                print_update=defaults.jobs['print_update'],
                                check_status_once=defaults.jobs['check_status_once'])

    def __init__(self, link, **kwargs):
        """C'tor

        Keyword arguments
        -----------------
        interface : `SysInterface` subclass
            Object used to interface with batch system

        usage : str
            Usage string for argument parser

        description : str
            Description string for argument parser

        job_archive : `fermipy.job_archive.JobArchive` [optional]
            Archive used to track jobs and associated data products
            Defaults to None

        scatter : `fermipy.chain.Link`
            Link run for the scatter stage
            Defaults to None

        """
        kwargs.setdefault('interface', get_batch_job_interface(self.job_time))
        self._scatter_link = link
        self.default_options.update(self.default_options_base.copy())
        Link.__init__(self, **kwargs)
        # Override the value of job_check_sleep to avoid excess waiting
        job_check_sleep = np.clip(int(self.job_time / 5), 60, 300)
        self.args['job_check_sleep'] = job_check_sleep
        self._job_configs = {}

    @property
    def scatter_link(self):
        """Return the `Link` object used the scatter phase of processing"""
        return self._scatter_link

    @classmethod
    def _make_scatter_logfile_name(cls, key, linkname, job_config):
        """Hook to inster the name of a logfile into the input config """
        logfile = job_config.get('logfile', "%s_%s_%s.log" %
                                 (cls.default_prefix_logfile, linkname, key))
        job_config['logfile'] = logfile

    @classmethod
    def create(cls, **kwargs):
        """Build and return a `ScatterGather` object """
        linkname = kwargs.setdefault('linkname', cls.clientclass.linkname_default)
        # Don't use setdefault b/c we don't want to build a JobArchive
        # Unless it is needed
        job_archive = kwargs.get('job_archive', None)
        if job_archive is None:
            job_archive = JobArchive.build_temp_job_archive()
            kwargs.setdefault('job_archive', job_archive)
        kwargs_client = dict(linkname=linkname,
                             link_prefix=kwargs.get('link_prefix', ''),
                             file_stage=kwargs.get('file_stage', None),
                             job_archive=job_archive)
        link = cls.clientclass.create(**kwargs_client)
        sg = cls(link, **kwargs)
        return sg

    @classmethod
    def main(cls):
        """Hook for command line interface to sub-classes """
        link = cls.create()
        link._invoke(sys.argv[1:])

    def build_job_configs(self, args):
        """Hook to build job configurations

        Sub-class implementation should return:

        job_configs : dict
            Dictionary of dictionaries passed to parallel jobs
        """
        raise NotImplementedError("ScatterGather.build_job_configs")


    def _latch_file_info(self):
        """Internal function to update the dictionaries
        keeping track of input and output files
        """
        self.files.file_dict.clear()
        self.sub_files.file_dict.clear()
        self.files.latch_file_info(self.args)
        self._scatter_link._update_sub_file_dict(self.sub_files)

    def _check_link_completion(self, link, fail_pending=False, fail_running=False):
        """Internal function to check the completion of all the dispatched jobs

        Returns
        -------

        status_vect : `JobStatusVector`
            Vector that summarize the number of jobs in various states.
        """

        status_vect = JobStatusVector()
        for job_key, job_details in link.jobs.items():
            # if job_details.status == JobStatus.failed:
            #    failed = True
            #    continue
            # elif job_details.status == JobStatus.done:
            #    continue
            if job_key.find(JobDetails.topkey) >= 0:
                continue
            job_details.status = self._interface.check_job(job_details)
            if job_details.status == JobStatus.pending:
                if fail_pending:
                    job_details.status = JobStatus.failed
            elif job_details.status == JobStatus.running:
                if fail_running:
                    job_details.status = JobStatus.failed
            status_vect[job_details.status] += 1
            link.jobs[job_key] = job_details
            link._set_status_self(job_details.jobkey, job_details.status)

        return status_vect

    def _build_job_dict(self):
        """Build a dictionary of `JobDetails` objects for the internal `Link`"""
        if self.args['dry_run']:
            status = JobStatus.unknown
        else:
            status = JobStatus.not_ready

        base_config = self.scatter_link.args

        for jobkey, job_config in sorted(self._job_configs.items()):
            full_job_config = base_config.copy()
            full_job_config.update(job_config)
            ScatterGather._make_scatter_logfile_name(jobkey,
                                                     self.linkname, full_job_config)
            logfile = job_config.get('logfile')
            self._scatter_link._register_job(key=jobkey,
                                             job_config=full_job_config,
                                             logfile=logfile,
                                             status=status)

    def _run_link(self, stream=sys.stdout, dry_run=False,
                  stage_files=True, resubmit_failed=False):
        """Internal function that actually runs this link.

        This checks if input and output files are present.

        If input files are missing this will raise `OSError` if dry_run is False
        If all output files are present this will skip execution.

        Parameters
        -----------
        stream : `file`
            Stream that this `Link` will print to,
            must have 'write' function.

        dry_run : bool
            Print command but do not run it.

        stage_files : bool
            Stage files to and from the scratch area.

        resubmit_failed : bool
            Resubmit failed jobs.
        """
        if resubmit_failed:
            self.args['action'] = 'resubmit'
        argv = self._make_argv()
        if dry_run:
            argv.append('--dry_run')
        self._invoke(argv, stream)

    def _invoke(self, argv, stream=sys.stdout):
        """Invoke this object to preform a particular action

        Parameters
        ----------

        argv : list
            List of command line arguments, passed to helper classes

        stream : `file`
            Stream that this function will print to,
            must have 'write' function.

        Returns
        -------
        status_vect : `JobStatusVector`
            Vector that summarize the number of jobs in various states.

        """
        args = self._run_argparser(argv)

        if args.action not in ACTIONS:
            sys.stderr.write(
                "Unrecognized action %s, options are %s\n" % (args.action, ACTIONS))

        if args.action == 'skip':
            return JobStatus.no_job
        elif args.action in ['run', 'resubmit', 'check_status', 'config']:
            self._job_configs = self.build_job_configs(args.__dict__)

        self._interface._dry_run = args.dry_run

        if args.action == 'run':
            status_vect = self.run_jobs(stream)
        elif args.action == 'resubmit':
            status_vect = self.resubmit(stream)
        elif args.action == 'check_status':
            self._build_job_dict()
            status_vect = self.check_status(stream)
        elif args.action == 'config':
            self._build_job_dict()
            status_vect = JobStatusVector()
            status_vect[JobStatus.done] += 1

        return status_vect

    def update_args(self, override_args):
        """Update the arguments used to invoke the application

        Note that this will also update the dictionary of input and output files

        Parameters
        ----------

        override_args : dict
            dictionary of arguments to override the current values
        """
        self.args = extract_arguments(override_args, self.args)
        self._job_configs = self.build_job_configs(self.args)
        if not self._scatter_link.jobs:
            self._build_job_dict()
        self._latch_file_info()

    def clear_jobs(self, recursive=True):
        """Clear the self.jobs dictionary that contains information
        about jobs associated with this `ScatterGather`

        If recursive is True this will include jobs from all internal `Link`
        """
        if recursive:
            self._scatter_link.clear_jobs(recursive)
        self.jobs.clear()

    def get_jobs(self, recursive=True):
        """Return a dictionary with all the jobs

        If recursive is True this will include jobs from all internal `Link`
        """
        if recursive:
            ret_dict = self.jobs.copy()
            ret_dict.update(self._scatter_link.get_jobs(recursive))
            return ret_dict
        return self.jobs


    def check_status(self, stream=sys.stdout,
                     check_once=False,
                     fail_pending=False, fail_running=False,
                     no_wait=False, do_print=True,
                     write_status=False):
        """Loop to check on the status of all the jobs in job dict.

        Parameters
        -----------
        stream : `file`
            Stream that this function will print to,
            Must have 'write' function.

        check_once : bool
            Check status once and exit loop.

        fail_pending : `bool`
            If True, consider pending jobs as failed

        fail_running : `bool`
            If True, consider running jobs as failed

        no_wait : bool
            Do not sleep before checking jobs.

        do_print : bool
            Print summary stats.

        write_status : bool
            Write the status the to log file.

        Returns
        -------
        status_vect : `JobStatusVector`
            Vector that summarize the number of jobs in various states.
        """
        running = True
        first = True

        if not check_once:
            if stream != sys.stdout:
                sys.stdout.write('Checking status (%is): ' %
                                 self.args['job_check_sleep'])
                sys.stdout.flush()

        status_vect = JobStatusVector()
        while running:
            if first:
                first = False
            elif self.args['dry_run']:
                break
            elif no_wait:
                pass
            else:
                stream.write("Sleeping %.0f seconds between status checks\n" %
                             self.args['job_check_sleep'])
                if stream != sys.stdout:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                time.sleep(self.args['job_check_sleep'])

            status_vect = self._check_link_completion(self._scatter_link,
                                                      fail_pending, fail_running)
            if self.args['check_status_once'] or check_once or no_wait:
                if do_print:
                    self.print_update(stream, status_vect)
                break

            if self.args['print_update']:
                if do_print:
                    self.print_update(stream, status_vect)

            if self._job_archive is not None:
                self._job_archive.write_table_file()

            n_total = status_vect.n_total
            n_done = status_vect.n_done
            n_failed = status_vect.n_failed
            if n_done + n_failed == n_total:
                running = False

        status = status_vect.get_status()
        if status in [JobStatus.failed, JobStatus.partial_failed]:
            if do_print:
                self.print_update(stream, status_vect)
                self.print_failed(stream)
            if write_status:
                self._write_status_to_log(status, stream)
        else:
            if write_status:
                self._write_status_to_log(0, stream)

        self._set_status_self(status=status)
        if not check_once:
            if stream != sys.stdout:
                sys.stdout.write("! %s\n" % (JOB_STATUS_STRINGS[status]))

        if self._job_archive is not None:
            self._job_archive.write_table_file()

        return status_vect

    def run_jobs(self, stream=sys.stdout):
        """Function to dipatch jobs and collect results

        Parameters
        -----------
        stream : `file`
            Stream that this function will print to,
            Must have 'write' function.

        Returns
        -------
        status_vect : `JobStatusVector`
            Vector that summarize the number of jobs in various states.
        """
        self._build_job_dict()

        self._interface._dry_run = self.args['dry_run']
        scatter_status = self._interface.submit_jobs(self.scatter_link,
                                                     job_archive=self._job_archive,
                                                     stream=stream)
        if scatter_status == JobStatus.failed:
            return JobStatus.failed

        status_vect = self.check_status(stream, write_status=True)
        return status_vect

    def resubmit(self, stream=sys.stdout, fail_running=False):
        """Function to resubmit failed jobs and collect results

        Parameters
        -----------
        stream : `file`
            Stream that this function will print to,
            Must have 'write' function.

        fail_running : `bool`
            If True, consider running jobs as failed

        Returns
        -------
        status_vect : `JobStatusVector`
            Vector that summarize the number of jobs in various states.

        """
        self._build_job_dict()
        status_vect = self.check_status(stream, check_once=True, fail_pending=True,
                                        fail_running=fail_running)
        status = status_vect.get_status()
        if status == JobStatus.done:
            return status

        failed_jobs = self._scatter_link.get_failed_jobs(True, True)
        if failed_jobs:
            scatter_status = self._interface.submit_jobs(self._scatter_link, failed_jobs,
                                                         job_archive=self._job_archive,
                                                         stream=stream)
            if scatter_status == JobStatus.failed:
                return JobStatus.failed

        status_vect = self.check_status(stream, write_status=True)

        if self.args['dry_run']:
            return JobStatus.unknown
        return status_vect

    def clean_jobs(self, recursive=False):
        """Clean up all the jobs associated with this object.

        If recursive is True this also clean jobs dispatch by this
        object."""
        self._interface.clean_jobs(self.scatter_link,
                                   clean_all=recursive)

    def run(self, stream=sys.stdout, dry_run=False,
            stage_files=True, resubmit_failed=True):
        """Runs this `Link`.

        This version is intended to be overwritten by sub-classes so
        as to provide a single function that behaves the same
        for all version of `Link`

        Parameters
        -----------
        stream : `file`
            Stream that this `Link` will print to,
            Must have 'write' function

        dry_run : bool
            Print command but do not run it.

        stage_files : bool
            Copy files to and from scratch staging area.

        resubmit_failed : bool
            Flag for sub-classes to resubmit failed jobs.
        """
        self._run_link(stream, dry_run, stage_files, resubmit_failed)

    def print_summary(self, stream=sys.stdout, indent="", recurse_level=2):
        """Print a summary of the activity done by this `Link`.

        Parameters
        ----------

        stream : `file`
            Stream to print to

        indent : str
            Indentation at start of line

        recurse_level : int
            Number of recursion levels to print
        """
        Link.print_summary(self, stream, indent, recurse_level)
        if recurse_level > 0:
            recurse_level -= 1
            indent += "  "
            stream.write("\n")
            self._scatter_link.print_summary(stream, indent, recurse_level)

    def print_update(self, stream=sys.stdout, job_stats=None):
        """Print an update about the current number of jobs running """
        if job_stats is None:
            job_stats = JobStatusVector()
            job_det_list = []
            job_det_list += self._scatter_link.jobs.values()

            for job_dets in job_det_list:
                if job_dets.status == JobStatus.no_job:
                    continue
                job_stats[job_dets.status] += 1

        stream.write("Status :\n  Total  : %i\n  Unknown: %i\n" %
                     (job_stats.n_total, job_stats[JobStatus.unknown]))
        stream.write("  Not Ready: %i\n  Ready: %i\n" %
                     (job_stats[JobStatus.not_ready], job_stats[JobStatus.ready]))
        stream.write("  Pending: %i\n  Running: %i\n" %
                     (job_stats[JobStatus.pending], job_stats[JobStatus.running]))
        stream.write("  Done: %i\n  Failed: %i\n" %
                     (job_stats[JobStatus.done], job_stats[JobStatus.failed]))

    def print_failed(self, stream=sys.stderr):
        """Print list of the failed jobs """
        for job_key, job_details in sorted(self.scatter_link.jobs.items()):
            if job_details.status == JobStatus.failed:
                stream.write("Failed job %s\n  log = %s\n" %
                             (job_key, job_details.logfile))

    def run_analysis(self, argv):
        """Implemented by sub-classes to run a particular analysis"""
        raise RuntimeError("run_analysis called for ScatterGather type object")
