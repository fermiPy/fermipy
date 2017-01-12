# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Abstract interface for parallel execution of multiple jobs
"""
from __future__ import absolute_import, division, print_function


import sys
import os
import time
import argparse

#from enum import Enum

from fermipy.jobs.job_archive import get_timestamp, JobStatus, JobDetails

ACTIONS = ['run', 'resubmit', 'init',
           'scatter', 'gather', 'check_status', 'config']


def remove_file(filepath, dry_run=False):
    """Remove a file

    Catches exception if the file does not exist.
    """
    if dry_run:
        sys.stdout.write("rm %s\n" % filepath)
    else:
        try:
            os.remove(filepath)
        except OSError:
            pass


def clean_job(logfile, outfiles, dry_run=False):
    """Removes log file and files created by failed jobs
    """
    remove_file(logfile, dry_run)
    for outfile in outfiles.values():
        remove_file(outfile, dry_run)


class ConfigMaker(object):
    """ Abstract helper class to build configuration dictionaries for parallel jobs
    """

    def __init__(self):
        """ C'tor
        """
        pass

    def add_arguments(self, parser, action):
        """Hook to add arguments to the command line argparser """
        raise NotImplementedError("ScatterGather.ConfigMaker.add_arguments")

    def make_base_config(self, args):
        """Hook to build a baseline job configuration

        Sub-class implementation should return a dictionary with the job options
        """
        raise NotImplementedError("ScatterGather.ConfigMaker.make_base_config")

    def build_job_configs(self, args):
        """Hook to build job configurations

        Sub-class implementation should return three dictionaries:

        input_config : dict
            Configuration options passed to initializers

        job_configs : dict
            Dictionary of dictionaries passed to parallel jobs

        output_config : dict
            Configuration options passed to gatherer
        """
        raise NotImplementedError("ScatterGather.ConfigMaker.build_job_configs")


class ScatterGather(object):
    """ Abstract base class to dispatch several jobs in parallel and
    collect and merge the results
    """
    default_init_logfile = 'init.log'
    default_prefix_logfile = 'scatter'
    default_gather_logfile = 'gather.log'

    def __init__(self, **kwargs):
        """ C'tor

        Keyword arguements
        ---------------
        config_maker : `ConfigMaker'
            Object used to translate arguments
            Must have functions 'add_arguments' and 'build_job_configs'
            Defaults to ConfigMaker()

        usage : str
            Usage string for argument parser

        description : str
            Description string for argument parser

        job_archive : `fermipy.job_archive.JobArchive' [optional]
            Archive used to track jobs and associated data products
            Defaults to None

        initialize : `fermipy.chain.Link' [optional]
            Link run for the initialization stage
            Defaults to None

        scatter : `fermipy.chain.Link'
            Link run for the scatter stage
            Defaults to None

        gather : `fermipy.chain.Link'
            Link run for the gather stage
            Defaults to None
        """
        self._config_maker = kwargs.pop('config_maker', ConfigMaker())
        self._usage = kwargs.pop('usage', "")
        self._description = kwargs.pop('description', "")
        self._job_archive = kwargs.pop('job_archive', None)
        self._initialize_link = kwargs.pop('initialize', None)
        self._scatter_link = kwargs.pop('scatter', None)
        self._gather_link = kwargs.pop('gather', None)
        self._parser = None
        self._args = None
        self._base_config = None
        self._input_config = {}
        self._job_configs = {}
        self._output_config = {}
        self._job_dict = {}

    @property
    def config_maker(self):
        """Return the object used to translate arguments """
        return self._config_maker

    @property
    def base_config(self):
        """Return the baseline job configuration """
        return self._base_config

    @property
    def job_archive(self):
        """Return the `JobArchive` object"""
        return self._job_archive

    @property
    def initialize_link(self):
        """Return the `Link` object used to initial the processing"""
        return self._initialize_link

    @property
    def scatter_link(self):
        """Return the `Link` object used the scatter phase of processing"""
        return self._scatter_link

    @property
    def gather_link(self):
        """Return the `Link` object used the scatter phase of processing"""
        return self._gather_link

    @property
    def args(self):
        """Return the parsed command line arguments"""
        return self._args

    def check_job(self, job_details):
        """ Check the status of a specfic job """
        raise NotImplementedError('ScatterGather.check_job')

    def dispatch_job_hook(self, link, key, job_config, logfile):
        """Hook to dispatch a single job"""
        raise NotImplementedError("ScatterGather.dispatch_job_hook")

    def _make_init_logfile_name(self):
        """ Hook to inster the name of a logfile into the input config """
        self._input_config['logfile'] = self._input_config.get('logfile',
                                                               ScatterGather.default_init_logfile)

    @staticmethod
    def _make_scatter_logfile_name(key, job_config):
        """ Hook to inster the name of a logfile into the input config """
        logfile = job_config.get('logfile', "%s_%s.log" %
                                 (ScatterGather.default_prefix_logfile, key))
        job_config['logfile'] = logfile

    def _make_gather_logfile_name(self):
        """ Hook to construct the name of a logfile the gatherer """
        self._output_config['logfile'] = self._output_config.get('logfile',
                                                                 self.default_gather_logfile)

    def check_status(self):
        """ Check on the status of all the jobs in job dict """
        running = True
        first = True
        while running:
            if first:
                first = False
            else:
                print ("Sleeping %.0f seconds between status checks" %
                       self.args.job_check_sleep)
                time.sleep(self.args.job_check_sleep)

            running, failed = self._check_completion()
            if self.args.force_gather:
                sys.stdout.write('Forcing results gathering\n')
                running = False
            elif self.args.check_status_once:
                self.print_update()
                break
            if self.args.print_update:
                self.print_update()

        if failed:
            self.print_update()
            self.print_failed()

        return running, failed

    def _merge_config(self, config_in):
        """ Merge a configuration with the baseline, return the merged configuration """
        config_out = self._base_config.copy()
        config_out.update(config_in)
        return config_out


    def _check_completion(self):
        """ Internal function to check the completion of all the dispatched jobs


        Returns (bool, bool)
        ---------------
        running : bool
            True if the jobs are still running

        failed : bool
            True if any jobs have failed
        """
        failed = False
        running = False
        for job_key, job_details in self._job_dict.items():
            if job_details.status == JobStatus.failed:
                failed = True
                continue
            elif job_details.status == JobStatus.done:
                continue
            job_details.status = self.check_job(job_details)
            if job_details.status == JobStatus.failed:
                failed = True
            elif job_details.status in [JobStatus.pending, JobStatus.running]:
                running = True
            self._job_dict[job_key] = job_details

        return running, failed

    def build_configs(self, args):
        """ Build the configuration objects """
        self._base_config = self._config_maker.make_base_config(args)
        self._input_config, self._job_configs, self._output_config =\
            self._config_maker.build_job_configs(args)

    def initialize(self):
        """Run the initialization `Link' using self._input_config as input """
        if self._initialize_link is None:
            return JobStatus.no_job
        self._make_init_logfile_name()
        job_details = self.dispatch_job(self._initialize_link, 'init',
                                        self._input_config)
        return job_details.status

    def submit_jobs(self, job_dict=None):
        """Run the scatter `Link' with all of the items in self._job_configs as input """
        failed = False
        if self._scatter_link:
            return failed
        if job_dict is None:
            job_dict = self._job_dict
        for job_key, job_details in sorted(job_dict.items()):
            job_config = job_details.job_config
            # clean failed jobs
            if job_details.status == JobStatus.failed:
                clean_job(job_details.logfile, job_details.outfiles, self.args.dry_run)
            job_config['logfile'] = job_details.logfile
            new_job_details = self.dispatch_job(self._scatter_link, job_key, job_config)
            if new_job_details.status == JobStatus.failed:
                failed = True
                clean_job(new_job_details.logfile, new_job_details.outfiles, self.args.dry_run)
            self._job_dict[job_key] = new_job_details
        if failed:
            return JobStatus.failed
        return JobStatus.done

    def gather_results(self):
        """Run the gather `Link' using self._output_config as input """
        if self._gather_link is None:
            return JobStatus.no_job
        self._make_gather_logfile_name()
        job_details = self.dispatch_job(self._gather_link, 'gather',
                                        self._output_config)
        return job_details.status

    @staticmethod
    def create_job_details(link, key, job_config, logfile, status):
        """Create a `JobDetails' object for a single job

        Parameters:
        ---------------
        link : `Link`
            Link object that sendes the job

        key : str
            Key used to identify this particular job

        job_config : dict
            Dictionary with arguements passed to this particular job

        logfile : str
            Name of the associated log file
        
        status : int
            Current status of the job

        Returns `JobDetails' object
        """
        link.update_args(job_config)
        job_details = JobDetails(jobname=link.linkname,
                                 jobkey=key, 
                                 appname=link.appname,
                                 logfile=logfile,
                                 job_config=job_config,
                                 timestamp=get_timestamp(),
                                 infiles=link.input_files,
                                 outfiles=link.output_files,
                                 status=status)     
        return job_details

    def dispatch_job(self, link, key, job_config):
        """ Function to dispatch a single job

        Parameters:
        ---------------
        link : `Link`
            Link object that sendes the job

        key : str
            Key used to identify this particular job

        job_config : dict
            Dictionary with arguements passed to this particular job

        Returns `JobDetails' object
        """
        logfile = job_config['logfile']
        try:
            self.dispatch_job_hook(link, key, job_config, logfile)
            status = JobStatus.running
        except IOError:
            status = JobStatus.failed
        link.update_args(job_config)
        job_details = ScatterGather.create_job_details(link, key, job_config, logfile, status)                       
        if self._job_archive is not None:
            self._job_archive.register_job(job_details)
        return job_details

    def _add_arguments(self, action):
        """ Hook to add arguments to the command line argparser """
        self._config_maker.add_arguments(self._parser, action)
        if action in ['run', 'resubmit', 'scatter']:
            self._parser.add_argument('--dry_run', action='store_true', default=False,
                                      help='Print commands, but do not execute them')
            self._parser.add_argument('--job_check_sleep', type=int, default=300,
                                      help='Sleep time between checking on job status (s)')
            self._parser.add_argument('--force_gather', action='store_true', default=False,
                                      help='Force gathering stage')
            self._parser.add_argument('--print_update', action='store_true', default=False,
                                      help='Print summary of job status')
            self._parser.add_argument('--check_status_once', action='store_true', default=False,
                                      help='Check status only once before proceeding')

    def _parse_arguments(self, action, argv):
        """ Build and fill an ArgumentParser object """
        self._parser = argparse.ArgumentParser(usage=self._usage,
                                               description=self._description)
        try:
            self._add_arguments(action)
        except argparse.ArgumentError:
            pass

        self._args = self._parser.parse_args(argv)
        return self._args

    def print_update(self, stream=sys.stdout):
        """ Print an update about the current number of jobs running """
        n_total = len(self._job_configs)
        n_running = 0
        n_done = 0
        n_failed = 0
        n_pending = 0
        for job_dets in self._job_dict.values():
            if job_dets.status == JobStatus.running:
                n_running += 1
            elif job_dets.status == JobStatus.done:
                n_done += 1
            elif job_dets.status == JobStatus.failed:
                n_failed += 1
            elif job_dets.status == JobStatus.pending:
                n_pending += 1

        stream.write("Status :\n  Total  : %i\n  Pending: %i\n" % (n_total, n_pending))
        stream.write("Running: %i\n  Done   : %i\n  Failed : %i\n" % (n_running, n_done, n_failed))

    def print_failed(self, stream=sys.stderr):
        """ Print list of the failed jobs """
        for job_key, job_details in sorted(self._job_configs.items()):
            if job_details.status == JobStatus.failed:
                stream.write("Failed job %s : log = %s\n" %
                             (job_key, job_details.logfile))

    def __call__(self, argv):
        """ Parses command line arguments and runs the requested action

        Parameters:
        ---------------
        argv : list-like
            List of command line arguments
            The first two arguments should be the command itself and the action
            The remaining arguments are passed by the helper classes

        Returns str with status
        """
        action = argv[1]
        return self.invoke(action, argv[2:])

    def invoke(self, action, argv):
        """ Invoke this object to preform a particular action

        Parameters:
        ---------------
        action : str
            The requested action
            'initialize' : Do only the initialization step
            'scatter' : Do only the parallel job dispatching step
            'gather' : Do only the results gathering step
            'check_status' : Check on the status of dispatched jobs
            'run' : Do 'initialize', 'scatter', 'gather'
            'resubmit' : Do 'scatter' on failed jobs and 'gather' results

        argv : list-like
            List of command line arguments, passed to helper classes

        Returns str with status
        """
        if action not in ACTIONS:
            sys.stderr.write("Unrecognized action %s, options are %s\n" % (action, ACTIONS))
            sys.stderr.write("  Try: <command> <action> -h for specific help\n")

        args = self._parse_arguments(action, argv)

        if action in ['run', 'resubmit', 'init', 'scatter', 'gather', 'check_status', 'config']:
            self.build_configs(args)

        if action in ['init', 'scatter', 'gather', 'check_status']:
            # This is called explicitly in run and resubmit
            self.build_job_dict()

        if action == 'run':
            return self.run()
        elif action == 'resubmit':
            return self.resubmit()
        elif action == 'init':
            return self.initialize()
        elif action == 'scatter':
            return self.submit_jobs()
        elif action == 'gather':
            return self.gather_results()
        elif action == 'check_status':
            running, failed = self.check_status()
            if failed:
                return job_details.failed
            elif running:
                return job_details.running
            elif self.args.dry_run:
                return JobStatus.no_job
            return JobStatus.done
        elif action == 'config':
            return JobStatus.done

    def build_job_dict(self):
        """ Build a dictionary of JobDetails objects """
        if self.args.dry_run:
            status = JobStatus.no_job
        else:
            status = JobStatus.pending
        for jobkey, job_config in sorted(self._job_configs.items()):
            full_job_config = self._merge_config(job_config)
            ScatterGather._make_scatter_logfile_name(jobkey, full_job_config)
            logfile = full_job_config.pop('logfile')
            self._job_dict[jobkey] = ScatterGather.create_job_details(self._scatter_link,
                                                                      key=jobkey,
                                                                      job_config=full_job_config,
                                                                      logfile=logfile,
                                                                      status=status)

    def run(self):
        """ Function to dipatch jobs and collect results
        """
        init_status = self.initialize()
        if init_status == JobStatus.failed:
            return JobStatus.failed

        self.build_job_dict()

        scatter_status = self.submit_jobs()
        if scatter_status == JobStatus.failed:
            return JobStatus.failed

        running, failed = self.check_status()
        if self.args.force_gather or (running is False):
            gather_status = self.gather_results()
            return gather_status

        if self.args.dry_run:
            return JobStatus.no_job
        return JobStatus.done

    def resubmit(self):
        """ Function to dipatch jobs and collect results
        """
        self.build_job_dict()
        running, failed = self.check_status()
        if failed is False:
            return JobStatus.done

        failed_jobs = {}
        for job_key, job_details in self._job_dict.items():
            if job_details.status == JobStatus.failed:
                failed_jobs[job_key] = job_details

        scatter_status = self.submit_jobs(failed_jobs)
        if scatter_status == JobStatus.failed:
            return JobStatus.failed

        running, failed = self.check_status()
        if self.args.force_gather or (running is False):
            gather_status = self.gather_results()
            return gather_status
        if self.args.dry_run:
            return JobStatus.no_job
        return JobStatus.done
