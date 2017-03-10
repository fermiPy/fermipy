# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Abstract interface for parallel execution of multiple jobs.

The main class is `ScatterGather`, which can submit many instances
of a job with different configurations.

The `ConfigMaker` abstract helper class is used to generate configurations.
"""
from __future__ import absolute_import, division, print_function


import sys
import os
import time
import argparse

#from enum import Enum

from fermipy.jobs.job_archive import JobStatus
from fermipy.jobs.chain import add_argument, extract_arguments, Link

ACTIONS = ['run', 'resubmit', 'init',
           'scatter', 'gather', 'check_status', 'config']


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


class ConfigMaker(object):
    """Abstract helper class to build configuration dictionaries for parallel jobs.

    Sub-classes will need to:

    Define options by passing a dictionary of option tuples to the c'tor.
    This will take a form something like:

    options=dict(string_opt=("default", "Some string", str),
                 float_opt=(3.0, "Some float", float),
                 list_opt=(None, "Some list", list))
    """

    def __init__(self, link, **kwargs):
        """C'tor
        """
        self.link = link
        self._options = kwargs.get('options', {})

    def add_options(self, option_dict):
        """Add options into an option dictionary"""
        option_dict.update(self._options)

    def add_arguments(self, parser, action):
        """Hook to add arguments to an `argparse.ArgumentParser` """
        if action is None:
            return
        for key, val in self._options.items():
            add_argument(parser, key, val)

    def make_base_config(self, args):
        """Hook to build a baseline job configuration

        Parameters
        ----------

        args : dict
            Command line arguments, see add_arguments
        """
        self.link.update_args(args)
        return self.link.args

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
        raise NotImplementedError(
            "ScatterGather.ConfigMaker.build_job_configs")


class ScatterGather(Link):
    """ Abstract base class to dispatch several jobs in parallel and
    collect and merge the results.
    """
    default_init_logfile = 'init.log'
    default_prefix_logfile = 'scatter'
    default_gather_logfile = 'gather.log'

    default_options = dict(action=('run', 'Action to perform', str),
                           dry_run=(
                               False, 'Print commands, but do not execute them', bool),
                           job_check_sleep=(
                               300, 'Sleep time between checking on job status (s)', int),
                           force_gather=(False, 'Force gathering stage', bool),
                           print_update=(
                               False, 'Print summary of job status', bool),
                           check_status_once=(False,
                                              'Check status only once before proceeding', bool),)

    def __init__(self, **kwargs):
        """C'tor

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

        job_archive : `fermipy.job_archive.JobArchive` [optional]
            Archive used to track jobs and associated data products
            Defaults to None

        initialize : `fermipy.chain.Link` [optional]
            Link run for the initialization stage
            Defaults to None

        scatter : `fermipy.chain.Link`
            Link run for the scatter stage
            Defaults to None

        gather : `fermipy.chain.Link` [optional]
            Link run for the gather stage
            Defaults to None

        no_batch : bool
            If true, do not send jobs to the batch to run
            Defaults to False
        """
        linkname = kwargs.pop('linkname', 'ScatterGather')
        self._config_maker = kwargs.pop('config_maker', None)
        self._usage = kwargs.pop('usage', "")
        self._description = kwargs.pop('description', "")
        self._job_archive = kwargs.pop('job_archive', None)
        self._initialize_link = kwargs.pop('initialize', None)
        self._scatter_link = kwargs.pop('scatter', None)
        self._gather_link = kwargs.pop('gather', None)
        self._no_batch = kwargs.pop('no_batch', False)
        options = kwargs.get('options', self.default_options.copy())
        self._config_maker.add_options(options)
        Link.__init__(self, linkname,
                      options=options,
                      parser=self._make_parser(),
                      **kwargs)
        self._base_config = None
        self._input_config = {}
        self._job_configs = {}
        self._output_config = {}

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
    def no_batch(self):
        """Return the value of the no_batch flag"""
        return self._no_batch

    @staticmethod
    def _make_init_logfile_name(input_config):
        """ Hook to inster the name of a logfile into the input config """
        logfile = input_config.get(
            'logfile', ScatterGather.default_init_logfile)
        input_config['logfile'] = logfile

    @staticmethod
    def _make_scatter_logfile_name(key, job_config):
        """ Hook to inster the name of a logfile into the input config """
        logfile = job_config.get('logfile', "%s_%s.log" %
                                 (ScatterGather.default_prefix_logfile, key))
        job_config['logfile'] = logfile

    @staticmethod
    def _make_gather_logfile_name(output_config):
        """ Hook to construct the name of a logfile the gatherer """
        logfile = output_config.get(
            'logfile', ScatterGather.default_gather_logfile)
        output_config['logfile'] = logfile

    def check_job(self, job_details):
        """ Check the status of a specfic job """
        raise NotImplementedError('ScatterGather.check_job')

    def dispatch_job_hook(self, link, key, job_config, logfile):
        """Hook to dispatch a single job"""
        raise NotImplementedError("ScatterGather.dispatch_job_hook")

    def update_args(self, override_args):
        """Update the arguments used to invoke the application

        Note that this will also update the dictionary of input and output files

        Parameters
        ----------

        override_args : dict
            dictionary of arguments to override the current values
        """
        self.args = extract_arguments(override_args, self.args, self.mapping)
        self.build_configs(self.args)
        if self._initialize_link is not None:
            self._initialize_link.update_args(self._input_config)
        if self._gather_link is not None:
            self._gather_link.update_args(self._output_config)
        if len(self._scatter_link.jobs) == 0:
            self.build_job_dict()
        self._latch_file_info()

    def get_jobs(self, recursive=True):
        """Return a dictionary with all the jobs

        If recursive is True this will include jobs from all internal `Link`
        """
        if recursive:
            ret_dict = self.jobs.copy()
            if self._initialize_link is not None:
                ret_dict.update(self._initialize_link.get_jobs(recursive))
            ret_dict.update(self._scatter_link.get_jobs(recursive))
            if self._gather_link is not None:
                ret_dict.update(self._gather_link.get_jobs(recursive))
            return ret_dict
        else:
            return self.jobs

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
            if self._initialize_link is not None:
                stream.write("\n")
                self._initialize_link.print_summary(
                    stream, indent, recurse_level)
            stream.write("\n")
            self._scatter_link.print_summary(stream, indent, recurse_level)
            if self._gather_link is not None:
                stream.write("\n")
                self._gather_link.print_summary(stream, indent, recurse_level)

    def _latch_file_info(self):
        """Internal function to update the dictionaries
        keeping track of input and output files
        """
        self.files.file_dict.clear()
        self.sub_files.file_dict.clear()
        self.files.latch_file_info(self.args)

        if self._initialize_link is not None:
            self.sub_files.update(self._initialize_link.files.file_dict)
            self.sub_files.update(self._initialize_link.sub_files.file_dict)
        self._scatter_link.update_sub_file_dict(self.sub_files)
        if self._gather_link is not None:
            self.sub_files.update(self._gather_link.files.file_dict)
            self.sub_files.update(self._gather_link.sub_files.file_dict)

    def check_status(self):
        """Check on the status of all the jobs in job dict.

        Returns
        -------

        running  : bool
            True if jobs are still running
        failed   : bool
            True if any jobs have failed
        """
        running = True
        first = True
        while running:
            if first:
                first = False
            elif self.args['dry_run']:
                break
            else:
                print ("Sleeping %.0f seconds between status checks" %
                       self.args['job_check_sleep'])
                time.sleep(self.args['job_check_sleep'])

            running, failed = self._check_completion()
            if self.args['force_gather']:
                sys.stdout.write('Forcing results gathering\n')
                running = False
            elif self.args['check_status_once']:
                self.print_update()
                break
            if self.args['print_update']:
                self.print_update()

            if self._job_archive is not None:
                self._job_archive.write_table_file()

        if failed:
            self.print_update()
            self.print_failed()

        return running, failed

    def _check_completion(self):
        """Internal function to check the completion of all the dispatched jobs

        Returns
        -------

        running : bool
            True if the jobs are still running

        failed : bool
            True if any jobs have failed
        """
        if self._initialize_link is not None:
            running, failed = self._check_link_completion(
                self._initialize_link)
            if running or failed:
                return running, failed
        running, failed = self._check_link_completion(self._scatter_link)
        if running or failed:
            return running, failed
        if self._gather_link is not None:
            running, failed = self._check_link_completion(self._gather_link)
        return running, failed

    def _check_link_completion(self, link):
        """Internal function to check the completion of all the dispatched jobs

        Returns
        -------

        running : bool
            True if the jobs are still running

        failed : bool
            True if any jobs have failed
        """
        failed = False
        running = False
        for job_key, job_details in link.jobs.items():
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
            link.jobs[job_key] = job_details

        return running, failed

    def _merge_config(self, config_in):
        """Merge a configuration with the baseline, return the merged configuration """
        config_out = self._base_config.copy()
        config_out.update(config_in)
        return config_out

    def build_configs(self, args):
        """Build the configuration objects.

        This invokes the `ConfigMaker` to build the configurations
        """
        self._base_config = self._config_maker.make_base_config(args)
        self._input_config, self._job_configs, self._output_config =\
            self._config_maker.build_job_configs(args)

    def initialize(self):
        """Run the initialization `Link` using self._input_config as input.

        Returns a `JobStatus` enum
        """
        if self._initialize_link is None:
            return JobStatus.no_job
        job_details = self.dispatch_job(
            self._initialize_link, key=self.linkname + '.init')
        return job_details.status

    def submit_jobs(self, link, job_dict=None):
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
                clean_job(job_details.logfile, job_details.outfiles,
                          self.args['dry_run'])
            job_config['logfile'] = job_details.logfile
            new_job_details = self.dispatch_job(self._scatter_link, job_key)
            if new_job_details.status == JobStatus.failed:
                failed = True
                clean_job(new_job_details.logfile,
                          new_job_details.outfiles, self.args['dry_run'])
            link.jobs[job_key] = new_job_details
        if failed:
            return JobStatus.failed
        return JobStatus.done

    def gather_results(self):
        """Run the gather `Link` using self._output_config as input

        Returns a `JobStatus` enum
        """
        if self._gather_link is None:
            return JobStatus.no_job
        job_details = self.dispatch_job(
            self._gather_link, key='gather@' + self._gather_link.linkname)
        return job_details.status

    def dispatch_job(self, link, key):
        """Function to dispatch a single job

        Parameters
        ----------

        link : `Link`
            Link object that sendes the job

        key : str
            Key used to identify this particular job

        Returns `JobDetails` object
        """
        try:
            job_details = link.jobs[key]
        except KeyError:
            print (key, link.jobs)
        job_config = job_details.job_config
        link.update_args(job_config)
        logfile = job_config['logfile']
        try:
            self.dispatch_job_hook(link, key, job_config, logfile)
            job_details.status = JobStatus.running
        except IOError:
            job_details.status = JobStatus.failed

        if self._job_archive is not None:
            self._job_archive.register_job(job_details)
        return job_details

    def _make_parser(self):
        """Make an argument parser for this chain """
        parser = argparse.ArgumentParser(usage=self._usage,
                                         description=self._description)
        return parser

    def print_update(self, stream=sys.stdout):
        """Print an update about the current number of jobs running """
        n_total = len(self._job_configs)
        n_running = 0
        n_done = 0
        n_failed = 0
        n_pending = 0
        job_det_list = []
        if self._initialize_link is not None:
            job_det_list += self._initialize_link.jobs.values()
        job_det_list += self._scatter_link.jobs.values()
        if self._gather_link is not None:
            job_det_list += self._gather_link.jobs.values()

        for job_dets in job_det_list:
            if job_dets.status == JobStatus.running:
                n_running += 1
            elif job_dets.status == JobStatus.done:
                n_done += 1
            elif job_dets.status == JobStatus.failed:
                n_failed += 1
            elif job_dets.status == JobStatus.pending:
                n_pending += 1

        stream.write("Status :\n  Total  : %i\n  Pending: %i\n" %
                     (n_total, n_pending))
        stream.write("Running: %i\n  Done   : %i\n  Failed : %i\n" %
                     (n_running, n_done, n_failed))

    def print_failed(self, stream=sys.stderr):
        """Print list of the failed jobs """
        for job_key, job_details in sorted(self._job_configs.items()):
            if job_details['status'] == JobStatus.failed:
                stream.write("Failed job %s : log = %s\n" %
                             (job_key, job_details.logfile))

    def __call__(self, argv):
        """Parses command line arguments and runs the requested action

        Parameters
        ----------

        argv : list-like
            List of command line arguments

        Returns str with status
        """
        return self.invoke(argv[1:])

    def invoke(self, argv):
        """Invoke this object to preform a particular action

        Parameters
        ----------

        argv : list-like
            List of command line arguments, passed to helper classes

        Returns str with status
        """
        args = self.run_argparser(argv)

        if args.action not in ACTIONS:
            sys.stderr.write(
                "Unrecognized action %s, options are %s\n" % (args.action, ACTIONS))

        if args.action in ['run', 'resubmit', 'init',
                           'scatter', 'gather', 'check_status', 'config']:
            self.build_configs(args.__dict__)

        if args.action in ['init', 'scatter', 'gather', 'check_status']:
            # This is called explicitly in run and resubmit
            self.build_job_dict()

        if args.action == 'run':
            return self.run_jobs()
        elif args.action == 'resubmit':
            return self.resubmit()
        elif args.action == 'init':
            return self.initialize()
        elif args.action == 'scatter':
            return self.submit_jobs(self.scatter_link)
        elif args.action == 'gather':
            return self.gather_results()
        elif args.action == 'check_status':
            running, failed = self.check_status()
            if failed:
                return JobStatus.failed
            elif running:
                return JobStatus.running
            elif self.args['dry_run']:
                return JobStatus.no_job
            return JobStatus.done
        elif args.action == 'config':
            return JobStatus.done

    def build_job_dict(self):
        """Build a dictionary of `JobDetails` objects for the internal `Link`"""
        if self.args['dry_run']:
            status = JobStatus.no_job
        else:
            status = JobStatus.pending

        if self._initialize_link is not None:
            full_init_config = self._input_config.copy()
            ScatterGather._make_init_logfile_name(full_init_config)
            logfile = full_init_config.get('logfile')
            self._initialize_link.register_job(key='init',
                                               job_config=full_init_config,
                                               logfile=logfile,
                                               status=status)

        for jobkey, job_config in sorted(self._job_configs.items()):
            full_job_config = self._merge_config(job_config)
            ScatterGather._make_scatter_logfile_name(jobkey, full_job_config)
            logfile = full_job_config.get('logfile')
            self._scatter_link.register_job(key=jobkey,
                                            job_config=full_job_config,
                                            logfile=logfile,
                                            status=status)

        if self._gather_link is not None:
            full_gather_config = self._output_config.copy()
            ScatterGather._make_gather_logfile_name(full_gather_config)
            logfile = full_gather_config.get('logfile')
            self._gather_link.register_job(key='gather',
                                           job_config=full_gather_config,
                                           logfile=logfile,
                                           status=status)

    def run_jobs(self):
        """Function to dipatch jobs and collect results
        """
        init_status = self.initialize()
        if init_status == JobStatus.failed:
            return JobStatus.failed

        self.build_job_dict()

        scatter_status = self.submit_jobs(self.scatter_link)
        if scatter_status == JobStatus.failed:
            return JobStatus.failed

        running, failed = self.check_status()
        if self.args['force_gather'] or (running is False):
            gather_status = self.gather_results()
            return gather_status

        if self.args['dry_run']:
            gather_status = self.gather_results()
            return JobStatus.no_job

        if failed:
            return JobStatus.failed
        else:
            return JobStatus.done

    def run(self, stream=sys.stdout, dry_run=False):
        argv = self.make_argv()
        if dry_run:
            argv.append('--dry_run')
        self.invoke(argv)

    def resubmit(self):
        """Function to resubmit failed jobs and collect results
        """
        self.build_job_dict()
        running, failed = self.check_status()
        if failed is False:
            return JobStatus.done

        if self._initialize_link is not None:
            failed_jobs = self._initialize_link.get_failed_jobs()
            if len(failed_jobs) != 0:
                init_status = self.initialize()
                if init_status == JobStatus.failed:
                    return JobStatus.failed

        failed_jobs = self._scatter_link.get_failed_jobs()
        if len(failed_jobs) != 0:
            scatter_status = self.submit_jobs(failed_jobs)
            if scatter_status == JobStatus.failed:
                return JobStatus.failed

        running, failed = self.check_status()
        if self.args['force_gather'] or (running is False):
            gather_status = self.gather_results()
            return gather_status

        if self.args['dry_run']:
            return JobStatus.no_job
        return JobStatus.done

    def run_argparser(self, argv):
        """Initialize self with a set of arguments
        """
        if self._parser is None:
            raise ValueError('Link was not given a parser on initialization')
        args = self._parser.parse_args(argv)
        self.update_args(args.__dict__)

        return args
