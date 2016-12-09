# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Abstract interface for parallel execution of multiple jobs
"""
from __future__ import absolute_import, division, print_function


import sys
import os
import time
import copy

import argparse

ACTIONS = ['run', 'resubmit', 'init',
           'scatter', 'gather', 'check_status', 'config']


def remove_file(filepath, dry_run=False):
    """ Remove a file

    Catches exception if the file does not exist.
    """
    if dry_run:
        sys.stdout.write("rm %s\n" % filepath)
    else:
        try:
            os.remove(filepath)
        except OSError:
            pass


class JobDetails(object):
    """ This is just a simple structure to keep track of the details of each
        of the sub-proccess jobs
    """

    def __init__(self, jobname, logfile, job_config, status):
        """ C'tor

        Parameters:
        ---------------
        jobname : str
            The key used to identify the job

        logfile : str
            The logfile for this job, may be used to check for success/ failure

        job_config : dict
            A dictionrary with the arguments for the job

        status : str
            Current job status, one of 'Pending', 'Running', 'Done', 'Failed'
        """
        self.jobname = jobname
        self.logfile = logfile
        self.job_config = job_config
        self.status = status


class Checker(object):
    """ Abstract helper class to check on status of indiviudal jobs
    """

    def __init__(self, **kwargs):
        """ C'tor
        """
        self._actions = copy.deepcopy(ACTIONS)

    def check_single_job(self, job_details, args):
        """ Check on the status of all single job """
        job_dict = {None: job_details}
        return self.check_status(job_dict, args)

    def check_status(self, job_dict, args):
        """ Check on the status of all the jobs in job dict """
        running = True
        first = True
        while running:
            if first:
                first = False
            else:
                print ("Sleeping %.0f seconds between status checks" %
                       args.job_check_sleep)
                time.sleep(args.job_check_sleep)

            job_dict, running, failed = self.check_completion(job_dict)
            if args.force_gather:
                sys.stdout.write('Forcing results gathering\n')
                running = False
            elif args.check_status_once:
                Checker.print_update(job_dict)
                break
            if args.print_update:
                Checker.print_update(job_dict)

        if failed:
            Checker.print_update(job_dict)
            Checker.print_failed(job_dict)

        return job_dict, running, failed

    def check_completion(self, job_dict):
        """ Internal function to check the completion of all the dispatched jobs

        Parameters:
        ---------------
        job_dict : dict
            Dictionary of JobDetails objects for all the jobs

        Returns str, either 'Failed' or 'Done'
        ---------------
        status : str
            One of 'Running', 'Failed', 'Done'

        job_dict : dict
            Updated version of JobDetails dictionary
        """
        failed = False
        running = False
        for job_key, job_details in job_dict.items():
            if job_details.status == 'Failed':
                failed = True
                continue
            elif job_details.status == 'Done':
                continue
            job_details.status = self.check_job(job_key, job_details)
            if job_details.status == 'Failed':
                failed = True
            elif job_details.status in ['Running', 'Pending', 'Unknown']:
                running = True
            job_dict[job_key] = job_details

        return job_dict, running, failed

    def supports_action(self, action):
        """ Check that this checker supports a particular action """
        if action not in self._actions:
            raise ValueError("Checker does not support action %s"%action)

    def add_arguments(self, parser, action):
        """ Hook to add arguments to the command line argparser """
        self.supports_action(action)
        try:
            parser.add_argument('--job_check_sleep', type=int, default=300)
            parser.add_argument('--force_gather', action='store_true',
                                default=False)
            parser.add_argument('--print_update', action='store_true',
                                default=False)
            parser.add_argument('--check_status_once', action='store_true',
                                default=False)
        except argparse.ArgumentError:
            # Already added
            pass


    def check_job(self, key, job_details):
        """ Check the status of a specfic job """
        # This is just to get pylint to shutup about unused variables.
        if None not in [self, key, job_details]:
            return "Done"

    @staticmethod
    def print_update(job_dict, stream=sys.stdout):
        """ Print an update about the current number of jobs running """
        n_total = len(job_dict)
        n_running = 0
        n_done = 0
        n_failed = 0
        n_pending = 0
        for job_dets in job_dict.values():
            if job_dets.status == 'Running':
                n_running += 1
            elif job_dets.status == 'Done':
                n_done += 1
            elif job_dets.status == 'Failed':
                n_failed += 1
            elif job_dets.status == 'Pending':
                n_pending += 1

        stream.write("Status :\n  Total  : %i\n  Pending: %i\n"%(n_total, n_pending))
        stream.write("Running: %i\n  Done   : %i\n  Failed : %i\n" %(n_running, n_done, n_failed))

    @staticmethod
    def print_failed(job_dict, stream=sys.stderr):
        """ Print list of the failed jobs """
        job_keys = job_dict.keys()
        job_keys.sort()
        for job_key in job_keys:
            job_details = job_dict[job_key]
            if job_details.status == 'Failed':
                stream.write("Failed job %s : log = %s\n" %
                             (job_key, job_details.logfile))

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
        raise NotImplementedError(
            "ScatterGather.ConfigMaker.make_base_config")  

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


class Initializer(object):
    """ Abstract helper class to perform initialization functions
    """

    def __init__(self, checker=None):
        """ C'tor
        """
        self._checker = checker
        self.default_logfile = 'init.log'

    def initialize(self, input_config, job_configs, args):
        """ Prepare to run the jobs """
        dry_run = args.dry_run
        logfile = self.make_init_logfile_name(input_config)
        try:
            job_details = self.initialize_hook(input_config, job_configs,
                                               logfile, dry_run)
            if self._checker is not None and job_details is not None:
                status = self._checker.check_single_job(job_details, args)
            else:
                status = "Done"
        except IOError:
            self.clean_initialize(input_config, job_configs, logfile, dry_run)
            return "Failed"
        return status

    def add_arguments(self, parser, action):
        """ Hook to add arguments to the command link argparser """
        if self._checker:
            self._checker.add_arguments(parser, action)

    def make_init_logfile_name(self, input_config):
        """ Hook to construct the name of a logfile the gatherer """
        return input_config.get('logfile', self.default_logfile)

    def initialize_hook(self, input_config, job_configs, logfile, dry_run):
        """ Hook to implement things that happen before jobs get dispathced """
        # This is just to get pylint to shutup about unused variables.
        return None in [self, input_config, job_configs, logfile, dry_run]

    def clean_initialize(self, input_config, job_configs, logfile, dry_run):
        """ Hook to clean things up in case initilization fails """
        # This is just to get pylint to shutup about unused variables.
        return None in [self, input_config, job_configs, logfile, dry_run]


class Dispatcher(object):
    """ Abstract helper class to dispatch indvidiual jobs
    """

    def __init__(self, **kwargs):
        """ C'tor
        """
        self.prefix = "scatter"

    def submit_jobs(self, job_dict, args):
        """ Submit all the jobs in job_dict """
        dry_run = args.dry_run
        job_keys = job_dict.keys()
        job_keys.sort()
        failed = False
        for job_key in job_keys:
            job_details = job_dict[job_key]
            job_config = job_details.job_config
            if job_details.status == 'Failed':
                outfiles = self.make_job_outfile_names(job_key, job_config)
                for outfile in outfiles:
                    remove_file(outfile, dry_run)
                remove_file(job_details.logfile, dry_run)
            job_dict[job_key] = self.dispatch_job(job_key, job_config, dry_run)
            if job_dict[job_key].status == "Failed":
                failed = True
        return job_dict, failed

    def dispatch_job(self, key, job_config, dry_run=False):
        """ Function to dispatch a single job

        Parameters:
        ---------------
        key : str
            Key used to identify this particular job

        job_config : dict
            Dictionary with arguements passed to this particular job

        dry_run : bool [False]
            Print batch commands, but do not submit jobs

        Returns `JobDetails' object
        """
        logfile = self.make_job_logfile_name(key, job_config)
        outfiles = self.make_job_outfile_names(key, job_config)
        try:
            self.dispatch_job_hook(key, job_config, logfile, dry_run)
        except IOError:
            self.clean_dispatch(key, logfile, outfiles, dry_run)
            return JobDetails(key, logfile, job_config, 'Failed')

        return JobDetails(key, logfile, job_config, 'Running')

    def add_arguments(self, parser, action):
        """ Hook to add arguments to the command link argparser """
        # This is just to get pylint to shutup about unused variables.
        return None in [self, parser, action]

    def make_job_logfile_name(self, key, job_config):
        """ Hook to construct the name of a logfile for a particular job """
        return job_config.get('logfile', '%s_%s.log' % (self.prefix, key))

    def make_job_outfile_names(self, key, job_config):
        """ Hook to construct the names of the output files for a particular job """
        # This is just to get pylint to shutup about unused variables.
        if not None in [self, key, job_config]:
            return job_config.get('outfiles', [])

    def dispatch_job_hook(self, key, job_config, logfile, dry_run):
        """ Hook to dispatch a single job """
        raise NotImplementedError("ScatterGather.dispatch_job_hook")

    def clean_dispatch(self, key, logfile, outfiles, dry_run):
        """ Hook to clean things up if a dispatching a job fails """
        return None in [self, key, logfile, outfiles, dry_run]


class Gatherer(object):
    """ Abstract helper class to gather results
    """

    def __init__(self, checker):
        """ C'tor, Takes a checker to test if jobs are successfuly completed
        """
        self._checker = checker
        self.default_logfile = 'gather.log'

    @property
    def checker(self):
        """ Return the object used the check for succesful job completion """
        return self._checker

    def gather_results(self, output_config, job_dict, args):
        """ Function to dispatch a single job

        Parameters:
        ---------------
        job_dict : dict
            Dictionary of JobDetails objects for all the jobs

        Returns str, either 'Failed' or 'Done'
        """
        dry_run = args.dry_run
        logfile = self.make_gather_logfile_name(output_config)
        try:
            self.gather_results_hook(output_config, job_dict, logfile, dry_run)
        except IOError:
            self.clean_gather(output_config, job_dict, logfile, dry_run)
            return "Failed"
        return "Done"

    def make_gather_logfile_name(self, output_config):
        """ Hook to construct the name of a logfile the gatherer """
        return output_config.get('logfile', self.default_logfile)

    def gather_results_hook(self, output_config, job_dict, logfile, dry_run):
        """ Hook to gather results """
        pass

    def clean_gather(self, output_config, job_config, logfile, dry_run):
        """ Hook to clean things up if gathering results fails """
        pass

    def add_arguments(self, parser, action):
        """ Hook to add arguments to the command link argparser """
        pass


class ScatterGather(object):
    """ Abstract base class to dispatch several jobs in parallel and
    collect and merge the results
    """

    def __init__(self, **kwargs):
        """ C'tor

        Keyword arguements
        ---------------
        config_maker : `ConfigMaker'
            Object used to translate arguments
            Must have functions 'add_arguments' and 'build_job_configs'
            Defaults to ConfigMaker()

        initializer : `Initializer'
            Object used to prepare inputs,
            Must have functions 'initialize_hook' and 'clean_initialize'
            Defaults to Initializer()

        dispatcher : `Dispatcher'
            Object used to dispatch individual jobs,
            Must have functions 'dispatch_job_hook', 'check_job' and 'clean_dispatch'
            Defaults to Dispather()

        gatherer : `Gatherer'
            Object used to gather results
            Must have functions 'gather_results_hook', 'check_job'and 'clean_gather'
            Defaults to Gatherer()

        usage : str
            Usage string for argument parser

        description : str
            Description string for argument parser
        """
        self._base_config = None
        self._actionlist = copy.deepcopy(ACTIONS)
        self._config_maker = kwargs.pop('config_maker', ConfigMaker())
        self._initializer = kwargs.pop('initializer', Initializer())
        self._dispatcher = kwargs.pop('dispatcher', Dispatcher())
        self._gatherer = kwargs.pop('gatherer', Gatherer(Checker()))
        self._usage = kwargs.pop('usage', "")
        self._description = kwargs.pop('description', "")
        self._parser = None
        self.input_config = {}
        self.job_configs = {}
        self.output_config = {}

    @property
    def base_config(self):
        """ Return the baseline job configuration """
        return self._base_config

    @property
    def config_maker(self):
        """ Return the object used to translate arguments """
        return self._config_maker

    @property
    def initializer(self):
        """ Return the object used prepare for the run """
        return self._initializer

    @property
    def dispatcher(self):
        """ Return the object used to dispatch individual jobs """
        return self._dispatcher

    @property
    def gatherer(self):
        """ Return the object used to gather the individual jobs """
        return self._gatherer

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
        if action not in self._actionlist:
            sys.stderr.write("Unrecognized action %s, options are %s\n" % (
                action, self._actionlist))
            sys.stderr.write(
                "  Try: <command> <action> -h for specific help\n")

        args = self.parse_arguments(action, argv)

        if action in ['run', 'resubmit', 'init', 'scatter', 'gather', 'check_status', 'config']:
            self.build_configs(args)

        if action in ['scatter', 'gather', 'check_status']:
            job_dict = self.build_job_dict()

        if action == 'run':
            return self.run(args)
        elif action == 'resubmit':
            return self.resubmit(args)
        elif action == 'init':
            return self._initializer.initialize(self.input_config, self.job_configs, args)
        elif action == 'scatter':
            return self._dispatcher.submit_jobs(job_dict, args)
        elif action == 'gather':
            return self._gatherer.gather_results(self.output_config, job_dict, args)
        elif action == 'check_status':
            return self._gatherer.checker.check_status(job_dict, args)
        elif action == 'config':
            return 'Done'

    def parse_arguments(self, action, argv):
        """ Build and fill an ArgumentParser object """
        self._parser = argparse.ArgumentParser(usage=self._usage,
                                               description=self._description)
        self._parser.add_argument('--dry_run', action='store_true', default=False,
                                  help='Print commands, but do not execute them')

        if action == 'run':
            self._initializer.add_arguments(self._parser, action)
            self._gatherer.add_arguments(self._parser, action)
            self._gatherer.checker.add_arguments(self._parser, action)
            self._dispatcher.add_arguments(self._parser, action)
        elif action == 'resubmit':
            self._gatherer.add_arguments(self._parser, action)
            self._gatherer.checker.add_arguments(self._parser, action)
            self._dispatcher.add_arguments(self._parser, action)
        elif action == 'initialize':
            self._initializer.add_arguments(self._parser, action)
        elif action == 'scatter':
            self._dispatcher.add_arguments(self._parser, action)
        elif action == 'gather':
            self._gatherer.add_arguments(self._parser, action)
            self._gatherer.checker.add_arguments(self._parser, action)
        elif action == 'check_status':
            self._gatherer.checker.add_arguments(self._parser, action)

        if action in ['run', 'resubmit', 'init', 'scatter', 'gather', 'check_status', 'config']:
            print ('config_maker.add_arguments')
            self._config_maker.add_arguments(self._parser, action)

        return self._parser.parse_args(argv)

    def merge_config(self, config_in):
        """ Merge a configuration with the baseline, return the merged configuration """
        config_out = self._base_config.copy()
        config_out.update(config_in)
        return config_out

    def build_configs(self, args):
        """ Build the configuration objects """
        self._base_config = self._config_maker.make_base_config(args)
        self.input_config, self.job_configs, self.output_config =\
            self._config_maker.build_job_configs(args)

    def build_job_dict(self):
        """ Build a dictionary of JobDetails objects """
        job_keys = self.job_configs.keys()
        job_keys.sort()

        job_dict = {}
        for key in job_keys:
            job_config = self.merge_config(self.job_configs[key])
            logfile = self._dispatcher.make_job_logfile_name(key, job_config)
            job_dict[key] = JobDetails(key, logfile, job_config, 'Unknown')

        return job_dict

    def run(self, args):
        """ Function to dipatch jobs and collect results
        """
        self._initializer.initialize(self.input_config, self.job_configs, args)

        job_dict = self.build_job_dict()
        job_dict, failed = self._dispatcher.submit_jobs(job_dict, args)
        job_dict, running, failed = self._gatherer.checker.check_status(
            job_dict, args)
        if args.force_gather or (running is False):
            gathered = self._gatherer.gather_results(
                self.output_config, job_dict, args)

        return job_dict, running, failed, gathered

    def resubmit(self, args):
        """ Function to dipatch jobs and collect results
        """
        job_dict = self.build_job_dict()
        job_dict, running, failed = self._gatherer.checker.check_status(
            job_dict, args)

        failed_jobs = {}
        job_keys = job_dict.keys()
        job_keys.sort()
        failed = False
        for job_key in job_keys:
            job_details = job_dict[job_key]
            if job_details.status in ['Failed', 'Pending']:
                failed_jobs[job_key] = job_details

        failed_jobs, failed = self._dispatcher.submit_jobs(failed_jobs, args)
        job_dict, running, failed = self._gatherer.checker.check_status(
            job_dict, args)
        if args.force_gather or (running is False):
            gathered = self._gatherer.gather_results(
                self.output_config, job_dict, args)

        return job_dict, running, failed, gathered
