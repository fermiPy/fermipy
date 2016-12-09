# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Implementation of interface for dealing with LSF batch jobs
"""
from __future__ import absolute_import, division, print_function

import sys
import os
import time
import subprocess

from fermipy.jobs.scatter_gather import remove_file, JobDetails, Checker,\
    Initializer, Dispatcher, Gatherer, ScatterGather



def get_lsf_status():
    """ Count and print the number of jobs in various LSF states
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
    """ Check a log file to determine status of LSF job

    Often logfile doesn't exist because the job hasn't begun
    to run. It is unclear what you want to do in that case...

    Parameters:
    ---------------
    logfile : str
        String with path to logfile

    exited  : str
        Value to check for in existing logfile for exit with failure

    successful : str
        Value to check for in existing logfile for success

    Returns str, one of 'Pending', 'Running', 'Done', 'Failed'
    """
    if not os.path.exists(logfile):
        return 'Pending'

    if exited in open(logfile).read():
        return 'Failed'
    elif successful in open(logfile).read():
        return 'Done'
    else:
        return 'Running'


def build_bsub_command(command_template, lsf_args):
    """ Build and return a lsf batch command template

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


class LsfJobChecker(Checker):
    """ Class to check status of LSF jobs
    """

    def __init__(self, **kwargs):
        """ C'tor

        Keyword arguements
        ---------------
        lsf_exited : str ['Exited with exit code']
            String used to identify failed jobs

        lsf_successful : str ['Successfully completed']
            String used to identify completed jobs
        """
        super(LsfJobChecker, self).__init__(**kwargs)
        self._exited = kwargs.pop('lsf_exited', 'Exited with exit code')
        self._successful = kwargs.pop(
            'lsf_successful', 'Successfully completed')

    def check_job(self, key, job_details):
        """ Check the status of a single job

        Returns str, one of 'Pending', 'Running', 'Done', 'Failed'
        """
        return check_log(job_details.logfile, self._exited, self._successful)


class LsfDispatcher(object):
    """ Class to dispatch jobs to the LSF batch
    """

    def __init__(self, command_template, **kwargs):
        """ C'tor

        Parameters:
        ---------------
        command_template : str
            String used as input to format indvidual job command lines

        Keyword arguements
        ---------------
        lsf_args : dict
            Dictionary with options to pass to bsub command
        """
        self._command_template = command_template
        self._lsf_args = kwargs.pop('lsf_args', {})
        self._base_command = build_bsub_command(
            self._command_template, self._lsf_args)

    def run(self, job_config, logfile, dry_run=False):
        """ Send a single job to the LSF batch

        Parameters:
        ---------------
        job_config : dict
            A dictionrary with the arguments for the job.  Used with
            the self._command_template job template

        logfile : str
            The logfile for this job, may be used to check for success/ failure

        dry_run : bool [False]
            Print batch commands, but do not submit jobs
        """
        full_sub_dict = job_config.copy()
        full_sub_dict['logfile'] = logfile

        full_command = self._base_command.format(**full_sub_dict)
        if dry_run:
            sys.stdout.write("%s\n" % full_command)
        else:
            os.system(full_command)

    @staticmethod
    def clean(logfile, outfiles, dry_run=False):
        """ Removes log file and files created by failed jobs
        """
        remove_file(logfile, dry_run)
        for outfile in outfiles.values():
            remove_file(outfile, dry_run)


class LsfInitializer(LsfDispatcher, Initializer):
    """ Class to use LSF batch to perform initialization functions
    """

    def __init__(self, command_template, **kwargs):
        """ C'tor
        """
        LsfDispatcher.__init__(self, command_template, **kwargs)
        Initializer.__init__(self, checker=kwargs.pop(
            'checker', LsfJobChecker(**kwargs)))

    def initialize_hook(self, input_config, job_configs, logfile, dry_run):
        """ Hook to implement things that happen before jobs get dispathced """
        LsfDispatcher.run(self, input_config, logfile, dry_run)
        return JobDetails('Initialize', logfile, input_config, 'Pending')

    def clean_initialize(self, input_config, job_configs, logfile, dry_run):
        """ Hook to clean things up in case initilization fails """
        LsfDispatcher.clean(logfile, input_config.get('outfiles', []), dry_run)


class LsfScatterDispatcher(LsfDispatcher, Dispatcher):
    """ Class to dispatch jobs to the LSF batch
    """

    def __init__(self, command_template, **kwargs):
        """ C'tor

        Parameters:
        ---------------
        command_template : str
            String used as input to format indvidual job command lines

        Keyword arguments
        ---------------
        lsf_args : dict
            Dictionary with options to pass to bsub command
        """
        LsfDispatcher.__init__(self, command_template, **kwargs)
        Dispatcher.__init__(self, **kwargs)

    def submit_jobs(self, job_dict, args):
        """ Submit all the jobs in job_dict """
        dry_run = args.dry_run
        job_keys = job_dict.keys()

        job_keys.sort()
        failed = False

        # copy & reverse the keys b/c we will be popping item off the back of
        # the list
        unsubmitted_jobs = job_keys
        unsubmitted_jobs.reverse()

        while len(unsubmitted_jobs) > 0:
            status = get_lsf_status()

            njob_to_submit = min(args.max_jobs - status['NJOB'],
                                 args.jobs_per_cycle,
                                 len(unsubmitted_jobs))
            if dry_run:
                njob_to_submit = len(unsubmitted_jobs)

            for i in range(njob_to_submit):
                job_key = unsubmitted_jobs.pop()

                job_details = job_dict[job_key]
                job_config = job_details.job_config

                if job_details.status == 'Failed':
                    outfiles = self.make_job_outfile_names(job_key, job_config)
                    for outfile in outfiles:
                        remove_file(outfile, dry_run)
                    remove_file(job_details.logfile, dry_run)

                job_dict[job_key] = self.dispatch_job(
                    job_key, job_config, dry_run)
                if job_dict[job_key].status == "Failed":
                    failed = True

            print('Sleeping %.0f seconds between submission cycles' %
                  args.time_per_cycle)
            time.sleep(args.time_per_cycle)

        return job_dict, failed

    def dispatch_job_hook(self, key, job_config, logfile, dry_run=False):
        """ Send a single job to the LSF batch

        Parameters:
        ---------------
        key : str
            The key used to identify the job

        job_config : dict
            A dictionrary with the arguments for the job.  Used with
            the self._command_template job template

        logfile : str
            The logfile for this job, may be used to check for success/ failure

        dry_run : bool [False]
            Print batch commands, but do not submit jobs
        """
        LsfDispatcher.run(self, job_config, logfile, dry_run)

    def add_arguments(self, parser, action):
        """ Hook to add arguments to the command link argparser """
        parser.add_argument('--max_jobs', default=500, type=int,
                            help='Limit on the number of running or queued jobs.  '
                            'New jobs will only be dispatched if the number of '
                            'existing jobs is smaller than this parameter.')
        parser.add_argument('--jobs_per_cycle', default=20, type=int,
                            help='Maximum number of jobs to submit in each cycle.')
        parser.add_argument('--time_per_cycle', default=15, type=float,
                            help='Time per submission cycle in seconds.')
        parser.add_argument('--max_job_age', default=90, type=float,
                            help='Max job age in minutes.  Incomplete jobs without '
                            'a return code and a logfile modification '
                            'time older than this parameter will be restarted.')

    def clean_dispatch(self, key, logfile, outfiles, dry_run=False):
        """ Removes log file and files created by failed jobs
        """
        LsfDispatcher.clean(logfile, outfiles, dry_run)

    def make_job_logfile_name(self, key, job_config):
        """ Hook to construct the name of a logfile for a particular job """
        return job_config.get('logfile', 'scatter_%s.log' % key)

    def make_job_outfile_names(self, key, job_config):
        """ Hook to construct the names of the output files for a particular job """
        return job_config.get('outfiles', [])


class LsfGatherer(LsfDispatcher, Gatherer):
    """ Class to gather jobs from LSF batch
    """

    def __init__(self, command_template, **kwargs):
        """
        """
        LsfDispatcher.__init__(self, command_template, **kwargs)
        Gatherer.__init__(self, checker=kwargs.get(
            'checker', LsfJobChecker(**kwargs)))

    def gather_results_hook(self, output_config, job_dict, logfile, dry_run):
        """ Hook to dispatch a single job """
        LsfDispatcher.run(self, output_config, logfile, dry_run)
        return JobDetails('Gather', logfile, output_config, 'Pending')

    def clean_gather(self, output_config, job_config, logfile, dry_run):
        """ Hook to clean things up if gathering results fails """
        LsfDispatcher.clean(logfile,
                            output_config.get('outfiles', []), dry_run)


def build_sg_from_link(link, config_maker, **kwargs):
    """Build a `ScatterGather' that will run multiple instance of a single link
    """
    kwargs['config_maker'] = config_maker
    kwargs['dispatcher'] = LsfScatterDispatcher(link.command_template(),
                                                lsf_args=kwargs.pop('scatter_lsf_args', {}))
    gather_command_template = kwargs.pop('gather_command', None)
    kwargs['gatherer'] = LsfGatherer(gather_command_template,
                                     lsf_args=kwargs.pop('gather_lsf_args', {}))

    lsf_sg = ScatterGather(**kwargs)
    return lsf_sg


def build_sg_from_chain(chain, config_maker, **kwargs):
    """Build a `ScatterGather' that will run multiple instance of an analysis chain
    """
    kwargs['config_maker'] = config_maker
    pyfile = chain.filename.replace('.pyc', '.py')
    chain_command_template = chain.command_template("python %s" % pyfile)
    kwargs['dispatcher'] = LsfScatterDispatcher(chain_command_template,
                                                lsf_args=kwargs.pop('scatter_lsf_args', {}))
    gather_command_template = kwargs.pop('gather_command', None)
    kwargs['gatherer'] = LsfGatherer(gather_command_template,
                                     lsf_args=kwargs.pop('gather_lsf_args', {}))

    lsf_sg = ScatterGather(**kwargs)
    return lsf_sg
