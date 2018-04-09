# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import sys
import time
import os
import stat
import datetime
import argparse
import pprint
from fermipy import utils
from fermipy.batch import check_log, get_lsf_status


def file_age_in_seconds(pathname):
    return time.time() - os.stat(pathname)[stat.ST_MTIME]


def collect_jobs(dirs, runscript, overwrite=False, max_job_age=90):
    """Construct a list of job dictionaries."""

    jobs = []

    for dirname in sorted(dirs):

        o = dict(cfgfile=os.path.join(dirname, 'config.yaml'),
                 logfile=os.path.join(
                     dirname, os.path.splitext(runscript)[0] + '.log'),
                 runscript=os.path.join(dirname, runscript))

        if not os.path.isfile(o['cfgfile']):
            continue

        if not os.path.isfile(o['runscript']):
            continue

        if not os.path.isfile(o['logfile']):
            jobs.append(o)
            continue

        age = file_age_in_seconds(o['logfile']) / 60.
        job_status = check_log(o['logfile'])

        print(dirname, job_status, age)

        if job_status is False or overwrite:
            jobs.append(o)
        elif job_status == 'Exited':
            print("Job Exited. Resending command.")
            jobs.append(o)
        elif job_status == 'None' and age > max_job_age:
            print(
                "Job did not exit, but no activity on log file for > %.2f min. Resending command." % max_job_age)
            jobs.append(o)
        #        elif job_status is True:
        #            print("Job Completed. Resending command.")
        #            jobs.append(o)

    return jobs


def main():
    usage = "usage: %(prog)s [config file]"
    description = "Dispatch analysis jobs to LSF."
    parser = argparse.ArgumentParser(usage=usage, description=description)

    parser.add_argument('--config', default='sample_config.yaml')
    parser.add_argument('--resources', default=None, type=str,
                        help='Set the LSF resource string.')
    parser.add_argument('--time', default=1500, type=int,
                        help='Set the wallclock time allocation for the '
                             'job in minutes.')
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
    parser.add_argument('--dry_run', default=False, action='store_true')
    parser.add_argument('--overwrite', default=False, action='store_true',
                        help='Force all jobs to be re-run even if the job has '
                             'completed successfully.')
    parser.add_argument('--runscript', default=None, required=True,
                        help='Set the name of the job execution script.  A '
                             'script with this name must be located in each '
                             'analysis subdirectory.')
    parser.add_argument('--ncpu', default=1, type=int,
                        help='Set the number of CPUs that are used for each job.')
    parser.add_argument('dirs', nargs='+', default=None,
                        help='List of directories in which the analysis will '
                             'be run.')

    args = parser.parse_args()

    dirs = [d for argdir in args.dirs for d in utils.collect_dirs(argdir)]
    jobs = collect_jobs(dirs, args.runscript,
                        args.overwrite, args.max_job_age)

    lsf_opts = {'W': args.time,
                'n': args.ncpu,
                'R': 'bullet,hequ,kiso && scratch > 5'}

    if args.resources is not None:
        lsf_opts['R'] = args.resources

    lsf_opt_string = ''
    for optname, optval in lsf_opts.items():

        if utils.isstr(optval):
            optval = '\"%s\"' % optval

        lsf_opt_string += '-%s %s ' % (optname, optval)

    while (1):

        print('-' * 80)
        print(datetime.datetime.now())
        print(len(jobs), 'jobs in queue')

        if len(jobs) == 0:
            break

        status = get_lsf_status()

        njob_to_submit = min(args.max_jobs - status['NJOB'],
                             args.jobs_per_cycle)

        pprint.pprint(status)
        print('njob_to_submit ', njob_to_submit)

        if njob_to_submit > 0:

            print('Submitting ', njob_to_submit, 'jobs')

            for job in jobs[:njob_to_submit]:
                cmd = 'bsub %s -oo %s bash %s' % (lsf_opt_string,
                                                  job['logfile'],
                                                  job['runscript'])
                print(cmd)
                if not args.dry_run:
                    print('submitting')
                    os.system(cmd)

            del jobs[:njob_to_submit]

        print('Sleeping %f seconds' % args.time_per_cycle)
        sys.stdout.flush()
        time.sleep(args.time_per_cycle)


if __name__ == "__main__":
    main()
