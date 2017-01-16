# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import os
import subprocess

__all__ = [
    'check_log',
    'get_lsf_status',
    'dispatch_jobs',
]


def check_log(logfile, exited='Exited with exit code',
              successful='Successfully completed', exists=True):
    """ Often logfile doesn't exist because the job hasn't begun
    to run. It is unclear what you want to do in that case...
    logfile : String with path to logfile
    exists  : Is the logfile required to exist
    string  : Value to check for in existing logfile
    """
    if not os.path.exists(logfile):
        return not exists

    if exited in open(logfile).read():
        return 'Exited'
    elif successful in open(logfile).read():
        return 'Successful'
    else:
        return 'None'


def get_lsf_status():
    status_count = {'RUN': 0,
                    'PEND': 0,
                    'SUSP': 0,
                    'USUSP': 0,
                    'NJOB': 0,
                    'UNKNWN': 0}

    p = subprocess.Popen(['bjobs'],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    p.stderr.close()

    output = p.stdout.readlines()

    for line in output[1:]:
        line = line.strip().split()

        status_count['NJOB'] += 1

        for k in status_count.keys():

            if line[2] == k:
                status_count[k] += 1

    return status_count


def add_lsf_args(parser):

    parser.add_argument("--batch", action="store_true",
                        help="Split this job into several batch jobs.")

    parser.add_argument('--time', default=1500, type=int,
                        help='Set the wallclock time allocation for the '
                             'job in minutes.')

    parser.add_argument('--resources', default='rhel60', type=str,
                        help='Set the resource string.')


def dispatch_jobs(exe, args, opts, batch_opts, dry_run=False):
    batch_opts.setdefault('W', 300)
    batch_opts.setdefault('R', 'rhel60')

    # skip_keywords = ['queue','resources','batch','W']

    cmd_opts = ''
    for k, v in opts.__dict__.items():
        if isinstance(v, list):
            continue

        if isinstance(v, bool) and v:
            cmd_opts += ' --%s ' % (k)
        elif isinstance(v, bool):
            continue
        elif v is not None:
            cmd_opts += ' --%s=\"%s\" ' % (k, v)

        #    for x in args:
        #            cmd = '%s %s '%(exe,x)
        #            batch_cmd = 'bsub -W %s -R %s '%(W,resources)
        #            batch_cmd += ' %s %s '%(cmd,cmd_opts)
        #            print(batch_cmd)
        #            os.system(batch_cmd)

    cmd = '%s %s ' % (exe, ' '.join(args))

    batch_optstr = ''
    for k, v in batch_opts.items():
        batch_optstr += ' -%s %s ' % (k, v)

    batch_cmd = 'bsub %s ' % (batch_optstr)
    batch_cmd += ' %s %s ' % (cmd, cmd_opts)
    print(batch_cmd)
    if not dry_run:
        os.system(batch_cmd)
