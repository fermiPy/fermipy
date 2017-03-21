# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import os
import subprocess
import shlex
import time
import numpy as np
from fermipy.utils import mkdir

__all__ = [
    'check_log',
    'get_lsf_status',
    'dispatch_jobs',
]


def parse_sleep(sleep):
    MINUTE = 60
    HOUR = 60 * MINUTE
    DAY = 24 * HOUR
    WEEK = 7 * DAY
    if isinstance(sleep, float) or isinstance(sleep, int):
        return sleep
    elif isinstance(sleep, str):
        try:
            return float(sleep)
        except ValueError:
            pass

        if sleep.endswith('s'):
            return float(sleep.strip('s'))
        elif sleep.endswith('m'):
            return float(sleep.strip('m')) * MINUTE
        elif sleep.endswith('h'):
            return float(sleep.strip('h')) * HOUR
        elif sleep.endswith('d'):
            return float(sleep.strip('d')) * DAY
        elif sleep.endswith('w'):
            return float(sleep.strip('w')) * WEEK
        else:
            raise ValueError
    else:
        raise ValueError


def sleep(sleep):
    return time.sleep(parse_sleep(sleep))


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


def dispatch_job(jobname, exe, args, opts, batch_opts, dry_run=True):

    batch_opts.setdefault('W', 300)
    batch_opts.setdefault('R', 'rhel60')
    batch_opts['oo'] = jobname + '.log'

    cmd_opts = ''
    for k, v in opts.items():
        if isinstance(v, list):
            continue
        if isinstance(v, bool) and v:
            cmd_opts += ' --%s ' % (k)
        elif isinstance(v, bool):
            continue
        elif v is not None:
            cmd_opts += ' --%s=\"%s\" ' % (k, v)

    bash_script = "{exe} {args} {opts}"
    scriptexe = jobname + '.sh'
    with open(os.path.join(scriptexe), 'wt') as f:
        f.write(bash_script.format(exe=exe, args=' '.join(args),
                                   opts=cmd_opts))

    batch_optstr = parse_lsf_opts(**batch_opts)
    batch_cmd = 'bsub %s ' % (batch_optstr)
    batch_cmd += ' bash %s' % scriptexe
    print(batch_cmd)
    if not dry_run:
        os.system(batch_cmd)


def bsub(jobname, command, logfile=None, submit=True, sleep='1m', **kwargs):

    # Just one command...
    if 'q' in kwargs and kwargs['q'] == 'local':
        if isinstance(command, str):
            job = command
        else:
            raise Exception("Cannot run job array locally.")
    else:
        if isinstance(command, str):
            job = create_job(jobname, command, logfile)
        else:
            job = create_job_array(jobname, command, logfile, sleep)
        opts = parse_lsf_opts(**kwargs)
        job = "bsub " + opts + job

    print(job)
    if submit:
        status = subprocess.call(shlex.split(job))
    else:
        status = 0
    return status


def parse_lsf_opts(**kwargs):

    for k, v in kwargs.items():
        if v is None:
            kwargs.pop(k)
    return ''.join('-%s "%s" ' % (k, v) for k, v in kwargs.items())


def create_job(jobname, command, logfile):
    params = dict(name=jobname,
                  cmnd=command,
                  log=logfile)
    if logfile is None:
        job = """-J %(name)s %(cmnd)s""" % (params)
    else:
        job = """-oo %(log)s -J %(name)s %(cmnd)s""" % (params)
    return job


def create_job_array(jobname, commands, logfiles=None, sleep='1m'):
    subdir = mkdir("sub")
    outdir = mkdir("log")

    subbase = os.path.join(subdir, os.path.basename(jobname))
    outbase = os.path.join(outdir, os.path.basename(jobname))

    create_scripts(commands, subbase, sleep)
    if logfiles is not None:
        link_logfiles(logfiles, outbase)

    submit = "sh " + subbase + ".${LSB_JOBINDEX}"
    output = outbase + ".%I"

    njobs = len(commands)
    params = dict(name=jobname,
                  cmnd=submit,
                  log=output,
                  njobs=njobs)

    job = """-oo %(log)s -J %(name)s[1-%(njobs)i] %(cmnd)s""" % (params)
    return job


def create_scripts(commands, subbase="submit", sleep='1m'):
    # Some interesting things we do here:
    # 1) cat the script for the logfile
    # 2) sleep to prevent overload
    # 3) Return the exit value of the command
    sleeps = np.linspace(0, parse_sleep(sleep), len(commands))
    for i, (command, sleep) in enumerate(zip(commands, sleeps)):
        filename = subbase + ".%i" % (i + 1)
        f = open(filename, 'w')
        f.write('%s' % (os.path.basename(filename)).center(35, '#'))
        f.write('\n\n')
        f.write("cat $0;\n")
        f.write("sleep %i;\n" % (sleep))
        f.write(command)
        f.write("\nexit $?;")
        f.write("\n\n")
        f.write("Output follows...".center(35, '#'))
        f.write("\n\n")
        f.close()


def link_logfiles(logfiles, outbase="output"):
    for i, log in enumerate(logfiles):
        log = os.path.expandvars(log)
        output = "%s.%i" % (outbase, i + 1)
        if os.path.lexists(log):
            if log == os.devnull:
                pass
            else:
                os.remove(log)
        if os.path.lexists(output):
            os.remove(output)
        os.symlink(log, output)


def submit_jobs(exe, infiles, outfiles, opts,
                overwrite=False, dry_run=False, **kwargs):

    batch_opts = {'W': kwargs.get('time', 300),
                  'R': kwargs.get('resources', 'rhel60'),
                  'oo': 'batch.log'}

    for infile, outfile in zip(infiles, outfiles):
        if os.path.isfile(outfile) and not overwrite:
            print('Output file exists, skipping.', outfile)
            continue
        jobname = os.path.splitext(outfile)[0]
        dispatch_job(jobname, exe, infile, opts, batch_opts, dry_run=dry_run)
