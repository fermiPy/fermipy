# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Factory module to return the default interace to the batch farm
"""
from __future__ import absolute_import, division, print_function


# This sets the default job type.
# We are going to want a better way to control this down the road
DEFAULT_JOB_TYPE = 'slac'


def get_batch_job_args(job_time=1500):
    """ Get the correct set of batch jobs arguments.

    Parameters
    ----------

    job_time : int
        Expected max length of the job, in seconds.
        This is used to select the batch queue and set the
        job_check_sleep parameter that sets how often
        we check for job completion.

    """
    if DEFAULT_JOB_TYPE == 'slac':
        from fermipy.jobs.slac_impl import get_slac_default_args
        return get_slac_default_args(job_time)
    elif DEFAULT_JOB_TYPE == 'native':
        from fermipy.jobs.native_impl import get_native_default_args
        return get_native_default_args()


def get_batch_job_interface(job_time=1500):
    """ Create a batch job interface object.

    Parameters
    ----------

    job_time : int
        Expected max length of the job, in seconds.
        This is used to select the batch queue and set the
        job_check_sleep parameter that sets how often
        we check for job completion.

    """
    batch_job_args = get_batch_job_args(job_time)

    if DEFAULT_JOB_TYPE == 'slac':
        from fermipy.jobs.slac_impl import SlacInterface
        return Slac_Interface(**batch_job_args)
    elif DEFAULT_JOB_TYPE == 'native':
        from fermipy.jobs.native_impl import NativeInterface
        return Native_Interface(**batch_job_args)