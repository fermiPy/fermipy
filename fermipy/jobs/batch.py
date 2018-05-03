# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Factory module to return the default interace to the batch farm 
"""
from __future__ import absolute_import, division, print_function


DEFAULT_JOB_TYPE = 'slac'

def get_batch_job_args(job_time=1500):
    if DEFAULT_JOB_TYPE == 'slac':
        from fermipy.jobs.slac_impl import get_slac_default_args
        return get_slac_default_args(job_time)
    elif DEFAULT_JOB_TYPE == 'native':
        from fermipy.jobs.native_impl import get_native_default_args
        return get_native_default_args()


def get_batch_job_interface(job_time=1500):
    batch_job_args = get_batch_job_args(job_time)
    
    if DEFAULT_JOB_TYPE == 'slac':
        from fermipy.jobs.slac_impl import Slac_Interface
        return Slac_Interface(**batch_job_args)
    elif DEFAULT_JOB_TYPE == 'native':
        from fermipy.jobs.native_impl import Native_Interface
        return Native_Interface(**batch_job_args)


    

