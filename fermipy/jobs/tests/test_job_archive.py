# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import os

from fermipy.jobs.job_archive import JobStatus, JobDetails, JobArchive
from fermipy.jobs.file_archive import FileFlags
from fermipy.jobs.chain import Link


def test_job_details():
    """ Test that we can build a `JobDetails` object """

    job = JobDetails(dbkey=-1,
                     jobname='test',
                     jobkey='dummy',
                     appname='test',
                     logfile='test.log',
                     job_config=str({}),
                     timestamp=0,
                     infile_ids=[3, 4],
                     outfile_ids=[6, 7],
                     rm_ids=[3],
                     status=JobStatus.no_job)
    job_dict = {job.dbkey: job}
    table, table_ids = JobDetails.make_tables(job_dict)
    job_dict2 = JobDetails.make_dict(table)
    job2 = job_dict2[job.dbkey]

    assert job.jobname == job2.jobname
    assert job.dbkey == job2.dbkey
    assert job.logfile == job2.logfile
    assert job.status == job2.status


def test_job_archive():
    """ Test that we can build a JobArchive """

    class DummyLink(Link):
        """ Dummy class for this test """

        appname = 'test_app'
        linkname_default = 'test'
        usage = '%s [options]' % (appname)
        description = "Link to run %s" % (appname)

        default_options = dict(optstr=('CALDB', 'options', str),
                               infile1=(None, 'Input file 1', str),
                               infile2=(None, 'Input file 2', str),
                               infile3=(None, 'Input file 3', str),
                               outfile1=(None, 'Output file 1', str),
                               outfile2=(None, 'Output file 1', str))
        default_file_args = dict(infile1=FileFlags.input_mask,
                                 infile2=FileFlags.input_mask,
                                 infile3=FileFlags.input_mask,
                                 outfile1=FileFlags.output_mask,
                                 outfile2=FileFlags.output_mask)

    link = DummyLink()
    job_archive = JobArchive(file_archive_table='archive_files.fits',
                             job_archive_table='archive_jobs.fits',
                             base_path=os.path.abspath('.'))

    job_archive._file_archive.register_file('input1_1.fits', 0)
    job_archive._file_archive.register_file('input1_2.fits', 0)
    job_archive._file_archive.register_file('input1_3.fits', 0)
    job_archive._file_archive.register_file('input2_1.fits', 0)
    job_archive._file_archive.register_file('input2_2.fits', 0)
    job_archive._file_archive.register_file('input2_3.fits', 0)

    config_1 = dict(infile1='input1_1.fits',
                    infile2='input1_2.fits',
                    infile3='input1_3.fits',
                    outfile1='output1_1.fits',
                    outfile2='output1_2.fits')
    config_2 = dict(infile1='input2_1.fits',
                    infile2='input2_2.fits',
                    infile3='input2_3.fits',
                    outfile1='output2_1.fits',
                    outfile2='output2_2.fits')

    link.update_args(config_1)
    job = job_archive.register_job_from_link(link, 'dummy1', logfile='dummy1.log')

    link.update_args(config_2)
    job2 = job_archive.register_job_from_link(link, 'dummy2', logfile='dummy2.log')

    assert job
    assert job2


if __name__ == '__main__':
    test_job_details()
    test_job_archive()
