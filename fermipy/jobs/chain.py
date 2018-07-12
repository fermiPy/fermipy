# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
The `Chain` class inherits from `Link` and allow chaining together several
applications into a single object.
"""
from __future__ import absolute_import, division, print_function

import sys
import os

from collections import OrderedDict

from fermipy.jobs.utils import is_null, is_not_null
from fermipy.jobs.link import Link, extract_arguments
from fermipy.jobs.file_archive import FileStageManager
from fermipy.jobs.job_archive import JobStatus, JobStatusVector,\
    JobDetails, JOB_STATUS_STRINGS


def purge_dict(idict):
    """Remove null items from a dictionary """
    odict = {}
    for key, val in idict.items():
        if is_null(val):
            continue
        odict[key] = val
    return odict


class Chain(Link):
    """An object tying together a series of applications into a
    single application.

    This class keep track of the arguments to pass to the applications
    as well as input and output files.

    Note that this class is itself a `Link`.  This allows you
    to write a python module that implements a chain and also has a
    __main__ function to allow it to be called from the shell.

    """
    def __init__(self, **kwargs):
        """C'tor """
        super(Chain, self).__init__(**kwargs)
        self._links = OrderedDict()

    @classmethod
    def main(cls):
        """Hook to run this `Chain` from the command line """
        chain = cls.create()
        args = chain._run_argparser(sys.argv[1:])
        chain._run_chain(sys.stdout, args.dry_run)
        chain._finalize(args.dry_run)

    @property
    def links(self):
        """Return the `OrderedDict` of `Link` objects owned by this `Chain` """
        return self._links

    @property
    def linknames(self):
        """Return the name of the `Link` objects owned by this `Chain` """
        return self._links.keys()

    def __getitem__(self, key):
        """Return the `Link` whose linkname is key"""
        return self._links[key]

    def _latch_file_info(self):
        """Internal function to update the dictionaries
        keeping track of input and output files
        """
        self._map_arguments(self.args)
        self.files.latch_file_info(self.args)
        self.sub_files.file_dict.clear()
        self.sub_files.update(self.files.file_dict)
        for link in self._links.values():
            self.sub_files.update(link.files.file_dict)
            self.sub_files.update(link.sub_files.file_dict)

    def _map_arguments(self, args):
        """Map arguments from the top-level `Chain` options
        to the options of `Link` object contained within.

        Note in many cases this function will also
        decide what set of `Link` objects to create.

        Parameters
        -------------
        args : dict
            The current values of the options of the top-level `Chain`

        """
        raise NotImplementedError('Chain._map_arguments')

    def _set_link(self, linkname, cls, **kwargs):
        """Transfer options kwargs to a `Link` object,
        optionally building the `Link if needed.

        Parameters
        ----------

        linkname : str
            Unique name of this particular link

        cls : type
            Type of `Link` being created or managed

        """
        val_copy = purge_dict(kwargs.copy())
        sub_link_prefix = val_copy.pop('link_prefix', '')
        link_prefix = self.link_prefix + sub_link_prefix
        create_args = dict(linkname=linkname,
                           link_prefix=link_prefix,
                           job_archive=val_copy.pop('job_archive', None),
                           file_stage=val_copy.pop('file_stage', None))
        job_args = val_copy
        if linkname in self._links:
            link = self._links[linkname]
            link.update_args(job_args)
        else:
            link = cls.create(**create_args)
            self._links[link.linkname] = link
            logfile_default = os.path.join('logs', '%s.log' % link.full_linkname)
            logfile = kwargs.setdefault('logfile', logfile_default)
            link._register_job(JobDetails.topkey, job_args,
                               logfile, status=JobStatus.unknown)
        return link

    def _set_links_job_archive(self):
        """Pass self._job_archive along to links"""
        for link in self._links.values():
            link._job_archive = self._job_archive

    def _run_chain(self,
                   stream=sys.stdout,
                   dry_run=False,
                   stage_files=True,
                   force_run=False,
                   resubmit_failed=False):
        """Run all the links in the chain

        Parameters
        -----------
        stream : `file`
            Stream to print to,
            Must have 'write' function

        dry_run : bool
            Print commands but do not run them

        stage_files : bool
            Stage files to and from the scratch area

        force_run : bool
            Run jobs, even if they are marked as done

        resubmit_failed : bool
            Resubmit failed jobs

        """
        self._set_links_job_archive()
        failed = False

        if self._file_stage is not None:
            input_file_mapping, output_file_mapping = self._map_scratch_files(
                self.sub_files)
            if stage_files:
                self._file_stage.make_scratch_dirs(input_file_mapping, dry_run)
                self._file_stage.make_scratch_dirs(
                    output_file_mapping, dry_run)
                self._stage_input_files(input_file_mapping, dry_run)

        for link in self._links.values():
            logfile = os.path.join('logs', "%s.log" % link.full_linkname)
            link._archive_self(logfile, status=JobStatus.unknown)
            key = JobDetails.make_fullkey(link.full_linkname)
            if hasattr(link, 'check_status'):
                link.check_status(stream, no_wait=True,
                                  check_once=True, do_print=False)
            else:
                pass
            link_status = link.check_job_status(key)
            if link_status in [JobStatus.done]:
                if not force_run:
                    print ("Skipping done link", link.full_linkname)
                    continue
            elif link_status in [JobStatus.running]:
                if not force_run and not resubmit_failed:
                    print ("Skipping running link", link.full_linkname)
                    continue
            elif link_status in [JobStatus.failed,
                                 JobStatus.partial_failed]:
                if not resubmit_failed:
                    print ("Skipping failed link", link.full_linkname)
                    continue
            print ("Running link ", link.full_linkname)
            link.run_with_log(dry_run=dry_run, stage_files=False,
                              resubmit_failed=resubmit_failed)
            link_status = link.check_jobs_status()
            link._set_status_self(status=link_status)
            if link_status in [JobStatus.failed]:
                print ("Stoping chain execution at failed link %s" %
                       link.full_linkname)
                failed = True
                break
            elif link_status in [JobStatus.partial_failed]:
                print ("Resubmitting partially failed link %s" %
                       link.full_linkname)
                link.run_with_log(dry_run=dry_run, stage_files=False,
                                  resubmit_failed=resubmit_failed)
                link_status = link.check_jobs_status()
                link._set_status_self(status=link_status)
                if link_status in [JobStatus.partial_failed]:
                    print ("Stoping chain execution: resubmission failed %s" %
                           link.full_linkname)
                    failed = True

        if self._file_stage is not None and stage_files and not failed:
            self._stage_output_files(output_file_mapping, dry_run)

        chain_status = self.check_links_status()
        print ("Chain status %i" % (chain_status))
        if chain_status == 5:
            job_status = 0
        else:
            job_status = -1
        self._write_status_to_log(job_status, stream)
        self._set_status_self(status=chain_status)

        if self._job_archive:
            self._job_archive.file_archive.update_file_status()
            self._job_archive.write_table_file()

    def clear_jobs(self, recursive=True):
        """Clear a dictionary with all the jobs

        If recursive is True this will include jobs from all internal `Link`
        """
        if recursive:
            for link in self._links.values():
                link.clear_jobs(recursive)
        self.jobs.clear()

    def get_jobs(self, recursive=True):
        """Return a dictionary with all the jobs

        If recursive is True this will include jobs from all internal `Link`
        """
        if recursive:
            ret_dict = self.jobs.copy()
            for link in self._links.values():
                ret_dict.update(link.get_jobs(recursive))
            return ret_dict
        return self.jobs

    def missing_input_files(self):
        """Make and return a dictionary of the missing input files.

        This returns a dictionary mapping
        filepath to list of `Link` that use the file as input.
        """
        ret_dict = OrderedDict()
        for link in self._links.values():
            link_dict = link.missing_input_files()
            for key, value in link_dict.items():
                try:
                    ret_dict[key] += value
                except KeyError:
                    ret_dict[key] = value
        return ret_dict

    def missing_output_files(self):
        """Make and return a dictionary of the missing output files.

        This returns a dictionary mapping
        filepath to list of links that produce the file as output.
        """
        ret_dict = OrderedDict()
        for link in self._links.values():
            link_dict = link.missing_output_files()
            for key, value in link_dict.items():
                try:
                    ret_dict[key] += value
                except KeyError:
                    ret_dict[key] = value
        return ret_dict

    def check_links_status(self,
                           fail_running=False,
                           fail_pending=False):
        """"Check the status of all the jobs run from the
        `Link` objects in this `Chain` and return a status
        flag that summarizes that.

        Parameters
        ----------

        fail_running : `bool`
            If True, consider running jobs as failed

        fail_pending : `bool`
            If True, consider pending jobs as failed

        Returns
        -------
        status : `JobStatus`
            Job status flag that summarizes the status of all the jobs,

        """
        status_vector = JobStatusVector()
        for link in self._links.values():
            key = JobDetails.make_fullkey(link.full_linkname)
            link_status = link.check_job_status(key,
                                                fail_running=fail_running,
                                                fail_pending=fail_pending)
            status_vector[link_status] += 1

        return status_vector.get_status()

    def run(self, stream=sys.stdout, dry_run=False,
            stage_files=True, resubmit_failed=False):
        """Runs this `Chain`.

        Parameters
        -----------
        stream : `file`
            Stream that this `Link` will print to,
            Must have 'write' function

        dry_run : bool
            Print command but do not run it.

        stage_files : bool
            Copy files to and from scratch staging area.

        resubmit_failed : bool
            Flag for sub-classes to resubmit failed jobs.

        """
        self._run_chain(stream, dry_run, stage_files,
                        resubmit_failed=resubmit_failed)

    def update_args(self, override_args):
        """Update the argument used to invoke the application

        Note that this will also update the dictionary of input
        and output files.

        Parameters
        -----------
        override_args : dict
            dictionary passed to the links

        """
        self.args = extract_arguments(override_args, self.args)
        self._map_arguments(self.args)

        scratch_dir = self.args.get('scratch', None)
        if is_not_null(scratch_dir):
            self._file_stage = FileStageManager(scratch_dir, '.')
        for link in self._links.values():
            link._set_file_stage(self._file_stage)
        self._latch_file_info()

    def print_status(self, indent="", recurse=False):
        """Print a summary of the job status for each `Link` in this `Chain`"""
        print ("%s%30s : %15s : %20s" %
               (indent, "Linkname", "Link Status", "Jobs Status"))
        for link in self._links.values():
            if hasattr(link, 'check_status'):
                status_vect = link.check_status(
                    stream=sys.stdout, no_wait=True, do_print=False)
            else:
                status_vect = None
            key = JobDetails.make_fullkey(link.full_linkname)
            link_status = JOB_STATUS_STRINGS[link.check_job_status(key)]
            if status_vect is None:
                jobs_status = JOB_STATUS_STRINGS[link.check_jobs_status()]
            else:
                jobs_status = status_vect
            print ("%s%30s : %15s : %20s" %
                   (indent, link.linkname, link_status, jobs_status))
            if hasattr(link, 'print_status') and recurse:
                print ("----------   %30s    -----------" % link.linkname)
                link.print_status(indent + "  ", recurse=True)
                print ("------------------------------------------------")

    def print_summary(self, stream=sys.stdout, indent="", recurse_level=2):
        """Print a summary of the activity done by this `Chain`.

        Parameters
        -----------

        stream : `file`
            Stream to print to, must have 'write' method.

        indent : str
            Indentation at start of line

        recurse_level : int
            Number of recursion levels to print

        """
        Link.print_summary(self, stream, indent, recurse_level)
        if recurse_level > 0:
            recurse_level -= 1
            indent += "  "
            for link in self._links.values():
                stream.write("\n")
                link.print_summary(stream, indent, recurse_level)

    def run_analysis(self, argv):
        """Implemented by sub-classes to run a particular analysis"""
        raise RuntimeError("run_analysis called for Chain type object")
