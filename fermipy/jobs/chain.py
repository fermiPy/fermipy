# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities to execute command line applications.

The `Chain` class inherits from `Link` and allow chaining together several
applications into a single object.
"""
from __future__ import absolute_import, division, print_function

import sys
import os
import copy
import argparse

from collections import OrderedDict

from fermipy.jobs.utils import is_null, is_not_null
from fermipy.jobs.link import Link, extract_arguments
from fermipy.jobs.file_archive import FileDict, FileStageManager
from fermipy.jobs.job_archive import get_timestamp, JobStatus, JobStatusVector, JobDetails, JOB_STATUS_STRINGS
from fermipy.jobs.factory import LinkFactory


def purge_dict(idict):
    odict = {}
    for k, v in idict.items():
        if v in [None, 'none', 'None']:
            continue
        odict[k] = v
    return odict

def insert_app_config(o_dict, key, appname, **kwargs):
    o_dict[key] = purge_dict(kwargs.copy())
    o_dict[key]['appname'] = appname


class Chain(Link):
    """ An object tying together a series of applications into a single application.

    This class keep track of the arguments to pass to the applications
    as well as input and output files.

    Note that this class is itself a `Link`.  This allows you
    to write a python module that implements a chain and also has a
    __main__ function to allow it to be called from the shell.

    Parameters
    ----------

    """

    def __init__(self, linkname, **kwargs):
        """ C'tor

        Parameters
        ----------

        linkname : str
            Unique name of this particular link

        Keyword arguments
        -----------------
        """
        options = kwargs.pop('options', {})
        Link.__init__(self, linkname, options=options, **kwargs)
        self._register_link_classes()
        self._update_options(self.args.copy())
        self._links = OrderedDict()
        self._arg_dict = OrderedDict()

    @classmethod
    def main(cls):
        """ Hook to run this `Chain` from the command line """
        chain = cls.create()
        args = chain.run_argparser(sys.argv[1:])
        chain._run_chain(sys.stdout, args.dry_run)
        chain._finalize(args.dry_run)

    @property
    def links(self):
        """ Return the `OrderedDict` of links """
        return self._links

    @property
    def linknames(self):
        """ Return the name of the links """
        return self._links.keys()

    @property
    def arg_dict(self):
        """ Return the `OrderedDict` of options """
        return self._arg_dict

    def __getitem__(self, key):
        """ Return the `Link` whose linkname is key"""
        return self._links[key]

    def _register_link_classes(self):
        """ Register the classes of the `Link` objects used by this chain """
        pass    

    def _latch_file_info(self):
        """Internal function to update the dictionaries
        keeping track of input and output files
        """
        self._arg_dict = self._map_arguments(self.args)
        self.files.latch_file_info(self._arg_dict)
        self.sub_files.file_dict.clear()
        self.sub_files.update(self.files.file_dict)
        for link in self._links.values():
            self.sub_files.update(link.files.file_dict)
            self.sub_files.update(link.sub_files.file_dict)

    def _map_arguments(self, args):
        """Map arguments to options.

        Parameters
        -------------
        args : dict or `Namespace`
            If a namespace is given, it will be cast to dict

        Returns dict
        """
        raise NotImplementedError('Chain._map_arguments')

    def _build_links(self):
        """Use the values in self._arg_dict to the links """
        for k, v in self._arg_dict.items():
            oo = v.copy()
            appname = oo.get('appname', None)
            if is_not_null(appname):
                oo.pop('appname')
                linkname = oo.get('linkname', appname)
                sub_link_prefix = oo.get('link_prefix', '')
                oo['link_prefix'] = self.link_prefix + sub_link_prefix
                if self._links.has_key(linkname):
                    link.update_args(**oo)
                else:
                    link = LinkFactory.create(appname, **oo) 
                    # This will call link.update_args
                    self.add_link(link, oo)
            else:
                raise KeyError("No appname for link %s %s"%(k,v))

    def _load_arguments(self):
        """Transfer the arguments from 
        self._arg_dict to the links """
        for linkname, opt_vals in self._arg_dict.items():
            appname = opt_vals.get('appname', None)
            if is_not_null(appname):
                oo = opt_vals.copy()
                oo.pop('appname')
                sub_link_prefix = oo.get('link_prefix', '')
                oo['link_prefix'] = self.link_prefix + sub_link_prefix
                if self._links.has_key(linkname):
                    link = self._links[linkname]
                    link.update_args(oo)
                else:
                    link = LinkFactory.create(appname, **oo)               
                    # This will call link.update_args
                    self.add_link(link, oo)
            else:
                raise KeyError("No appname for link %s %s"%(k,v))

    def _set_links_job_archive(self):
        """Pass job_archive along to links"""
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
            Stream to print to

        dry_run : bool
            Print commands but do not run them

        stage_files : bool
            Stage files to and from the scratch area

        """
        #ok = self._pre_run_checks(stream, dry_run)
        # if not ok:
        #    return
        #print ('Chain._run_chain ', self.args)

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

        for linkname, link in self._links.items():
            logfile = os.path.join('logs', "top_%s.log"%link.full_linkname)
            link._archive_self(logfile, status=JobStatus.unknown)
            key = JobDetails.make_fullkey(link.full_linkname)
            if hasattr(link, 'check_status'):
                status_vect = link.check_status(stream, no_wait=True, check_once=True, do_print=False)
            else:
                status_vect = None
            link_status = link.check_job_status(key)
            if link_status in [JobStatus.running, 
                               JobStatus.done]:
                if not force_run:
                    print ("Skipping done link", link.full_linkname)
                    continue
            elif link_status in [JobStatus.failed,
                                 JobStatus.partial_failed]:
                if not resubmit_failed:
                    print ("Skipping failed link", link.full_linkname)
                    continue
            print ("Running link ", link.full_linkname)
            close_file = False
            if dry_run:
                outstr = sys.stdout
            else:
                outstr = stream
            link.run_with_log(dry_run=dry_run, stage_files=False, resubmit_failed=resubmit_failed)
            link_status = link.check_jobs_status()
            link._set_status_self(status=link_status)
            if link_status in [JobStatus.failed, JobStatus.partial_failed]:
                print ("Stoping chain execution at failed link %s"%link.full_linkname)
                failed = True
                break

        if self._file_stage is not None and stage_files and not failed:
            self._stage_output_files(output_file_mapping, dry_run)

        chain_status = self.check_links_status()
        print ("Chain status %i"%(chain_status))
        self._set_status_self(status=chain_status)

        if self._job_archive:
            self._job_archive.file_archive.update_file_status()
            self._job_archive.write_table_file()


    def clear_jobs(self, recursive=True):
        """Clear a dictionary with all the jobs

        If recursive is True this will include jobs from internal `Link`
        """
        if recursive:
            for link in self._links.values():
                link.clear_jobs(recursive)
        self.jobs.clear()  

    def get_jobs(self, recursive=True):
        """Return a dictionary with all the jobs

        If recursive is True this will include jobs from internal `Link`
        """
        if recursive:
            ret_dict = self.jobs.copy()
            for link in self._links.values():
                ret_dict.update(link.get_jobs(recursive))
            return ret_dict
        else:
            return self.jobs

    def missing_input_files(self):
        """Return a dictionary of the missing input files and `Link` they are associated with """
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
        """Return a dictionary of the missing output files and `Link` they are associated with """
        ret_dict = OrderedDict()
        for link in self._links.values():
            link_dict = link.missing_output_files()
            for key, value in link_dict.items():
                try:
                    ret_dict[key] += value
                except KeyError:
                    ret_dict[key] = value
        return ret_dict

    def add_link(self, link, options=None):
        """Append link to this `Chain` """
        self._links[link.linkname] = link
        if options is None:
            options = OrderedDict()
        self._arg_dict[link.linkname] = options
        logfile = os.path.join('logs','top_%s.log'%link.full_linkname)
        link._register_job(JobDetails.topkey, options, logfile, status=JobStatus.unknown)

    def check_links_status(self,
                           fail_running=False,
                           fail_pending=False):
        """Check the status of all the links"""
        n_failed = 0
        n_passed = 0
        n_total = 0        
        status_vector = JobStatusVector()
        for linkname, link in self._links.items():
            link_status = link.check_job_status(fail_running=fail_running,
                                                fail_pending=fail_pending)
            status_vector[link_status] += 1
            
        return status_vector.get_status()


    def run(self, stream=sys.stdout, dry_run=False, stage_files=True, resubmit_failed=False):
        """Run the chain"""
        self._run_chain(stream, dry_run, stage_files, resubmit_failed=resubmit_failed)

    def update_args(self, override_args):
        """Update the argument used to invoke the application

        Note that this will also update the dictionary of input and output files.

        Parameters
        -----------
        override_args : dict
            dictionary passed to the links

        """
        self.args = extract_arguments(override_args, self.args)
        self._arg_dict = self._map_arguments(override_args)
        self._load_arguments()
        #fullkey = JobDetails.make_fullkey(self.linkname)
        #logfile = os.path.join('logs','top_%s.log'%self.linkname)
        #self._register_self(logfile, fullkey, status=JobStatus.unknown)

        scratch_dir = self.args.get('scratch', None)
        if is_not_null(scratch_dir):
            self._file_stage = FileStageManager(scratch_dir, '.')
        for link in self._links.values():
            link._set_file_stage(self._file_stage)
        self._latch_file_info()


    def print_status(self, indent="", recurse=False):
        """Print a summary of the job status for each `Link` in this `Chain`"""
        print ("%s%20s : %20s : %20s"%(indent, "Linkname","Link Status","Jobs Status"))    
        for link in self._links.values():
            if isinstance(link, Chain):
                if recurse:
                    print ("----------   %20s    -----------"%link.linkname)
                    link.print_status(indent+"  ", recurse=True)
                    continue
            if hasattr(link, 'check_status'):
                status_vect = link.check_status(stream=sys.stdout, no_wait=True, do_print=False)
            else:
                status_vect = None
            key = JobDetails.make_fullkey(link.full_linkname)
            link_status = JOB_STATUS_STRINGS[link.check_job_status(key)]
            if status_vect is None:
                jobs_status = JOB_STATUS_STRINGS[link.check_jobs_status()]
            else:
                jobs_status = status_vect
            print ("%s%20s : %20s : %20s"%(indent, link.linkname, link_status, jobs_status))
 
    def print_summary(self, stream=sys.stdout, indent="", recurse_level=2):
        """Print a summary of the activity done by this `Chain`.

        Parameters
        -----------
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
            for link in self._links.values():
                stream.write("\n")
                link.print_summary(stream, indent, recurse_level)
