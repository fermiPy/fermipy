# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities to execute command line applications.

The main class is `Link`, which wraps a single command line application.

The `Chain` class inherits from `Link` and allow chaining together several
applications into a single object.
"""
from __future__ import absolute_import, division, print_function

import sys
import os
import copy

from collections import OrderedDict

from fermipy.jobs.file_archive import FileDict, FileStageManager
from fermipy.jobs.job_archive import get_timestamp, JobStatus, JobDetails


def extract_arguments(args, defaults, mapping):
    """ Extract a set of arguments from a large dictionary

    Parameters
    ----------

    args : dict
        Dictionary with the arguments values to use

    defaults : dict
        Dictionary with all the argument to extract, and default values for each

    mapping : dict
        Dictionary mapping key in defaults to key in args
        This is useful:
        1) when two applications use different names for what is
        effectively the same parameter
        2) when you want to build a chain with multiple instances of
        the same application and pass different argument values to the
        different instances

    Returns dict filled with the arguments to pass to gtapp
    """
    out_dict = convert_option_dict_to_dict(defaults)
    for key in defaults.keys():
        if mapping is not None:
            try:
                mapped_key = mapping[key]
                mapped_val = args.get(mapped_key, None)
                if mapped_val is None:
                    mapped_key = key
            except KeyError:
                mapped_key = key
        else:
            mapped_key = key

        mapped_val = args.get(mapped_key, None)
        if mapped_val is None:
            pass
        else:
            out_dict[key] = mapped_val
    return out_dict


def check_files(filelist,
                file_stage_manager=None,
                return_found=True,
                return_missing=True):
    """Check that all files in a list exist

    Return two lists: (found, missing)
    """
    found = []
    missing = []
    none_count = 0
    for fname in filelist:

        if fname is None:
            none_count += 1
            continue
        if os.path.exists(fname):
            found.append(fname)
            continue
        if os.path.exists(fname + '.gz'):
            found.append(fname)
            continue
        if file_stage_manager is not None:
            fname = file_stage_manager.get_scratch_path(fname)
            if os.path.exists(fname):
                found.append(fname)
                continue
        missing.append(fname)
    if return_found and return_missing:
        return found, missing
    elif return_found:
        return found
    elif return_missing:
        return missing
    else:
        return None


def add_argument(parser, dest, info):
    """ Add an argument to an `argparse.ArgumentParser` object """
    default, helpstr, typeinfo = info

    if typeinfo == list:
        parser.add_argument('%s' % dest, nargs='+', default=None, help=helpstr)
    elif typeinfo == bool:
        parser.add_argument('--%s' % dest, action='store_true', help=helpstr)
    else:
        parser.add_argument('--%s' % dest, action='store', type=typeinfo,
                            default=default, help=helpstr)


def convert_value_to_option_tuple(value, helpstr=None):
    """Convert a value to a tuple of the form expected by `Link.options`

    Returns (value, helpstr, type(value)
    """
    if helpstr is None:
        helpstr = "Unknown"
    return (value, helpstr, type(value))


def convert_dict_to_option_dict(input_dict):
    """Convert a simple key-value dictionary to a dictionary of options tuples"""
    ret_dict = {}
    for key, value in input_dict.items():
        ret_dict[key] = convert_value_to_option_tuple(value)
    return ret_dict


def convert_option_dict_to_dict(option_dict):
    """Convert a dictionary of options tuples to a simple key-value dictionary"""
    ret_dict = {}
    for key, value in option_dict.items():
        if value is None:
            ret_dict[key] = None
        elif isinstance(value, tuple):
            ret_dict[key] = value[0]
        else:
            ret_dict[key] = value
    return ret_dict


class Link(object):
    """A wrapper for a command line application.

    This class keeps track for the arguments to pass to the application
    as well as input and output files.

    This can be used either with other Link to build a chain, or as
    as standalone wrapper to pass conifguration to the application.

    Parameters
    ----------

    appname : str
        Name of the application
    args : dict
        Up-to-date dictionary with the arguments that will be passed to the application
    defaults : dict
        Dictionary with defaults values for the arguments
    options : dict
        Dictionary with the options that we are allowed to set and default values
    mapping : dict
        Dictionary remapping keys in options to arguments sent to the application
        This is useful when two ScienceTools use different names for what is
        effectively the same parameter
    files : `FileDict`
        Object that keeps track of input and output files
    jobs : `OrderedDict`
        Dictionary mapping keys to `JobDetails`
    """

    def __init__(self, linkname, **kwargs):
        """ C'tor

        Parameters
        -----------
        linkname : str
            Unique name of this particular link

        Keyword arguments
        -----------
        appname : str
            Name of the application (e.g., gtbin)
        parser: `argparse.ArguemntParser'
            Parser with the options that we are allow to set and default values
        options : dict
            Dictionary with the tuples defining that we are allowed to set and default values
        mapping : dict
            Dictionary remapping input argument names
            This is useful when two `Link` use different names for what is
            effectively the same parameter
        file_args : dict
            Dictionary mapping argument to `FileFlags' enum
        file_stage : `FileStageManager`
            Manager for staging files to and from a scratch area
        """
        self.linkname = linkname
        self.appname = kwargs.pop('appname', linkname)
        self.mapping = kwargs.pop('mapping', {})
        self._parser = kwargs.pop('parser', None)
        self._file_stage = kwargs.pop('file_stage', None)
        self._options = {}
        self._options.update(kwargs.pop('options', {}))
        if self._parser is not None:
            self.fill_argparser(self._parser)
        self.args = {}
        self.args.update(convert_option_dict_to_dict(self._options))
        self.files = FileDict(**kwargs)
        self.sub_files = FileDict()
        self.jobs = OrderedDict()

    def print_summary(self, stream=sys.stdout, indent="", recurse_level=2):
        """Print a summary of the activity done by this `Link`.

        Parameters
        -----------
        stream : `file`
            Stream to print to
        indent : str
            Indentation at start of line
        recurse_level : int
            Number of recursion levels to print
        """
        if recurse_level < 0:
            return
        stream.write("%sLink: %s\n" % (indent, self.linkname))
        stream.write("%sN_jobs: %s\n" % (indent, len(self.get_jobs())))
        self.sub_files.print_chain_summary(stream, indent)

    def _get_args(self):
        """Internal function to cast self._options into dictionary

        Returns dict with argument key : value pairings
        """
        args = {}
        args.update(convert_option_dict_to_dict(self._options))
        return args

    def _latch_file_info(self):
        """Internal function to update the dictionaries
        keeping track of input and output files
        """
        self.files.file_dict.clear()
        self.files.latch_file_info(self.args)

    def update_options(self, input_dict):
        """Update the values in self.options

        Parameters
        ----------
        input_dict : dict
            Dictionary with argument key : value pairings

        Inserts values into self._options
        """
        for key, value in input_dict.items():
            new_tuple = (value, self._options[key][1], self._options[key][2])
            self._options[key] = new_tuple

    def update_args(self, override_args):
        """Update the argument used to invoke the application

        Note that this will also update the dictionary of input and output files.

        Parameters
        -----------
        override_args : dict
            Dictionary of arguments to override the current values
        """
        self.args = extract_arguments(override_args, self.args, self.mapping)
        self._latch_file_info()
        scratch_dir = self.args.get('scratch', None)
        if scratch_dir is not None:
            self._file_stage = FileStageManager(scratch_dir, '.')

    def get_failed_jobs(self):
        """Return a dictionary with the subset of jobs that are marked as failed"""
        failed_jobs = {}
        for job_key, job_details in self.jobs.items():
            if job_details.status == JobStatus.failed:
                failed_jobs[job_key] = job_details
        return failed_jobs

    def get_jobs(self, recursive=True):
        """Return a dictionary with all the jobs

        If recursive is True this will include jobs from internal `Link`
        """
        if recursive:
            ret_dict = self.jobs.copy()
            return ret_dict
        else:
            return self.jobs

    @property
    def arg_names(self):
        """Return the list of arg names """
        return [self.args.keys()]

    def update_sub_file_dict(self, sub_files):
        """Update a file dict with information from self"""
        sub_files.file_dict.clear()
        for job_details in self.jobs.values():
            if job_details.file_dict is not None:
                sub_files.update(job_details.file_dict)
            if job_details.sub_file_dict is not None:
                sub_files.update(job_details.sub_file_dict)

    def check_input_files(self,
                          return_found=True,
                          return_missing=True):
        """Check if input files exist.

        Return two lists: (found, missing)
        """
        all_input_files = self.files.chain_input_files + self.sub_files.chain_input_files
        return check_files(all_input_files, self._file_stage,
                           return_found, return_missing)

    def check_output_files(self,
                           return_found=True,
                           return_missing=True):
        """Check if output files exist.

        Return two lists: (found, missing)
        """
        all_output_files = self.files.chain_output_files + self.sub_files.chain_output_files
        return check_files(all_output_files, self._file_stage,
                           return_found, return_missing)

    def missing_input_files(self):
        """Make and return a dictionary of the missing input files.

        This returns a dictionary mapping
        filepath to list of links that use the file as input.
        """
        missing = self.check_input_files(return_found=False)
        ret_dict = {}
        for miss_file in missing:
            ret_dict[miss_file] = [self.linkname]
        return ret_dict

    def missing_output_files(self):
        """Make and return a dictionary of the missing output files.

        This returns a dictionary mapping
        filepath to list of links that product the file as output.
        """
        missing = self.check_output_files(return_found=False)
        ret_dict = {}
        for miss_file in missing:
            ret_dict[miss_file] = [self.linkname]
        return ret_dict

    def formatted_command(self):
        """Build and return the formatted command for this `Link`.

        This is exactly the command as called from the Unix command line.
        """
        command_template = self.command_template()
        format_dict = self.args.copy()
        command = command_template.format(**format_dict)
        return command

    def make_argv(self):
        """Generate the vector of arguments for this `Link`.

        This is exactly the 'argv' generated for the
        command as called from the Unix command line.
        """
        command = self.formatted_command()
        tokens = command.split()
        return tokens[1:]

    def pre_run_checks(self, stream=sys.stdout, dry_run=False):
        """Do some checks before running this link

        This checks if input and output files are present.

        If input files are missing this will raise `OSError` if dry_run is False
        If all output files are present this return False.

        Parameters
        -----------
        stream : `file`
            Must have 'write' function

        dry_run : bool
            Print command but do not run it

        Returns bool
            True if it is ok to proceed with running the link
        """
        input_missing = self.check_input_files(return_found=False)
        if len(input_missing) != 0:
            if dry_run:
                stream.write("Input files are missing: %s: %i\n" %
                             (self.linkname, len(input_missing)))
            else:
                raise OSError("Input files are missing: %s" % input_missing)

        output_found, output_missing = self.check_output_files()
        if len(output_missing) == 0:
            stream.write("All output files for %s already exist: %i %i %i\n" %
                         (self.linkname, len(output_found),
                          len(output_missing), len(self.files.output_files)))
            if dry_run:
                pass
            else:
                return False
        return True

    def set_file_stage(self, file_stage):
        """Set this link to use a `FileStageManager` to copy files
        to and from a scratch area
        """
        self._file_stage = file_stage

    def run_link(self, stream=sys.stdout, dry_run=False, stage_files=True):
        """Runs this link.

        This checks if input and output files are present.

        If input files are missing this will raise `OSError` if dry_run is False
        If all output files are present this will skip execution.

        Parameters
        -----------
        stream : `file`
            Must have 'write' function

        dry_run : bool
            Print command but do not run it

        stage_files : bool
            Stage files to and from the scratch area
        """
        check_ok = self.pre_run_checks(stream, dry_run)
        if not check_ok:
            return

        if self._file_stage is not None:
            input_file_mapping, output_file_mapping = self.map_scratch_files(
                self.files)
            if stage_files:
                self._file_stage.make_scratch_dirs(input_file_mapping, dry_run)
                self._file_stage.make_scratch_dirs(
                    output_file_mapping, dry_run)
                self.stage_input_files(input_file_mapping, dry_run)

        self.run_command(stream, dry_run)
        if self._file_stage is not None and stage_files:
            self.stage_output_files(output_file_mapping, dry_run)
        self.finalize(dry_run)

    def run_command(self, stream=sys.stdout, dry_run=False):
        """Runs the command for this link.  This method can be overridden by
        sub-classes to invoke a different command

        Parameters
        -----------
        stream : `file`
            Must have 'write' function

        dry_run : bool
            Print command but do not run it
        """
        command = self.formatted_command()
        if dry_run:
            stream.write("%s\n" % command)
            stream.flush()
        else:
            os.system(command)

    def run(self, stream=sys.stdout, dry_run=False):
        """Runs this link.

        This version is intended to be overwritten by sub-classes so
        as to provide a single function that behaves the same
        for all version of `Link`

        Parameters
        -----------
        stream : `file`
            Must have 'write' function

        dry_run : bool
            Print command but do not run it
        """
        self.run_link(stream, dry_run)

    def command_template(self):
        """Build and return a string that can be used as a template invoking
        this chain from the command line.

        The actual command can be obtainted by using
        `self.command_template().format(**self.args)`
        """
        com_out = self.appname
        arg_string = ""
        flag_string = ""
        # Loop over the key, value pairs in self.args
        for key, val in self.args.items():
            # Check if the value is set in self._options
            # If so, get the value from there
            if val is None:
                opt_val = self._options[key][0]
            else:
                opt_val = val
            opt_type = self._options[key][2]
            if key == 'args':
                # 'args' is special, pull it out and move it to the back
                arg_string += ' {%s}' % key
            elif opt_type is bool:
                if opt_val:
                    flag_string += ' --%s' % (key)
            elif opt_type is list:
                if opt_val is None:
                    continue
                elif isinstance(opt_val, str):
                    arg_string += ' %s' % opt_val
                elif isinstance(opt_val, list):
                    for arg_val in opt_val:
                        arg_string += ' %s' % arg_val
            else:
                com_out += ' --%s {%s}' % (key, key)
        com_out += flag_string
        com_out += arg_string
        return com_out

    def run_argparser(self, argv):
        """Initialize a link with a set of arguments using an `argparser.ArgumentParser`
        """
        if self._parser is None:
            raise ValueError('Link was not given a parser on initialization')
        args = self._parser.parse_args(argv)
        self.update_args(args.__dict__)
        return args

    def fill_argparser(self, parser):
        """Fill an `argparser.ArgumentParser` with the options from this chain
        """
        for key, val in self._options.items():
            add_argument(parser, key, val)

    def create_job_details(self, key, job_config, logfile, status):
        """Create a `JobDetails` for a single job

        Parameters
        ----------

        key : str
            Key used to identify this particular job

        job_config : dict
            Dictionary with arguements passed to this particular job

        logfile : str
            Name of the associated log file

        status : int
            Current status of the job

        Returns `JobDetails`
        """
        self.update_args(job_config)
        job_details = JobDetails(jobname=self.linkname,
                                 jobkey=key,
                                 appname=self.appname,
                                 logfile=logfile,
                                 job_config=job_config,
                                 timestamp=get_timestamp(),
                                 file_dict=copy.deepcopy(self.files),
                                 sub_file_dict=copy.deepcopy(self.sub_files),
                                 status=status)
        return job_details

    def register_job(self, key, job_config, logfile, status):
        """Create a `JobDetails` for this link
        and add it to the self.jobs dictionary.

        Parameters
        ----------

        key : str
            Key used to identify this particular job

        job_config : dict
            Dictionary with arguments passed to this particular job

        logfile : str
            Name of the associated log file

        status : int
            Current status of the job

        Returns `JobDetails`
        """
        job_details = self.create_job_details(key, job_config, logfile, status)
        self.jobs[job_details.fullkey] = job_details
        return job_details

    def map_scratch_files(self, file_dict):
        """Build and return the mapping for copying files to and from scratch area"""
        if self._file_stage is None:
            return ({}, {})
        input_files = file_dict.input_files_to_stage
        output_files = file_dict.output_files_to_stage
        input_file_mapping = self._file_stage.map_files(input_files)
        output_file_mapping = self._file_stage.map_files(output_files)
        self.update_file_args(input_file_mapping)
        self.update_file_args(output_file_mapping)
        return input_file_mapping, output_file_mapping

    def update_file_args(self, file_mapping):
        """Adjust the arguments to deal with staging files to the scratch area"""
        for key, value in self.args.items():
            new_value = file_mapping.get(value, value)
            if new_value != value:
                self.args[key] = new_value

    def stage_input_files(self, file_mapping, dry_run=True):
        """Stage the input files to the scratch area and adjust the arguments accordingly"""
        print ("Staging input ", file_mapping)
        if self._file_stage is None:
            return
        self._file_stage.copy_to_scratch(file_mapping, dry_run)

    def stage_output_files(self, file_mapping, dry_run=True):
        """Stage the input files to the scratch area and adjust the arguments accordingly"""
        print ("Staging output ", file_mapping)
        if self._file_stage is None:
            return
        self._file_stage.copy_from_scratch(file_mapping, dry_run)

    def finalize(self, dry_run=False):
        """Remove / compress files as requested """
        for rmfile in self.files.temp_files:
            if dry_run:
                print ("remove %s" % rmfile)
            else:
                os.remove(rmfile)
        for gzfile in self.files.gzip_files:
            if dry_run:
                print ("gzip %s" % gzfile)
            else:
                os.system('gzip -9 %s' % gzfile)


class Chain(Link):
    """ An object tying together a series of applications into a single application.

    This class keep track of the arguments to pass to the applications
    as well as input and output files.

    Note that this class is itself a `Link`.  This allows you
    to write a python module that implements a chain and also has a
    __main__ function to allow it to be called from the shell.

    Parameters
    ----------

    argmapper : function or None
        Function that maps input options (in self._options) to the
        format that is passed to the links in the chains.
        If None, then no mapping is applied.
        This is useful if you want to build a complicated set of options
        from a few inputs.
    """

    def __init__(self, linkname, links, **kwargs):
        """ C'tor

        Parameters
        ----------

        linkname : str
            Unique name of this particular link
        links : list
            A list of `Link` objects

        Keyword arguments
        -----------------
        argmapper : function or None
            Function that maps input options (in self._options) to the
            format that is passed to the links in the chains.
        """
        Link.__init__(self, linkname, **kwargs)
        self._argmapper = kwargs.get('argmapper', None)
        self.update_options(self.args.copy())
        self._links = OrderedDict()
        for link in links:
            self._links[link.linkname] = link

    @property
    def links(self):
        """ Return the list of links """
        return self._links

    @property
    def argmapper(self):
        """Return the arugment mapping function, if exits """
        return self._argmapper

    def __getitem__(self, key):
        """ Return the `Link` whose linkname is key"""
        return self._links[key]

    def _latch_file_info(self):
        """Internal function to update the dictionaries
        keeping track of input and output files
        """
        remapped = self.map_arguments(self.args)
        self.files.latch_file_info(remapped)
        self.sub_files.file_dict.clear()
        self.sub_files.update(self.files.file_dict)
        for link in self._links.values():
            self.sub_files.update(link.files.file_dict)
            self.sub_files.update(link.sub_files.file_dict)

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

    def map_arguments(self, args):
        """Map arguments to options.

        This will use self._argmapper is it is defined.

        Parameters
        -------------
        args : dict or `Namespace`
            If a namespace is given, it will be cast to dict

        Returns dict
        """
        if self._argmapper is None:
            try:
                return args.__dict__
            except AttributeError:
                return args
        else:
            mapped = self._argmapper(args)
            if mapped is None:
                try:
                    return args.__dict__
                except AttributeError:
                    return args
            else:
                return mapped

    def add_link(self, link):
        """Append link to this `Chain` """
        self._links[link.linkname] = link

    def run_chain(self, stream=sys.stdout, dry_run=False, stage_files=True):
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
        #ok = self.pre_run_checks(stream, dry_run)
        # if not ok:
        #    return

        if self._file_stage is not None:
            input_file_mapping, output_file_mapping = self.map_scratch_files(
                self.sub_files)
            if stage_files:
                self._file_stage.make_scratch_dirs(input_file_mapping, dry_run)
                self._file_stage.make_scratch_dirs(
                    output_file_mapping, dry_run)
                self.stage_input_files(input_file_mapping, dry_run)

        for link in self._links.values():
            print ("Running link ", link.linkname)
            link.run_link(stream=stream, dry_run=dry_run, stage_files=False)

        if self._file_stage is not None and stage_files:
            self.stage_output_files(output_file_mapping, dry_run)

    def run(self, stream=sys.stdout, dry_run=False):
        self.run_chain(stream, dry_run)

    def update_args(self, override_args):
        """Update the argument used to invoke the application

        Note that this will also update the dictionary of input and output files.

        Parameters
        -----------
        override_args : dict
            dictionary passed to the links

        """
        self.args = extract_arguments(override_args, self.args, self.mapping)
        remapped = self.map_arguments(override_args)

        scratch_dir = self.args.get('scratch', None)
        if scratch_dir is not None:
            self._file_stage = FileStageManager(scratch_dir, '.')
        for link in self._links.values():
            link.update_args(remapped)
            link.set_file_stage(self._file_stage)
        self._latch_file_info()
