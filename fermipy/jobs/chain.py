# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities to chain together a series of ScienceTools apps
"""
from __future__ import absolute_import, division, print_function

import sys
import os

from collections import OrderedDict


def extract_arguments(args, defaults, mapping):
    """ Extract a set of arguments from a large dictionary

    Parameters:
    ---------------
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
    out_dict = defaults.copy()
    for key in defaults.keys():
        if mapping is not None:
            try:
                mapped_key = mapping[key]
            except KeyError:
                mapped_key = key
        else:
            mapped_key = key

        mapped_val = args.get(mapped_key, None)
        if mapped_val is not None:
            out_dict[key] = mapped_val
    return out_dict


def extract_by_keys(in_dict, keys):
    """ Extract some items from a dictionary, by key

    Parameters:
    ---------------
    in_dict : dict
        Input dictionary

    keys : list
        List of keys to extract

    Returns:
    ---------------
    out_dict : dict
        Dictionary with only those items in keys
    """
    out_dict = {}
    for key in keys:
        try:
            out_value = in_dict[key]
            out_dict[key] = out_value
        except KeyError:
            pass
    return out_dict


def add_flags_to_dict(the_dict, flags):
    """ Add flags to a dict as flag : None pairs """
    for flag in flags:
        the_dict[flag] = None


def extract_parameters(pil, keys=None):
    """ Extract parameter names and values from a pil object

    Parameters:
    ---------------
    pil : `Pil' object

    keys : list
        List of parameter names, if None, extact all parameters

    Returns:
    ---------------
    out_dict : dict
        Dictionary with parameter name, value pairs
    """
    out_dict = {}
    if keys is None:
        keys = pil.keys()
    for key in keys:
        try:
            out_dict[key] = pil[key]
        except ValueError:
            out_dict[key] = None
    return out_dict


def check_files(filelist):
    """ check that all files in a list exist

    return two lists (found, missing)
    """
    found = []
    missing = []
    for fname in filelist:
        if os.path.exists(fname):
            found.append(fname)
        else:
            missing.append(fname)
    return found, missing


class Link(object):
    """  A wrapper for a command line application

    This class keeps track for the arguments to pass to the application
    as well as input and output files.

    This can be used either with other Link to build a chain, or as
    as standalone wrapper to pass conifguration to the application.

    Class Members
    -------------
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
    input_files : list
        List of input files need to run this application
    output_files : list
        List of files produced by this application
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
            Dictionary with the options that we are allowed to set and default values
        flags : list
            List with the flags that we are allowed to set
        mapping : dict
            Dictionary remapping input argument names
            This is useful when two ScienceTools use different names for what is
            effectively the same parameter
        input_file_args : list
            List of args which specify input files
        output_file_args : list
            List of args which specify output files
        """
        self.linkname = linkname
        self.appname = kwargs.get('appname', linkname)
        self.defaults = kwargs.get('defaults', {})
        self.mapping = kwargs.get('mapping', {})
        self.input_file_args = kwargs.get('input_file_args', [])
        self.output_file_args = kwargs.get('output_file_args', [])
        self._parser = kwargs.get('parser', None)
        if self._parser is not None:
            args = self._parser.parse_args([])
            self.options = args.__dict__
            self.flags = kwargs.get('flags', [])
        else:
            self.options = {}
            for key in self.mapping.keys():
                self.options[key] = None
            self.options.update(kwargs.get('options', {}))
            self.flags = kwargs.get('flags', [])
        self.args = self.defaults.copy()
        self.args.update(self.options)
        add_flags_to_dict(self.args, self.flags)
        self.input_file_dict = extract_by_keys(self.args, self.input_file_args)
        self.output_file_dict = extract_by_keys(
            self.args, self.output_file_args)

    def _get_args(self):
        """ Get the arguments """
        args = self.defaults.copy()
        args.update(self.options)
        return args

    def update_args(self, override_args):
        """ Update the argument used to invoke the application

        Note that this will also update the dictionary of input and output files

        Parameters
        -----------
        override_args : dict
            dictionary of arguments to override the current values

        """
        self.args = extract_arguments(override_args, self.args, self.mapping)
        self.input_file_dict = extract_by_keys(self.args,
                                               self.input_file_args)
        self.output_file_dict = extract_by_keys(self.args,
                                                self.output_file_args)

    @property
    def arg_names(self):
        """ Return the list of option names """
        return [self.args.keys()]

    @property
    def input_files(self):
        """ Returns a list of the input files needed by this link """
        return sorted(self.input_file_dict.values())

    @property
    def output_files(self):
        """ Returns a list of the input files produced by this link """
        return sorted(self.output_file_dict.values())

    def check_input_files(self):
        """ Check if input files exist """
        return check_files(self.input_files)

    def check_output_files(self):
        """ Check if input files exist """
        return check_files(self.output_files)

    def missing_input_files(self):
        """ Make a dictionary of the missing input files """
        found, missing = self.check_input_files()
        ret_dict = {}
        for miss_file in missing:
            ret_dict[miss_file] = [self.linkname]
        return ret_dict

    def missing_output_files(self):
        """ Make a dictionary of the missing input files """
        found, missing = self.check_output_files()
        ret_dict = {}
        for miss_file in missing:
            ret_dict[miss_file] = [self.linkname]
        return ret_dict

    def run_link(self, stream=sys.stdout, dry_run=False):
        """ Runs this link

        Parameters
        -----------
        stream : stream object
            Must have 'write' function

        dry_run : bool
            Print command but do not run it
        """
        command_template = self.command_template()
        format_dict = self.args.copy()
        for flag in self.flags:
            if format_dict[flag]:
                format_dict[flag] = '--%s' % flag
            else:
                format_dict[flag] = ''
        command = command_template.format(**format_dict)
        input_found, input_missing = self.check_input_files()
        if len(input_missing) != 0:
            raise OSError("Input files are missing: %s" % input_missing)

        output_found, output_missing = self.check_output_files()
        if len(output_missing) == 0:
            stream.write("All output files for %s already exist: %s" %
                         (self.linkname, output_found))
        if dry_run:
            stream.write("%s\n" % command)
            stream.flush()
        else:
            os.system(command)

    def command_template(self):
        """ Build and return a string that can be used as a template invoking
            this chain from the command line
        """
        com_out = self.appname
        arg_string = ""
        flag_string = ""
        # Loop over the key, value pairs in self.args
        for key, val in self.args.items():
            try:
                # Check if the value is set in self.options
                # If so, get the value from there
                opt_val = self.options[key]
                if key == 'args':
                    # 'args' is special, pull it out and move it to the back
                    arg_string += ' {%s}' % key
                elif key in self.flags:
                    if opt_val:
                        flag_string += ' --%s' % (key)
                else:
                    com_out += ' --%s {%s}' % (key, key)
            except KeyError:
                # Not in self.options
                # So we just take the value from self.args
                # We
                if key == 'args':
                    if isinstance(val, list):
                        for item in val:
                            arg_string += ' %s' % item
                    else:
                        arg_string += ' %s' % val
                elif key in self.flags:
                    if val:
                        flag_string += ' --%s' % (key)
                else:
                    com_out += ' --%s %s' % (key, val)
        com_out += flag_string
        com_out += arg_string
        return com_out

    def run_argparser(self, argv):
        """  Initialize a link with a set of arguments
        """
        if self._parser is None:
            raise ValueError('Link was not given a parser on initialization')
        args = self._parser.parse_args(argv)
        self.update_args(args.__dict__)
        return args

    def fill_argparser(self, parser):
        """ Fill an argument parser with the options from this chain
        """
        for key, val in self.options.items():
            if val is None:
                parser.add_argument("--%s" % (key), type=str, default=None)
            else:
                parser.add_argument("--%s" %
                                    (key), type=type(val), default=val)


class Chain(Link):
    """ An object tying together a series of applications into a single application.

    This class keep track of the arguments to pass to the applications
    as well as input and output files.

    Note that this class is itself a link.  This allows you
    to write a python module tha implements a chain and also has a
    __main__ function to allow it to be called from the shell.

    Class Members
    -------------
    links : `collections.OrderedDict'
       List of the links, in the order they will by executed
    arg_names : list
       List of the input options available for this chain
    args : dict
       Dictionary of input arguments and associated values
    options : dict
       Dictionary of options and associated values, as they will be
       passed to the links in the chain
    argmapper : function or None
       Fucntion that maps input options (in self.options) to the
       format that is passed to the links in the chains.
       If None, then no mapping is applied.
       This is useful if you want to build a complicated set of options
       from a few inputs.
    input_files_dict : dict
       Dictionary of the input files for each link in the chain
    output_file_dict : dict
       Dictionary of the output files for each link in the chain
    """

    def __init__(self, linkname, links, **kwargs):
        """ C'tor

        Parameters:
        ---------------
        links : list
            A list of gtlink objects

        Keyword arguments
        -----------
        rm_keys : list
            Keys for files to remove on completion
        gzip_keys : list
            Keys for files to compress on completion
        argmapper : function or None
            Function that maps input options (in self.options) to the
            format that is passed to the links in the chains.
        """
        Link.__init__(self, linkname, **kwargs)
        self._rm_keys = kwargs.get('rm_keys', [])
        self._gz_keys = kwargs.get('gz_keys', [])
        self._argmapper = kwargs.get('argmapper', None)
        self.options.update(self.map_arguments(self.args.copy()))
        self._links = OrderedDict()
        for link in links:
            link.update_args(self.options)
            self._links[link.linkname] = link

    def __getitem__(self, key):
        """ Return the `gtlink' object corresponding to key """
        return self._links[key]

    @property
    def links(self):
        """ Return the list of links """
        return self._links

    @property
    def input_file_full_dict(self):
        """ Return a dictionary of the input files for each link in the chain """
        ret_dict = OrderedDict()
        for link in self._links.values():
            ret_dict[link.linkname] = link.input_files
        return ret_dict

    @property
    def output_file_full_dict(self):
        """ Return a dictionary of the output files for each link in the chain """
        ret_dict = OrderedDict()
        for link in self._links.values():
            ret_dict[link.linkname] = link.output_files
        return ret_dict

    def missing_input_files(self):
        """ Return a dictionary of the missing input files and links they are associated with """
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
        """ Return a dictionary of the missing input files and links they are associated with """
        ret_dict = OrderedDict()
        for link in self._links.values():
            link_dict = link.missing_output_files()
            for key, value in link_dict.items():
                try:
                    ret_dict[key] += value
                except KeyError:
                    ret_dict[key] = value
        return ret_dict

    @property
    def argmapper(self):
        """ Return the arugment mapping function, if exits """
        return self._argmapper

    def map_arguments(self, args):
        """ Map arguments to options """
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
        """ Add a link to this chain

        link : `link'
            Name for this link, used as a key
        """
        self._links[link.linkname] = link

    def run_chain(self, stream=sys.stdout, dry_run=False):
        """ Run all the links in the chain

        Parameters
        -----------
        stream : stream object
            Must have 'write' function

        dry_run : bool
            Print commands but do not run them
        """
        for link in self._links.values():
            link.run_link(stream=stream, dry_run=dry_run)

    def finalize(self, dry_run=False):
        """ Remove / compress files as requested """
        rmfiles = extract_by_keys(self.options, self._rm_keys)
        gzfiles = extract_by_keys(self.options, self._gz_keys)
        for rmfile in rmfiles.values():
            if dry_run:
                print ("remove %s" % rmfile)
            else:
                os.remove(rmfile)
        for gzfile in gzfiles.values():
            if dry_run:
                print ("gzip %s" % gzfile)
            else:
                os.system('gzip -9 %s' % gzfile)

    def update_links_from_single_dict(self, single_dict):
        """ Update the argument used to invoke the application

        Parameters
        -----------
        single_dict : dict
            dictionary  pass to the links

        """
        remapped = self.map_arguments(single_dict)
        reduced = extract_arguments(remapped, self.options, None)
        
        self.options.update(reduced)
        for link in self._links.values():
            link.update_args(remapped)

