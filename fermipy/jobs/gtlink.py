# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities to chain together a series of ScienceTools apps
"""
from __future__ import absolute_import, division, print_function

import sys
import os

from collections import OrderedDict

from fermipy.jobs.chain import Link
import GtApp


def extract_parameters(pil, keys=None):
    """Extract and return parameter names and values from a pil object

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

def update_gtapp(gtapp, **kwargs):
    """Update the parameters of the object that can run ScienceTools applications

    Parameters:
    ---------------
    gtapp :  `GtApp.GtApp'
        Object that will run the application in question

    kwargs : arguments used to invoke the application
    """
    for key, val in kwargs.items():
        if key in ['pfiles']:
            continue
        if val is None:
            continue
        try:
            gtapp[key] = val
        except ValueError:
            raise ValueError("gtapp failed to set parameter %s %s"%(key, val))


def build_gtapp(appname, **kwargs):
    """Build an object that can run ScienceTools application

    Parameters:
    ---------------
    appname : str
        Name of the application (e.g., gtbin)

    kwargs : arguments used to invoke the application

    Returns `GtApp.GtApp' object that will run the application in question
    """
    gtapp = GtApp.GtApp(appname)
    update_gtapp(gtapp, **kwargs)
    return gtapp


def run_gtapp(gtapp, stream, dry_run, **kwargs):
    """Runs one on the ScienceTools apps

    Taken from fermipy.gtanalysis.run_gtapp by Matt Wood

    Parameters:
    ---------------
    gtapp : `GtApp.GtApp' object
        The application (e.g., gtbin)

    stream : stream object
        Must have 'write' function

    dry_run : bool
        Print command but do not run it

    kwargs : arguments used to invoke the application
    """
    if stream is None:
        stream = sys.stdout

    update_gtapp(gtapp, **kwargs)
    pfiles = kwargs.get('pfiles', None)
    pfiles_orig = os.environ['PFILES']
    if pfiles:
        pfiles = "%s:%s" % (pfiles, pfiles_orig)
        print ("Setting PFILES=%s" % pfiles)
        os.environ['PFILES'] = pfiles

    stream.write("%s\n" % gtapp.command())
    stream.flush()
    if dry_run:
        os.environ['PFILES'] = pfiles_orig
        return

    stdin, stdout = gtapp.runWithOutput(print_command=False)
    for line in stdout:
        stream.write(line.strip())
    stream.flush()
    os.environ['PFILES'] = pfiles_orig



class Gtlink(Link):
    """A wrapper for a single ScienceTools application

    This class keeps track for the arguments to pass to the application
    as well as input and output files.

    This can be used either with other `Link` to build a `Chain`, or as
    as standalone wrapper to pass conifguration to the application.

    See help for `chain.Link' for additional details
    """

    def __init__(self, linkname, **kwargs):
        """C'tor

        See help for `chain.Link' for details

        This calls the base class c'tor then builds a GtApp object
        """
        Link.__init__(self, linkname, **kwargs)
        self.__app = build_gtapp(self.appname, **self.args)

    def update_args(self, override_args):
        """Update the argument used to invoke the application

        See help for `chain.Link' for details

        This calls the base class function then fills the parameters of the GtApp object
        """
        Link.update_args(self, override_args)
        update_gtapp(self.__app, **self.args)

    def get_gtapp(self):
        """Returns a `GTApp` object that will run this `Link` """
        return self.__app

    def run_link(self, stream=sys.stdout, dry_run=False):
        """Runs this `Link`

        Parameters
        -----------
        stream : `file`
            Must have 'write' function

        dry_run : bool
            Print command but do not run it
        """
        input_found, input_missing = self.check_input_files()
        if len(input_missing) != 0:
            if dry_run:
                stream.write("Input files are missing: %s: %i\n" % (self.linkname, len(input_missing)))
            else:
                raise OSError("Input files are missing: %s" % input_missing)

        output_found, output_missing = self.check_output_files()
        if len(output_missing) == 0:
            stream.write("All output files for %s already exist: %i %i %i\n" %
                         (self.linkname, len(output_found), len(output_missing), len(self.output_files)))
            if dry_run:
                pass
            else:
                return

        run_gtapp(self.__app, stream, dry_run, **self.args)

    def command_template(self):
        """Build and return a string that can be used as a template invoking
        this chain from the command line.

        The actual command can be obtainted by using
        self.command_template().format(**self.args)        
        """
        com_out = self.__app.appName
        for key, val in self.args.items():
            if self._options.has_key(key):
                com_out += ' %s={%s}' % (key, key)
            else:
                com_out += ' %s=%s' % (key, val)
        return com_out
