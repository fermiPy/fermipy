# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities to chain together a series of ScienceTools apps
"""
from __future__ import absolute_import, division, print_function

import sys
import os

from fermipy.jobs.link import Link
import GtApp


def extract_parameters(pil, keys=None):
    """Extract and return parameter names and values from a pil object

    Parameters
    ----------

    pil : `Pil` object

    keys : list
        List of parameter names, if None, extact all parameters

    Returns
    -------

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

    Parameters
    ----------

    gtapp :  `GtApp.GtApp`
        Object that will run the application in question

    kwargs : arguments used to invoke the application
    """
    for key, val in kwargs.items():
        if key in ['pfiles', 'scratch']:
            continue
        if val is None:
            continue
        try:
            gtapp[key] = val
        except ValueError:
            raise ValueError(
                "gtapp failed to set parameter %s %s" % (key, val))
        except KeyError:
            raise KeyError("gtapp failed to set parameter %s %s" % (key, val))


def _set_pfiles(dry_run, **kwargs):
    """Set the PFILES env var
    
    Parameters
    ----------

    dry_run : bool
        Don't actually run

    Keyword arguments
    -----------------

    pfiles : str
        Value to set PFILES

    Returns
    -------

    pfiles_orig : str
        Current value of PFILES envar
    """
    pfiles_orig = os.environ['PFILES']
    pfiles = kwargs.get('pfiles', None)
    if pfiles:
        if dry_run:
            print("mkdir %s" % pfiles)
        else:
            try:
                os.makedirs(pfiles)
            except OSError:
                pass
        pfiles = "%s:%s" % (pfiles, pfiles_orig)
        os.environ['PFILES'] = pfiles
    return pfiles_orig


def _reset_pfiles(pfiles_orig):
    """Set the PFILES env var
    
    Parameters
    ----------

    pfiles_orig : str
        Original value of PFILES

    """
    os.environ['PFILES'] = pfiles_orig


def build_gtapp(appname, dry_run, **kwargs):
    """Build an object that can run ScienceTools application

    Parameters
    ----------
    appname : str
        Name of the application (e.g., gtbin)

    dry_run : bool
        Print command but do not run it

    kwargs : arguments used to invoke the application

    Returns `GtApp.GtApp` object that will run the application in question
    """
    pfiles_orig = _set_pfiles(dry_run, **kwargs)
    gtapp = GtApp.GtApp(appname)
    update_gtapp(gtapp, **kwargs)
    _reset_pfiles(pfiles_orig)
    return gtapp


def run_gtapp(gtapp, stream, dry_run, **kwargs):
    """Runs one on the ScienceTools apps

    Taken from fermipy.gtanalysis.run_gtapp by Matt Wood

    Parameters
    ----------

    gtapp : `GtApp.GtApp` object
        The application (e.g., gtbin)

    stream : stream object
        Must have 'write' function

    dry_run : bool
        Print command but do not run it

    kwargs : arguments used to invoke the application
    """
    if stream is None:
        stream = sys.stdout

    pfiles_orig = _set_pfiles(dry_run, **kwargs)    
    update_gtapp(gtapp, **kwargs)

    stream.write("%s\n" % gtapp.command())
    stream.flush()
    if dry_run:
        _reset_pfiles(pfiles_orig)       
        return 0

    try:
        stdin, stdout = gtapp.runWithOutput(print_command=False)
        for line in stdout:
            stream.write(line.strip())
        stream.flush()
        return_code = 0
    except:
        stream.write('Exited with exit code -1\n')
        return_code = -1

    _reset_pfiles(pfiles_orig)
    return return_code

class Gtlink(Link):
    """A wrapper for a single ScienceTools application

    This class keeps track for the arguments to pass to the application
    as well as input and output files.

    This can be used either with other `Link` to build a `Chain`, or as
    as standalone wrapper to pass conifguration to the application.

    See help for `chain.Link` for additional details
    """
    appname = 'dummy'
    linkname_default = 'dummy'
    usage = '%s [options]' %(appname)
    description = "Link to run %s"%(appname)

    def __init__(self, **kwargs):
        """C'tor

        See help for `chain.Link` for details

        This calls the base class c'tor then builds a GtApp object
        """
        super(Gtlink, self).__init__(**kwargs)
        self.__app = None

    def update_args(self, override_args):
        """Update the argument used to invoke the application

        See help for `chain.Link` for details

        This calls the base class function then fills the parameters of the GtApp object
        """
        Link.update_args(self, override_args)
        dry_run = override_args.get('dry_run', False)
        if self.__app is None:
            self.__app = build_gtapp(self.appname, dry_run, **self.args)
#except:
#                raise ValueError("Failed to build link %s %s %s" %
#                                 (self.linkname, self.appname, self.args))
        else:
            update_gtapp(self.__app, **self.args)

    def get_gtapp(self):
        """Returns a `GTApp` object that will run this `Link` """
        return self.__app

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
        return run_gtapp(self.__app, stream, dry_run, **self.args)

    def command_template(self):
        """Build and return a string that can be used as a template invoking
        this chain from the command line.

        The actual command can be obtainted by using
        `self.command_template().format(**self.args)`
        """
        com_out = self.appname
        for key, val in self.args.items():
            if key in self._options:
                com_out += ' %s={%s}' % (key, key)
            else:
                com_out += ' %s=%s' % (key, val)
        return com_out

    def run_analysis(self, argv):
        """Implemented by sub-classes to run a particular analysis"""
        raise RuntimeError("run_analysis called for Gtlink type object")
