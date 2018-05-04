# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities to chain together a series of ScienceTools apps
"""
from __future__ import absolute_import, division, print_function


from fermipy.jobs.link import Link


class AppLink(Link):
    """A wrapper for a single fermipy application

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
        super(AppLink, self).__init__(**kwargs)

    def update_args(self, override_args):
        """Update the argument used to invoke the application

        See help for `chain.Link` for details

        This calls the base class function then fills the parameters of the GtApp object
        """
        Link.update_args(self, override_args)

    def run_analysis(self, argv):
        """Implemented by sub-classes to run a particular analysis"""
        self.run_argparser(argv)
        self.run()
