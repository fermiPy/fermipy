# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

from fermipy.jobs.file_archive import FileFlags
from fermipy.jobs.link import Link
from fermipy.jobs.chain import Chain
from fermipy.jobs.app_link import AppLink
from fermipy.jobs import defaults

def test_applink():
    """ Test that we can create `Link` class """

    class Link_FermipyCoadd(AppLink):
        """Small wrapper to run fermipy-coadd """
        
        appname = 'fermipy-coadd'
        linkname_default = 'coadd'
        usage = '%s [options]' % (appname)
        description = "Link to run %s" % (appname)
        
        default_options = dict(args=([], "List of input files", list),
                               output=(None, "Output file", str))
        default_file_args = dict(args=FileFlags.input_mask,
                                 output=FileFlags.output_mask)

    link = Link_FermipyCoadd()
    formatted_command = link.formatted_command()
    print (formatted_command)
    #assert formatted_command == 'gtsrcmaps irfs=CALDB expcube=None cmap=None srcmdl=None outfile=None bexpmap=None'
  

def test_chain():
    """ Test that we can create `Chain` class """

    class Link_FermipyCoadd(AppLink):
        """Small wrapper to run fermipy-coadd """
        
        appname = 'fermipy-coadd'
        linkname_default = 'coadd'
        usage = '%s [options]' % (appname)
        description = "Link to run %s" % (appname)
        
        default_options = dict(args=([], "List of input files", list),
                               output=(None, "Output file", str))
        default_file_args = dict(args=FileFlags.input_mask,
                                 output=FileFlags.output_mask)

    class Link_FermipyCoadd_v2(AppLink):
        """Small wrapper to run fermipy-coadd """
        
        appname = 'fermipy-coadd-v2'
        linkname_default = 'coadd-v2'
        usage = '%s [options]' % (appname)
        description = "Link to run %s" % (appname)
        
        default_options = dict(args=([], "List of input files", list),
                               output=(None, "Output file", str))
        default_file_args = dict(args=FileFlags.input_mask,
                                 output=FileFlags.output_mask)


    class TestChain(Chain):
        """Small test class """
        appname = 'dmpipe-pipeline-data'
        linkname_default = 'pipeline-data'
        usage = '%s [options]' % (appname)
        description = 'Data analysis pipeline'
        
        default_options = dict(dry_run=defaults.common['dry_run'])

        def _map_arguments(self, input_dict):
            self._load_link_args('coadd',
                                 Link_FermipyCoadd,
                                 output="dummy.fits")
            self._load_link_args('coadd-v2',
                                 Link_FermipyCoadd_v2,
                                 output="dummy2.fits")

    chain = TestChain()
    assert chain


if __name__ == '__main__':
    test_applink()
    test_chain()
