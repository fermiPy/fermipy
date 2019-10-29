.. _merit_skimmer:

Merit Skimmer
=============

The ``fermipy-merit-skimmer`` script can be used to create skimmed
Merit files (either MC or data) and serves as a replacement for the
web-based merit skimmer tool.  The script accepts as input a sequence
of Merit file paths or lists of Merit file paths which can be either
local (nfs) or xrootd.

.. code-block:: bash
                
   $ fermipy-merit-skimmer merit_list.txt --output=merit.root --selection='FswGamFilter' \
   --aliases=aliases.yaml

   $ fermipy-merit-skimmer merit_list.txt --output=merit-clean.root \
   --selection='FswGamFilter && CLEAN' --aliases=EvtClassDefs_P8R2.xml

where ``merit_list.txt`` is a text file with one path per line.  The
``--selection`` option sets the selection that will be applied when
filtering events in each file.  The ``--output`` option sets the path to
the output merit file.  The ``--aliases`` option can be used to load an
alias file (set of key/value cut expression pairs).  This option
can accept either a YAML alias file or an XML event class definition
file.  The following illustrates the YAML alias file format:

.. code-block:: yaml

   FswGamFilter : FswGamState == 0
   TracksCutFilter : FswGamState == 0 && TkrNumTracks > 0
   CalEnergyFilter : FswGamState == 0 && CalEnergyRaw > 0

One can restrict the set of output branches with the ``--branches``
option which accepts an alias file or a YAML file containing a list of
branch names.  In the former case all branch names used in the given
alias set will be extracted and added to the list of output branches.

.. code-block:: bash
                
   $ fermipy-merit-skimmer merit_list.txt --output=merit.root --selection='FswGamFilter' \
   --aliases=EvtClassDefs_P8R2.xml --branches=EvtClassDefs_P8R2.xml

One can split the skimming task into separate batch jobs by running
with the ``--batch`` option.  This will subdivide task into N jobs
when N is the number of files in the list divided by
``--files_per_job``.  The name of the output ROOT file of each job
will be appended with the index of the job in the sequence
(e.g. skim_000.root, skim_001.root, etc.).  The ``--time`` and
``--resources`` options can be used to set the LSF wallclock time and
resource flags.

.. code-block:: bash
                
   $ fermipy-merit-skimmer merit_list.txt --output=merit.root --selection='SOURCE && FRONT' \
   --branches=EvtClassDefs_P8R2.xml --files_per_job=1000 --batch --aliases=EvtClassDefs_P8R2.xml

When skimming MC files it can be useful to extract the ``jobinfo`` for
tracking the number of thrown events.  The ``--extra_trees`` option
can be used to copy one or more trees to the output file in addition
to the Merit Tuple:

.. code-block:: bash
                
   $ fermipy-merit-skimmer merit_list.txt --output=merit.root --extra_trees=jobinfo
