.. _output:

Output File
===========

The current state of the ROI can be written at any point by calling
`~fermipy.gtanalysis.GTAnalysis.write_roi`.  

.. code-block:: python

   >>> gta.write_roi('output.npy')

The output file will contain all information about the state of the
ROI as calculated up to that point in the analysis including model
parameters and measured source characteristics (flux, TS, NPred).  An
XML model file will also be saved for each analysis component.
   
The output file can be read with `~numpy.load`:

.. code-block:: python

   >>> o = np.load('output.npy').flat[0]
   >>> print(o.keys())
   ['roi', 'config', 'sources','version']
   
The output file is organized in four top-level of dictionaries:

.. csv-table:: File Dictionary
   :header:    Key, Type, Description
   :file: config/file_output.csv
   :delim: tab
   :widths: 10,10,80

ROI Dictionary
--------------
            
Source Dictionary
-----------------

The ``sources`` dictionary contains one element per source keyed to the
source name.  The following table lists the elements of the source
dictionary and their descriptions.

.. csv-table:: Source Dictionary
   :header:    Key, Type, Description
   :file: config/source_output.csv
   :delim: tab
   :widths: 10,10,80


