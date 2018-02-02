.. _lightcurve:

Light Curves
============

:py:meth:`~fermipy.gtanalysis.GTAnalysis.lightcurve` fits the
charateristics of a source (flux, TS, etc.) in a sequence of time
bins.  This method uses the data selection and model of a baseline
analysis (e.g. the full mission) and is therefore restricted to
analyzing time bins that are encompassed by the time selection of the
baseline analysis.  In general when using this method it is
recommended to use a baseline time selection of at least several years
or more to ensure the best characterization of background sources in
the ROI.

When fitting a time bin the method will initialize the model to the
current parameters of the baseline analysis.  The parameters to be
refit in each time bin may be controlled with ``free_background``,
``free_sources``, ``free_radius``, ``free_params``, and
``shape_ts_threshold`` options.


Examples
--------

.. code-block:: python
   
   # Generate a lightcurve with two bins
   lc = gta.lightcurve('sourceA', nbins=2)

   # Generate a lightcurve with 1-week binning
   lc = gta.lightcurve('sourceA', binsz=86400.*7.0)

   # Generate a lightcurve freeing sources within 3 deg of the source
   # of interest
   lc = gta.lightcurve('sourceA', binsz=86400.*7.0, free_radius=3.0)
   
   # Generate a lightcurve with arbitrary MET binning
   lc = gta.lightcurve('sourceA', time_bins=[239557414,242187214,250076614],
                       free_radius=3.0)
   

Optimizing Computation Speed
----------------------------

By default the ``lightcurve`` method will run an end-to-end analysis
in each time bin using the same processing steps as the baseline analysis.
Depending on the data selection and ROI size each time bin may take
10-15 minutes to process.  There are several options which can be used
to reduce the lightcurve computation time.  The ``multithread`` option splits the
analysis of time bins across multiple cores:

.. code-block:: python
   
   # Split lightcurve across all available cores
   lc = gta.lightcurve('sourceA', nbins=2, multithread=True)
   
   # split lightcurve across 2 cores
   lc = gta.lightcurve('sourceA', nbins=2, multithread=True, nthread=2)

Note that when using the ``multithread`` option in a computing cluster
environment one should reserve the appropriate number of cores when
submitting the job.
   
The ``use_scaled_srcmap`` option generates an approximate source map
for each time bin by scaling the source map of the baseline analysis
by the relative exposure.  

.. code-block:: python
   
   # Enable scaled source map
   lc = gta.lightcurve('sourceA', nbins=2, use_scaled_srcmap=True)   

Enabling this option can speed up the lightcurve calculation by at
least a factor of 2 or 3 at the cost of slightly reduced accuracy in the
model evaluation.  For point-source analysis on medium to long
timescales (days to years) the additional systematic uncertainty
incurred by using scaled source maps should be no more than 1-2%.  For
analysis of diffuse sources or short time scales (< day) one should
verify the systematic uncertainty is less than the systematic
uncertainty of the IRFs.

   
.. _lightcurve_dict:
            
Output
------
   
The following tables describe the contents of the method output:

.. csv-table:: *lightcurve* Output 
   :header:    Key, Type, Description
   :file: ../config/lightcurve_output.csv
   :delim: tab
   :widths: 10,10,80

.. csv-table:: *lightcurve* Source Output 
   :header:    Key, Type, Description
   :file: ../config/source_flux_output.csv
   :delim: tab
   :widths: 10,10,80
            

Reference/API
-------------

.. automethod:: fermipy.gtanalysis.GTAnalysis.lightcurve
   :noindex:

