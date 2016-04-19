.. _fitting:

############################
ROI Optimization and Fitting
############################

Source fitting with fermipy is generally performed with the
`~fermipy.gtanalysis.GTAnalysis.optimize` and
`~fermipy.gtanalysis.GTAnalysis.fit` methods.

Fitting
=======

`~fermipy.gtanalysis.GTAnalysis.fit` is a wrapper on the pyLikelihood
fit method and performs a likelihood fit of all free parameters of the
model.  This method can be used to manually optimize of the model by
calling it after freeing one or more source parameters.  The following
example demonstrates the commands that would be used to fit the
normalizations of all sources within 3 deg of the ROI center:

.. code-block:: python

   >>> gta.free_sources(distance=3.0,pars='norm')
   >>> gta.print_params(True)
                
   >>> o = gta.fit()

   >>> gta.print_params(True)

By default `~fermipy.gtanalysis.GTAnalysis.fit` will repeat the fit
until a fit quality of 3 is obtained.  After the fit returns all
sources with free parameters will have their properties (flux, TS,
NPred, etc.) updated in the `~fermipy.roi_mode.ROIModel` instance.
The return value of the method is a dictionary containing the
following diagnostic information about the fit:

.. csv-table:: *fit* Output Dictionary
   :header:    Key, Type, Description
   :file: config/fit_output.csv
   :delim: tab
   :widths: 10,10,80

The `~fermipy.gtanalysis.GTAnalysis.fit` also accepts keyword
arguments which can be used to configure its behavior at runtime:

.. code-block:: python
                
   >>> o = gta.fit(min_fit_quality=2,optimizer='NEWMINUIT',reoptimize=True)

Reference/API
-------------

.. automethod:: fermipy.gtanalysis.GTAnalysis.fit
   :noindex:
   

ROI Optimization
================
   
The `~fermipy.gtanalysis.GTAnalysis.optimize` method performs an
automatic optimization of the ROI by fitting all sources with an
iterative strategy. 

.. code-block:: python

   >>> o = gta.optimize()

It is generally good practice to run this method once at the start of
your analysis to ensure that all parameters are close to their global
likelihood maxima.

.. csv-table:: *optimization* Output Dictionary
   :header:    Key, Type, Description
   :file: config/roiopt_output.csv
   :delim: tab
   :widths: 10,10,80

   
Reference/API
-------------

.. automethod:: fermipy.gtanalysis.GTAnalysis.optimize
   :noindex:
      
