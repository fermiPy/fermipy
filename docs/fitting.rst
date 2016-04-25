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

   >>> gta.free_sources(distance=2.0,pars='norm')
   >>> gta.print_params(True)
    idx parname                  value     error       min       max     scale free
   --------------------------------------------------------------------------------
   3FGL J1104.4+3812
     18 Prefactor                 1.77         0     1e-05       100     1e-11    *
   3FGL J1109.6+3734
     24 Prefactor                 0.33         0     1e-05       100     1e-14    *
   galdiff
     52 Prefactor                    1         0       0.1        10         1    *
   isodiff
     55 Normalization                1         0     0.001     1e+03         1    *                
   >>> o = gta.fit()
   2016-04-19 14:07:55 INFO     GTAnalysis.fit(): Starting fit.
   2016-04-19 14:08:56 INFO     GTAnalysis.fit(): Fit returned successfully.
   2016-04-19 14:08:56 INFO     GTAnalysis.fit(): Fit Quality: 3 LogLike:   -77279.869 DeltaLogLike:      501.128
   >>> gta.print_params(True)
   2016-04-19 14:10:02 INFO     GTAnalysis.print_params(): 
    idx parname                  value     error       min       max     scale free
   --------------------------------------------------------------------------------
   3FGL J1104.4+3812
     18 Prefactor                 2.13    0.0161     1e-05       100     1e-11    *
   3FGL J1109.6+3734
     24 Prefactor                0.342    0.0904     1e-05       100     1e-14    *
   galdiff
     52 Prefactor                0.897    0.0231       0.1        10         1    *
   isodiff
     55 Normalization             1.15     0.016     0.001     1e+03         1    *

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
      
