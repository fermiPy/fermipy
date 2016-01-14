.. _detection:

Source Detection
================

fermipy provides two methods for source detection that can be used to
investigate whether additional sources are present in the ROI as well
as evaluate the general fit quality of the model.  These methods are

* :py:meth:`~fermipy.gtanalysis.GTAnalysis.tsmap`: Evaluate the test
  statistic (TS) for a new source centered at each spatial bin in the
  ROI.  

* :py:meth:`~fermipy.gtanalysis.GTAnalysis.residmap`: Evaluate the
  significance of the difference between smoothed data and model maps
  (i.e. the residual) at each spatial bin in the ROI.

The function signatures and outputs of these two methods are similar and
they provide complementary information about the .

TS Map
------

:py:meth:`~fermipy.gtanalysis.GTAnalysis.tsmap` evaluates the TS for
an additional source at each point in the ROI.  Note that this
calculation does not re-fit parameters of background components.  One
or more test source models can be defined with the *models* argument:

.. code-block:: python
   
   # Test a Power-law source with Index=2.0
   models = {'Index' : 2.0}
   maps = gta.tsmap('fit1',models=models)

   # Test a Power-law source with Index=1.5, 2.0, and 2.5
   models=[{'Index' : 1.5},{'Index' : 2.0},{'Index' : 2.5}]
   maps = gta.tsmap('fit1',models=models)

The return argument is a *maps* dictionary containing `~femripy.utils.Map` object.  



.. |image0| image:: tsmap_sqrt_ts.png
   :width: 100%
   
.. |image1| image:: tsmap_npred.png
   :width: 100%

+-----------+-----------+
| Sqrt(TS)  | NPred     |
+===========+===========+
| |image0|  + |image1|  |
+-----------+-----------+




..   Test statistics map computed using `~gammapy.detect.compute_ts_map` for an
..   example Fermi dataset.
