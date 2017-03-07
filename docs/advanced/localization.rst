.. _localization:

Source Localization
===================

The :py:meth:`~fermipy.gtanalysis.GTAnalysis.localize` method can be
used to spatially localize a source.  Localization is performed by
scanning the 2D likelihood surface in a local patch around the nominal
source position.  The current implementation of the localization
analysis proceeds in two steps:

* **TS Map Scan**: Obtain a rough estimate of the source position by
  generating a fast TS Map of the region using the
  `~fermipy.gtanalysis.GTAnalysis.tsmap` method.  In this step all background
  parameters are fixed to their nominal values.

* **Likelihood Scan**: Refine the position of the source by performing a
  scan of the likelihood surface in a box centered on the best-fit
  position found with the TS Map method.  The size of the search
  region is set to encompass the 99% positional uncertainty contour.
  This method uses a full likelihood fit at each point in the
  likelihood scan and will re-fit all free parameters of the model.

The localization method is executed by passing the name of a source as
its argument.  The method returns a python dictionary with the best-fit source
position and localization errors and also saves this information to
the *localization* dictionary of the `~fermipy.roi_model.Source`
object.

.. code-block:: python
   
   >>> loc = gta.localize('3FGL J1722.7+6104')
   >>> print(loc['ra'],loc['dec'],loc['r68'],loc['r95'])
   (260.53164555483784, 61.04493807148745, 0.14384100879403075, 0.23213050350030126)

By default the method will save a plot to the working directory with a
visualization of the localization contours.  The black and red
contours show the uncertainty ellipse derived from the TS Map and
likelihood scan, respectively.

.. image:: 3fgl_j1722.7+6104_localize.png
   :width: 75%
   :align: center
   
The default configuration for the localization analysis can be
overriden by supplying one or more *kwargs*:
   
.. code-block:: python
   
   # Localize the source and update its properties in the model
   # with the localized position
   >>> o = gta.extension('sourceA', update=True)

The localization method will not profile over any background
parameters that were free when the method was executed.  One can free
background parameters with the ``free_background`` parameter:

.. code-block:: python
   
   # Free a nearby source that may be be partially degenerate with the
   # source of interest
   gta.free_norm('sourceB')
   gta.localize('sourceA', free_background=True)

The contents of the output dictionary are described in the following table:

.. csv-table:: *localize* Output
   :header:    Key, Type, Description
   :file: ../config/localize_output.csv
   :delim: tab
   :widths: 10,10,80


Reference/API
-------------

.. automethod:: fermipy.gtanalysis.GTAnalysis.localize
   :noindex:


