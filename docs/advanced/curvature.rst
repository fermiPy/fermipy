.. _curvature:

Curvature test
==============

The :py:meth:`~fermipy.gtanalysis.GTAnalysis.curvature` method
tests for spectral curvature (deviation from a power-law energy
spectrum) for a given source via likelihood ratio test.

The likelihood is maximized under three different spectral hypotheses for the source in question:

 * `PowerLaw <https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/source_models.html#PowerLaw>`_
 * `LogParabola <https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/source_models.html#LogParabola>`_, and
 * `PLSuperExpCutoff4 <https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/source_models.html#PLSuperExpCutoff4>`_.

For the first two models, all parameters except for the pivot energy are fit.
For the power law with super-exponential cutoff, the parameter ``Index2``
(also referred to as ``b``) is fixed to 0.6667 (the recommended value for pulsars
from `4FGL-DR3 <https://arxiv.org/abs/2201.11184>`_) by default.
The user may supply a different value of `Index2`
and/or allow its value to float during the likelihood fit. The latter is
only recommended for sources with high detection significance.

The likelihood ratios are calculated with the PowerLaw fit as the baseline,
e.g. :math:`TS_{LP} = -2 \left( \mathrm{ln}\left(L_{PL}\right) -  \mathrm{ln}\left(L_{LP}\right) \right)`.
The ``LogParabola`` and ``PLSuperExpCutoff4`` model with fixed ``Index2``
have three free parameters each compared to two for the baseline model,
and both models contain the baseline model as a special case.
In the absence of spectral curvature, the likelihood ratios defined as above
should thus follow a chi2 distribution with one degree of freedom.

The ``PLSuperExpCutoff4`` model with free ``Index2`` has four free parameters
and its associated likelihood ratio should thus follow a chi2 distribution
with two degrees of freedom in the absence of curvature.

Note that the ``PLSuperExpCutoff4`` model with free ``Index2``
contains both the ``LogParabola`` and ``PLSuperExpCutoff4`` model with
fixed ``Index2`` as special cases. ``PLSuperExpCutoff4`` with ``Index2=0``
is equivalent to ``LogParabola``.

.. warning::

   The likelihood fits within the :py:meth:`~fermipy.gtanalysis.GTAnalysis.curvature` function sometimes fail to find the global minima. This can lead to an over- or under-estimate of the curvature TS. If in doubt, please perform dedicated fits with your own starting values and check fit quality.


Usage
-----

The :py:meth:`~fermipy.gtanalysis.GTAnalysis.cuvature` method is executed
by passing the name of a source in the ROI as a single argument.
Additional keyword arguments can also be provided to override the
default configuration of the method:

.. code-block:: python
   
   # Run curvature test with default settings
   sed = gta.curvature('sourceA')

   # Override the value for Index2 and free said parameter
   sed = gta.sed('sourceA', Index2=0.2, free_Index2=True)

The return value of :py:meth:`~fermipy.gtanalysis.GTAnalysis.curvature` is a
dictionary with the results of the analysis. The contents of the output dictionary
are documented in :ref:`curvature_dict`.
   
.. _curvature_dict:
            
Curvature Dictionary
--------------------
   
The following table describes the contents of the
:py:meth:`~fermipy.gtanalysis.GTAnalysis.curvature` output dictionary:

.. csv-table:: *curvature* Output Dictionary
   :header:    Key, Type, Description
   :file: ../config/curvature_output.csv
   :delim: tab
   :widths: 10,10,80

.. _config_curvature:

Configuration
-------------

The default configuration of the method is controlled with the
Curvature section of the configuration file.  The default
configuration can be overriden by passing the option as a *kwargs*
argument to the method.

.. csv-table:: *curvature* Options
   :header:    Option, Default, Description
   :file: ../config/curvature.csv
   :delim: tab
   :widths: 10,10,80
            
Reference/API
-------------

.. automethod:: fermipy.gtanalysis.GTAnalysis.curvature
   :noindex:


