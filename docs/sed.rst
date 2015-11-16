.. _sed:

SED Analysis
============

The :py:meth:`~fermipy.gtanalysis.GTAnalysis.sed` method fits of the
normalization of a source in a sequence of energy bins to derive a
spectral energy distribution (SED).  Each bin is fit independently
using a power-law source with a fixed index while keeping all other
components of the model fixed.  This index can be set to
a fixed value (with the *bin_index* parameter) or varied over the
energy range according to the local slope of the global spectral model
(with the *use_local_index* parameter).


The configuration of :py:meth:`~fermipy.gtanalysis.GTAnalysis.sed` can
be controlled with the *sed* block of the configuration file:

.. code-block:: yaml
   
   sed:
     bin_index : 2.0
     use_local_index : False
     
At runtime the default settings for the SED analysis can be
overriden by supplying one or more *kwargs* when executing
:py:meth:`~fermipy.gtanalysis.GTAnalysis.sed`:

.. code-block:: python
   
   # Run analysis with default energy binning
   gta.sed('sourceA')

   # Override the energy binning for the SED
   gta.sed('sourceA',energies=[2.0,2.5,3.0,3.5,4.0,4.5,5.0])

By default the method will use the energy bins of the underlying
analysis.  The *energies* keyword argument can be used to override
this default binning with the restriction that the SED energy bins
most align with the analysis bins.

The results of the SED analysis are written to a dictionary
which is the return argument of the SED method.  This dictionary
is also written to the *sed* dictionary of the corresponding
source and is also contained in the output file generated
with :py:meth:`~fermipy.gtanalysis.GTAnalysis.write_roi`.
   
.. code-block:: python
   
   # Get the sed results from the return argument
   sed = gta.sed('sourceA')

   # Get the sed results from the source object
   sed = gta.roi['sourceA']

The contents of the output dictionary are documented below:

============= =================================================================
Key           Description
============= =================================================================
emin          Lower edges of SED energy bins (log10(E/MeV)).
emax          Upper edges of SED energy bins (log10(E/MeV)).
ecenter       Centers of SED energy bins (log10(E/MeV)).
flux          Flux in each bin (cm^{-2} s^{-1}).
eflux         Energy flux in each bin (MeV cm^{-2} s^{-1}).
dfde          Differential flux in each bin (MeV^{-1} cm^{-2} s^{-1}).
e2dfde        E^2 x the differential flux in each bin (MeV^{-1} cm^{-2} s^{-1}).
dfde_err      1-sigma error on dfde estimated from likelihood curvature.
dfde_err_lo   Lower 1-sigma error on dfde estimated from the profile likelihood (MINOS errors).
dfde_err_hi   Upper 1-sigma error on dfde estimated from the profile likelihood (MINOS errors).
dfde_ul95     95% CL upper limit on dfde estimated from the profile likelihood (MINOS errors).
e2dfde_err    1-sigma error on e2dfde estimated from likelihood curvature.
e2dfde_err_lo Lower 1-sigma error on e2dfde estimated from the profile likelihood (MINOS errors).
e2dfde_err_hi Upper 1-sigma error on e2dfde estimated from the profile likelihood (MINOS errors).
e2dfde_ul95   95% CL upper limit on e2dfde estimated from the profile likelihood (MINOS errors).
ts            Test statistic.
Npred         Number of model counts.
fit_quality   Fit quality parameter.
index         Spectral index of the power-law model used to fit this bin.
lnlprofile    Likelihood scan for each energy bin.
config        Copy of the input parameters to this method.
============= =================================================================


Reference/API
-------------

.. automethod:: fermipy.gtanalysis.GTAnalysis.sed
   :noindex:


