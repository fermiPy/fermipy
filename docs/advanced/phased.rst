.. _phased:

Phased Analysis
===============

Fermipy provides several options to support analysis with selections
on pulsar phase.  The following examples assume that you already have
a phased FT1 file that contains a PULSE_PHASE column with the pulsar
phase for each event.

The following examples illustrates the settings for the
:ref:`config_gtlike` and :ref:`config_selection` sections of the
configuration file that would be used for a single-component ON- or
OFF-phase analysis:

.. code-block:: yaml
   
   selection :
     emin : 100
     emax : 316227.76
     zmax    : 90
     evclass : 128
     evtype  : 3
     tmin    : 239557414
     tmax    : 428903014
     target : '3FGL J0534.5+2201p'
     phasemin : 0.68
     phasemax : 1.00

   gtlike :
     edisp : True
     irfs : 'P8R2_SOURCE_V6'
     edisp_disable : ['isodiff','galdiff']
     expscale : 0.32

The ``gtlike.expscale`` parameter defines the correction that should
be applied to the nominal exposure to account for the phase selection
defined by ``selection.phasemin`` and ``selection.phasemax``.
Normally this should be set to the size of the phase selection
interval.


To perform a joint analysis of multiple phase selections you can use
the :ref:`config_components` section to define separate ON- and OFF-phase
components:

.. code-block:: yaml

   components:
     - selection : {phasemin : 0.68, phasemax: 1.0}
       gtlike    : {expscale : 0.32, src_expscale : {'3FGL J0534.5+2201p':0.0}}
     - selection : {phasemin : 0.0 , phasemax: 0.68}
       gtlike    : {expscale : 0.68, src_expscale : {'3FGL J0534.5+2201p':1.0}}

The ``src_expscale`` parameter can be used to define an exposure
correction for indvidual sources.  In this example it is used to zero
the pulsar component for the OFF-phase selection.
