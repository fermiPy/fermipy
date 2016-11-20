.. _sensitivity:

Sensitivity Tools
-----------------

The ``fermipy-flux-sensitivity`` script can be used to calculate the
LAT detection threshold versus energy.  Inputs are the galactic
diffuse model and the livetime cube of the observation epoch.  The
``obs_time_yr`` option can be used to rescale the livetime cube to a
shorter or longer observation time.

.. code-block:: bash

   $ fermipy-flux-sensitivity --glon=30 --glat=30 --output=flux.fits \
   --ltcube=ltcube.fits --galdiff=gll_iem_v06.fits --event_class=P8R2_SOURCE_V6

If no livetime cube is provided then the sensitivity will be computed
assuming an "ideal" survey-mode operation with uniform exposure over
the whole sky and no Earth obscuration or deadtime.  By default the
flux sensitivity will be calculated for a TS threshold of 25 and at
least 3 counts.
   
The output FITS file contains a table with the flux threshold in each
energy bin.
   
.. code-block:: python

   from astropy.table import Table
   tab = Table.read('flux.fits')
   print(tab['e_min'], tab['e_max'], tab['flux'])
