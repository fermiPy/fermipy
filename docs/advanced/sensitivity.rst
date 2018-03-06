.. _sensitivity:

Sensitivity Tools
-----------------

The ``fermipy-flux-sensitivity`` script calculates the LAT flux
threshold for a gamma-ray source in bins of energy (differential
sensitivity) and integrated over the full LAT energy range (integral
sensitivity).  The source flux threshold is the flux at which the
median TS of a source (twice the likelihood ratio of the best-fit model with and
without the source) equals a certain value.  Primary inputs to this script
are the livetime cube (output of ``gtltcube``) and the model cube for the galactic diffuse
background.  The ``obs_time_yr`` option can be used to rescale the
livetime cube to a shorter or longer observation time.

.. code-block:: bash

   $ fermipy-flux-sensitivity --glon=30 --glat=30 --output=lat_sensitivity.fits \
   --ltcube=ltcube.fits --galdiff=gll_iem_v06.fits --event_class=P8R2_SOURCE_V6 \
   --ts_thresh=25.0 --min_counts=10.0

If no livetime cube is provided then the sensitivity will be computed
assuming an "ideal" survey-mode operation with uniform exposure over
the whole sky and no Earth obscuration or deadtime.  By default the
flux sensitivity will be calculated for a TS threshold of 25 and at
least 3 counts.

A map of sensitivity with WCS or HEALPix pixelization can be generated
by setting the ``map_type`` argument to either ``wcs`` or ``hpx``:

.. code-block:: bash

   # Generate a WCS sensitivity map of 50 x 50 deg centered at (glon,glat) = (30,30) 
   $ fermipy-flux-sensitivity  --glon=30 --glat=30 --output=lat_sensitivity_map.fits \
   --ltcube=ltcube.fits --galdiff=gll_iem_v06.fits --event_class=P8R2_SOURCE_V6 \
   --map_type=wcs --wcs_npix=100 --wcs_cdelt=0.5 --wcs_proj=AIT

   # Generate a HPX sensitivity map of nside=16
   $ fermipy-flux-sensitivity  --output=lat_sensitivity_map.fits \
   --ltcube=ltcube.fits --galdiff=gll_iem_v06.fits --event_class=P8R2_SOURCE_V6 \
   --map_type=hpx --hpx_nside=16

The integral and differential sensitivity maps will be written to the
``MAP_INT_FLUX`` and ``MAP_DIFF_FLUX`` extensions respectively.

By default the flux sensitivity will be computed for a point-source
morphology.  The assumed source morphology can be changed with the
``spatial_model`` and ``spatial_size`` parameters:

It is possible to choose among PowerLaw, LogParabola and
PLSuperExpCutoff SED shapes using the option ``sedshape``.

.. code-block:: bash

   # Generate the sensitivity to a source with a 2D gaussian morphology
   # and a 68% containment radius of 1 deg located at longitude 30deg and 
   # latitude 30 deg and with a PLSuperExpCutoff SED with index 2.0 and
   # cutoff energy 10 GeV
   $ fermipy-flux-sensitivity --output=lat_sensitivity_map.fits \
   --ltcube=ltcube.fits --galdiff=gll_iem_v06.fits --event_class=P8R2_SOURCE_V6 \
   --spatial_model=RadialGaussian --spatial_size=1.0 --glon=30 --glat=30
   --sedshape=PLSuperExpCutoff --index=2.0 --cutoff=1e4
 
   # Generate the sensitivity map in healpix with nside 128 of a point source with 
   # LogParabola SED and with spectral index 2.0 and curvature index beta=0.50
   # between 1 and 10 GeV
   $ fermipy-flux-sensitivity --output=lat_sensitivity_map.fits \
   --ltcube=ltcube.fits --galdiff=gll_iem_v06.fits --event_class=P8R2_SOURCE_V6 \
   --spatial_model=PointSource --sedshape=LogParabola --index=2.0 --beta=0.50 \
   --hpx_nside=128 --map_type=hpx --emin=1000 --emax=10000
   
The output FITS file set with the ``output`` option contains the
following tables.  Note that ``MAP`` tables are only generated when
the ``map_type`` argument is set.

* ``DIFF_FLUX`` : Differential flux sensitivity for a gamma-ray source
  at sky positition set by ``glon`` and ``glat``.
* ``INT_FLUX`` : Integral flux sensitivity evaluated for PowerLaw
  sources with spectral indices between 1.0 and 5.0 at sky positition
  set by ``glon`` and ``glat``.  Columns starting with ``ebin`` contain the source amplitude vs. energy bin.
* ``MAP_DIFF_FLUX`` : Sky cube with differential flux threshold
  vs. sky position and energy.  
* ``MAP_DIFF_NPRED`` : Sky cube with counts amplitude
  (NPred) of a source at the detection threshold vs. position and energy. 
* ``MAP_INT_FLUX`` : Sky map with integral flux threshold vs. sky
  position.  Integral sensitivity will be computed for a PowerLaw
  source with index equal to the ``index`` parameter.
* ``MAP_INT_NPRED`` : Sky map with counts amplitude (NPred)
  of a source at the detection threshold vs. sky position.
  
The output file can be read using the `~astropy.table.Table` module:
  
.. code-block:: python

   from astropy.table import Table
   
   tab = Table.read('lat_sensitivity.fits','DIFF_FLUX')
   print(tab['e_min'], tab['e_max'], tab['flux'])
   tab = Table.read('lat_sensitivity.fits','INT_FLUX')
   print(tab['index'], tab['flux'])
