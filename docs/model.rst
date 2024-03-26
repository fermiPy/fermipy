.. _model:

Customizing the Model
=====================

The ROIModel class is responsible for managing the source and diffuse
components in the ROI.  Configuration of the model is controlled with
the :ref:`config_model` block of YAML configuration file.

Configuring Diffuse Components
------------------------------

The simplest configuration uses a single file for the galactic and
isotropic diffuse components.  By default the galactic diffuse and
isotropic components will be named *galdiff* and *isodiff*
respectively.  An alias for each component will also be created with
the name of the mapcube or file spectrum.  For instance the galactic
diffuse can be referred to as *galdiff* or *gll_iem_v07* in the
following example.

.. code-block:: yaml
   
   model:
     src_roiwidth : 10.0
     galdiff  : '$FERMI_DIR/refdata/fermi/galdiffuse/gll_iem_v07.fits'
     isodiff  : '$FERMI_DIR/refdata/fermi/galdiffuse/iso_P8R3_SOURCE_V3_v1.txt'
     catalogs : ['4FGL-DR3']

To define two or more galactic diffuse components you can optionally define
the *galdiff* and *isodiff* parameters as lists.  A separate
component will be generated for each element in the list with the name
*galdiffXX* or *isodiffXX* where *XX* is an integer position in the
list.

.. code-block:: yaml
   
   model:
     galdiff  : 
       - '$FERMI_DIFFUSE_DIR/diffuse_component0.fits'
       - '$FERMI_DIFFUSE_DIR/diffuse_component1.fits'

To explicitly set the name of a component you can define any element
as a dictionary containing *name* and *file* fields:

.. code-block:: yaml
   
   model:
     galdiff  : 
       - { 'name' : 'component0', 'file' : '$FERMI_DIFFUSE_DIR/diffuse_component0.fits' }
       - { 'name' : 'component1', 'file' : '$FERMI_DIFFUSE_DIR/diffuse_component1.fits' }

Configuring Source Components
-----------------------------

The list of sources for inclusion in the ROI model is set by defining
a list of catalogs with the *catalogs* parameter.  Catalog files can
be in either XML or FITS format.  Sources from the catalogs in this
list that satisfy either the *src_roiwidth* or *src_radius* selections
are added to the ROI model.  If a source is defined in multiple
catalogs the source definition from the last file in the catalogs list
takes precedence.

.. code-block:: yaml
   
   model:
   
     src_radius: 5.0
     src_roiwidth: 10.0
     catalogs : 
       - 'gll_psc_v31.fit'
       - 'extra_sources.xml'

Individual sources can also be defined within the configuration file
with the *sources* parameter.  This parameter contains a list of
dictionaries that defines the spatial and spectral parameters of each
source.  The keys of the source dictionary map to the spectral and
spatial source properties as they would be defined in the XML model
file.

.. code-block:: yaml
   
   model:
     sources  : 
       - { name: 'SourceA', glon : 120.0, glat : -3.0, 
        SpectrumType : 'PowerLaw', Index : 2.0, Scale : 1000, Prefactor : !!float 1e-11, 
        SpatialModel: 'PointSource' }
       - { name: 'SourceB', glon : 122.0, glat : -3.0,
        SpectrumType : 'LogParabola', norm : !!float 1E-11, Scale : 1000, beta : 0.0,
        SpatialModel: 'PointSource' }

For parameters defined as scalars, the scale and value properties will
be assigned automatically from the input value.  To set these manually
a parameter can also be initialized with a dictionary that explicitly
sets the value and scale properties:

.. code-block:: yaml
   
   model:
     sources  : 
       - { name: 'SourceA', glon : 120.0, glat : -3.0, 
           SpectrumType : 'PowerLaw', Index : 2.0, Scale : 1000,
           Prefactor : { value : 1.0, scale : !!float 1e-11, free : '0' }, 
           SpatialModel: 'PointSource' }

Spatial Models
--------------

Fermipy supports four spatial models which are defined with the
``SpatialModel`` property:

* PointSource : A point source (SkyDirFunction).
* RadialGaussian : A symmetric 2D Gaussian with width parameter 'Sigma'.
* RadialDisk : A symmetric 2D Disk with radius 'Radius'.
* SpatialMap : An arbitrary 2D shape with morphology defined by a FITS template.
  
The spatial extension of RadialDisk and RadialGaussian can be
controlled with the ``SpatialWidth`` parameter which sets the 68%
containment radius in degrees.  Note for ST releases prior to
11-01-01, RadialDisk and RadialGaussian sources will be represented
with the ``SpatialMap`` type.

.. code-block:: yaml
   
   model:
     sources  :
       - { name: 'PointSource', glon : 120.0, glat : 0.0, 
        SpectrumType : 'PowerLaw', Index : 2.0, Scale : 1000, Prefactor : !!float 1e-11, 
        SpatialModel: 'PointSource' }
       - { name: 'DiskSource', glon : 120.0, glat : 0.0, 
        SpectrumType : 'PowerLaw', Index : 2.0, Scale : 1000, Prefactor : !!float 1e-11, 
        SpatialModel: 'RadialDisk', SpatialWidth: 1.0 }
       - { name: 'GaussSource', glon : 120.0, glat : 0.0, 
        SpectrumType : 'PowerLaw', Index : 2.0, Scale : 1000, Prefactor : !!float 1e-11, 
        SpatialModel: 'RadialGaussian', SpatialWidth: 1.0 }
       - { name: 'MapSource', glon : 120.0, glat : 0.0, 
        SpectrumType : 'PowerLaw', Index : 2.0, Scale : 1000, Prefactor : !!float 1e-11, 
        SpatialModel: 'SpatialMap', Spatial_Filename : 'template.fits' }
        


Editing the Source List at Runtime
----------------------------------

.. tip::

   Many users chose to delete sources that are not signifcantly
   detected (e.g. ``TS<1`` and/or ``nPred<1``) from the model
   after the :py:meth:`~fermipy.gtanalysis.GTAnalysis.optimize` step.
   

The model can be manually editing at runtime with the
:py:meth:`~fermipy.gtanalysis.GTAnalysis.add_source` and
:py:meth:`~fermipy.gtanalysis.GTAnalysis.delete_source` methods.
Sources should be added after calling
:py:meth:`~fermipy.gtanalysis.GTAnalysis.setup` as shown in the
following example.

.. code-block:: python

   from fermipy.gtanalysis import GTAnalysis
           
   gta = GTAnalysis('config.yaml',logging={'verbosity' : 3})
   gta.setup()
   
   # Remove isodiff from the model
   gta.delete_source('isodiff')

   # Add SourceA to the model
   gta.add_source('SourceA',{ 'glon' : 120.0, 'glat' : -3.0, 
                   'SpectrumType' : 'PowerLaw', 'Index' : 2.0, 
		   'Scale' : 1000, 'Prefactor' : 1e-11, 
        	   'SpatialModel' : 'PointSource' })

   # Add SourceB to the model
   gta.add_source('SourceB',{ 'glon' : 121.0, 'glat' : -2.0, 
                    'SpectrumType' : 'PowerLaw', 'Index' : 2.0, 
		    'Scale' : 1000, 'Prefactor' : 1e-11, 
        	    'SpatialModel' : 'PointSource' })

Sources added after calling
:py:meth:`~fermipy.gtanalysis.GTAnalysis.setup` will be created
dynamically through the pyLikelihood object creation mechanism.  


Freeing and Fixing Parameters
-----------------------------

In addition to freeing and fixing parameters for a source or list
of sources as explained in
:ref:`quickstart.html#creating-an-analysis-script`, we can also
free and fix parameters by name using the
:py:meth:`~fermipy.gtanalysis.GTAnalysis.free_parameter` method.
For example:

.. code-block:: python

   gta.free_parameter(name="SourceA", par="Index", free=False)
   gta.free_parameter(name="SourceB", par="Prefactor", free=True)


Manual Setting of Parameters and Parameter Ranges
-------------------------------------------------

.. note::

   As in `fermitools <https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/source_models.html>`_,
   "The actual value of a given parameter that is used in the
   calculation is the value attribute multiplied by the scale
   attribute. The value attribute is what the optimizers see."

In rare cases, the user may want to access or change the current
value or allowed range of a given parameter. The function
:py:meth:`~fermipy.gtanalysis.GTAnalysis._get_param` returns a
dictionary with information about a given parameter such as its
value and allowed range. Parameters can also accessed via the ROI
dictionsary. For example, the following are equivalent:

.. code-block:: python

   indexA = gta._get_param(name="SourceA", par="Index")["value"]
   indexA = gta.roi["SourceA"].spectral_pars["Index"]["value"]


:py:meth:`~fermipy.gtanalysis.GTAnalysis.set_parameter` can be
used to set the value or range. Note that the ``bounds`` argument
is always unscaled. The following calls are equivalent:


.. code-block:: python

   gta.set_parameter("SourceA", "Prefactor", 1.236583491e-13, bounds=[0.01, 100], scale=1e-13, true_value=True)
   gta.set_parameter("SourceA", "Prefactor", 1.236583491,     bounds=[0.01, 100], scale=1e-13, true_value=False)

In either case, the allowed parameter values range from ``1e-15`` to ``1e-11``.

