.. _model:

Customizing the Model
=====================

The ROIModel class is responsible for managing the source and diffuse
components in the ROI.  Configuration of the model is controlled with
the *model* block of YAML configuration file.

The simplest configuration uses a single file for the galactic
diffuse.  By default the galactic diffuse and isotropic model
components will be named *galdiff* and *isodiff* respectively.  An
alias for each component will also be created with the name of the
mapcube or file spectrum.  For instance the galactic diffuse can be
referred to as *galdiff* or *gll_iem_v06* in the following example.

.. code-block:: yaml
   
   model:
     src_roiwidth : 10.0
     galdiff  : '$FERMI_DIFFUSE_DIR/gll_iem_v06.fits'
     isodiff  : '$FERMI_DIFFUSE_DIR/isotropic_source_4years_P8V3.txt'
     catalogs : ['gll_psc_v14.fit']

To define two or more galactic diffuse components you can define
either the *galdiff* and *isodiff* parameters as lists.  A separate
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
       - { 'name' : 'component0' : 'file' : '$FERMI_DIFFUSE_DIR/diffuse_component0.fits' }
       - { 'name' : 'component0' : 'file' : '$FERMI_DIFFUSE_DIR/diffuse_component0.fits' }

Additional sources can be defined with the *sources* block:

.. code-block:: yaml
   
   model:
     sources  : 
       - { name: 'SourceA', glon : 120.0, glat : -3.0, 
        SpectrumType : 'PowerLaw', Index : 2.0, Scale : 1000, Prefactor : !!float 1e-11, 
        SpatialType: 'PointSource' }
       - { name: 'SourceB', glon : 120.0, glat : 0.0, 
        SpectrumType : 'PowerLaw', Index : 2.0, Scale : 1000, Prefactor : !!float 1e-11, 
        SpatialType: 'DiskSource', SpatialWidth: 1.0 }
       - { name: 'SourceC', glon : 122.0, glat : -3.0,
        SpectrumType : 'LogParabola', norm : !!float 1E-11, Scale : 1000, beta : 0.0,
        SpatialType: 'PointSource' }


Editing the Model at Runtime
----------------------------

The model can be manually editing at runtime by adding or removing
sources before calling the
:py:meth:`~fermipy.gtanalysis.GTAnalysis.setup` method.


.. code-block:: python

   from fermipy.gtanalysis import GTAnalysis
           
   gta = GTAnalysis('config.yaml',logging={'verbosity' : 3})

   # Remove SourceA from the model
   gta.delete_source('SourceA')

   # Add SourceB to the model
   gta.add_source({ 'name': 'SourceA', 'glon' : 120.0, 'glat' : -3.0, 
                     SpectrumType : 'PowerLaw', Index : 2.0, 
		     Scale : 1000, Prefactor : !!float 1e-11, 
        	     SpatialType: 'PointSource' })

   gta.setup()
