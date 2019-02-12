.. _fermipy_diffuse:

fermipy.diffuse subpackage
==========================

The fermipy.diffuse sub-package is a collection of standalone
utilities that allow the user to parallelize the data and template
preparation for all-sky diffuse analysis.


Overview
--------

This package implements an analysis pipeline prepare data and
templates for analysis.  This involves a lot of bookkeeping and loops
over various things.It is probably easiest to first describe this with
a bit of pseudo-code that represents the various analysis steps.


The various loop variables are:

* Input data files
  
    For practical reasons, the input photon event files (FT1) files
    are split into monthly files.   In the binning step of the
    analysis we loop over those files.

* binning components

    We split the data into several "binning components" and make
    seperate binned counts maps for each components.    A binning
    component is defined by energy range and data sub-selection (such as
    PSF event type and zenith angle cuts).
    
   
* Diffuse model components

    The set of all the model components that represent diffuse
    emission, such as contributions for cosmic ray interactions with 


* Catalog model components

    The set of all the catalog sources (both point sources and
    extended source), merged into a few distinct contributions. 

  
* Diffuse emission model definitions

    A set of user defined models that merge the various model
    components with specific spectral models.



    
.. code-block:: python

    # Data Binning, prepare the analysis directories and precompute the DM spectra		

    # First we loop over all the input input files and split up the
    # data by binning component and bin the data using the command
    # fermipy-split-and-bin-sg, which is equivalent to:
    for file in input_data_files:        
        fermipy-split-and-bin(file)

    # Then we loop over the binning components and coadd the binned
    # data from all the input files using the command
    # fermipy-coadd-split-sf, which is equivalent to:
    for comp in binning_components:
        fermipy-coadd-split(comp)
	
    # We also loop over the binning components and compute the
    # exposure maps for each binning component using the command
    # fermipy-gtexpcube2-sg, which is equivalent to:
    for comp in binned_components:
	gtexpcube2(comp)
	
    # We loop over the diffuse components that come from GALProp
    # templates and refactor them using the command
    # fermipy-sum-ring-gasmaps-sg, which is equivalent to
    for galprop_comp in diffuse_galprop_components:
       fermipy-coadd(galprop_comp)

   # We do a triple loop over all of the diffuse components, all the
   # binning components and all the energy bins and convolve the
   # emission template with the instrument response using the command
   # fermipy-srcmaps-diffuse-sg, which is equivalent to
   for diffuse_comp in diffuse_components:
       for binning_comp in binned_components:
           for energy in energy_bins:
	        fermipy-srcmap-diffuse(diffuse_comp, binning_comp, energy)

   # We then do a double loop over all the diffuse components and all
   # the binning components and stack the template maps into single
   # files using the command
   # fermipy-vstack-diffuse-sg, which is equivalent to
   for diffuse_comp in diffuse_components:
       for binning_comp in binned_components:
           fermipy-vstack-diffuse(diffuse_comp, binning_comp)

    # We then do a double loop over source catalogs and binning
    # components and compute the templates for each source using the
    # command
    # fermipy-srcmaps-catalog-sg, which is equivalent to 
    for catalog in catalogs:
       for binning_comp in binned_components:
           fermipy-srcmaps-catalog(catalog, binning_comp)

    # We then loop over the catalog components (essentially
    # sub-sets of catalog sources that we want to merge)
    # and merge those sources into templates using the command
    # fermipy-merge-srcmaps-sg, which is equivalent to
    for catalog_comp in catalog_components:
       for binning_comp in binned_components:
           fermipy-merge-srcmaps(catalog_comp, binning_comp)

    # At this point we have a library to template maps for all the
    # emision components that we have defined.
    # Now we want to define specific models.  We do this
    # using the commands
    # fermipy-init-model and fermipy-assemble-model-sg, which is equivalent to
    for model in models:
        fermipy-assemble-model(model)

    # At this point we, for each model under consideration we have an
    # analysis directory that is set up for fermipy

    
Configuration
----------------

This section describes the configuration management scheme used within
the fermipy.diffuse package and documents the configuration parameters
that can be set in the configuration file.

Analysis classes in the dmpipe package all inherit from the `fermipy.jobs.Link`
class, which allow user to invoke the class either interactively within python
or from the unix command line.

From the command line

.. code-block:: bash

   $ fermipy-srcmaps-diffuse-sg --comp config/binning.yaml --data config/dataset_source.yaml --library models/library.yaml


From python there are a number of ways to do it, we recommend this:

.. code-block:: python

   from fermipy.diffuse.gt_srcmap_partial import SrcmapsDiffuse_SG
   link = SrcmapsDiffuse_SG( )
   link.update_args(dict(comp='config/binning.yaml', data='config/dataset_source.yaml',library='models/library.yaml'))   
   link.run()				     


Top Level Configuration
-----------------------

We use a yaml file to define the top-level analysis parameters.  

.. code-block:: yaml
   :caption: Sample *top level* Configuration

    # The binning components
    comp : config/binning.yaml
    # The dataset
    data : config/dataset_source.yaml
    # Library with the fitting components
    library : models/library.yaml
    # Yaml file with the list of models to prepare 
    models : models/modellist.yaml
    # Input FT1 file
    ft1file : P8_P305_8years_source_zmax105.lst
    # HEALPix order for counts cubes
    hpx_order_ccube : 9
    # HEALPix order for exposure cubes
    hpx_order_expcube : 6
    # HEALPix order fitting models
    hpx_order_fitting : 7
    # Build the XML files for the diffuse emission model components
    make_diffuse_comp_xml : True
    # Build the XML files for the catalog source model components 
    make_catalog_comp_xml : True
    # Name of the directory for the merged GALProp gasmaps
    merged_gasmap_dir : merged_gasmap
    # Number of catalog sources per batch job
    catalog_nsrc : 500




Binning Configuration
---------------------

We use a yaml file to define the binning components.  

.. code-block:: yaml
   :caption: Sample *binning* Configuration

   coordsys : 'GAL'
   E0:                      
       log_emin : 1.5
       log_emax : 2.0
       enumbins : 2
       zmax : 80.
       psf_types :
           PSF3 : 
               hpx_order : 5
   E1:                      
       log_emin : 2.0
       log_emax : 2.5
       enumbins : 2
       zmax : 90.
       psf_types :
           PSF23 : 
               hpx_order : 6
   E2:                      
       log_emin : 2.5
       log_emax : 3.0
       enumbins : 3
       zmax : 100.
       psf_types :
           PSF123 : 
               hpx_order : 8
   E3:                      
       log_emin : 3.0
       log_emax : 6.0
       enumbins : 9
       zmax : 105.
       psf_types :
           PSF0123 : 
              hpx_order : 9

* coordsys : 'GAL' or 'CEL'
     Coordinate system to use

* log_emin, log_emax: float
     Energy bin boundries in log10(MeV)

* enumbins: int
     Number of energy bins for this binning component  

* zmax : float
     Maximum zenith angle (in degrees) for this binning component

* psf_types:  dict
    Sub-dictionary of binning components by PSF event type, PSF3 means
    PSF event type 3 events only.    PSF0123 means all four PSF event types.

* hpx_order: int
    HEALPix order to use for binning.   The more familiar nside
    parameter is nside = 2**order
    

Dataset Configuration
---------------------

We use a yaml file to define the data set we are using.   The example
below specifies using a pre-defined 8 year dataset, selecting the
"SOURCE" event class and using the V2 version of the corresponding
IRFs (specifically P8R3_SOURCE_V2).

.. code-block:: yaml
   :caption: Sample *dataset* Configuration

    basedir : '/gpfs/slac/kipac/fs1/u/dmcat/data/flight/diffuse_dev'
    data_pass : 'P8'
    data_ver : 'P305'
    evclass : 'source'
    data_time : '8years'
    irf_ver : 'V2'

The basedir parameter should point at the analysis directory.
For the most part the other parameter are using to make the names of
the various files produced by the pipeline.   The evclass parameter
defines the event selection, and the IRF version is defined by a
combination of the data_ver, evclass and irf_ver parameters.


GALProp Rings Configuration
---------------------------

We use a yaml file to define the how we combine GALProp emission
templates.  The example below specifies how to construct as series of
'merged_CO' rings by combining GALProp intensity template predictions.


.. code-block:: yaml
   :caption: Sample *GALProp rings* Configuration

    galprop_run : 56_LRYusifovXCO5z6R30_QRPD_150_rc_Rs8
    ring_limits : [1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 15]
    diffuse_comp_dict :
        merged_CO : ['pi0_decay_H2R', 'bremss_H2R']
    remove_rings : ['merged_CO_7']


* galprop_run : string
     Define the GALProp run to use for this component.   This is used
     to make the filenames for input template maps.

* ring_limits : list of int
     This specfies how to combine the GALProp rings into a smaller set
     of rings.

* diffuse_comp_dict : dict
     This specifies how to make GALProp components into merged
     components for the diffuse analysis

* remove_rings: list of str
     This allow use to remove certain rings from the model

  
Catalog Component Configuration
-------------------------------

We use a yaml file to define the how we split up the catalog source
components.   The example below specifies using the FL8Y source list,
and to split out the faint sources (i.e., those with the Signif_Avg
value less that 100.), and the extended source, and to keep all the
remaining sources (i.e., the bright, pointlike, sources) as individual sources.

.. code-block:: yaml
   :caption: Sample *catalog component* Configuration

    catalog_name : FL8Y
    catalog_file : /nfs/slac/kipac/fs1/u/dmcat/ancil/catalogs/official/4FGLp/gll_psc_8year_v4.fit
    catalog_extdir : /nfs/slac/kipac/fs1/u/dmcat/ancil/catalogs/official/extended/Extended_archive_v18
    catalog_type : FL8Y
    rules_dict :
        faint : 
            cuts :
                - { cut_var: Signif_Avg, max_val : 100. }
                - mask_extended
        extended :
            cuts :
                - select_extended
        remainder :
            merge : False

	

  
Model Component Library
-----------------------


We use a yaml file to define a "library" of model components.   The
comprises a set of named emission components, and a set one or more versions
for each named component.   Here is an example library defintion file.

.. code-block:: yaml
   :caption: Sample *Model Component Library* Configuration

    # Catalog Components
    FL8Y :
        model_type : catalog
        versions : [v00]
    # Diffuse Components
    galprop_rings :
        model_type : galprop_rings
        versions : [p8-ref_IC_thin, p8-ref_HI_150, p8-ref_CO_300_mom, p8-ref_dnm_300hp]
    dnm_merged:
        model_type : MapCubeSource
       versions : ['like_4y_300K']
    gll-iem :
        model_type : MapCubeSource
        versions : [v06]
    loopI : 
        model_type : MapCubeSource
        versions : [haslam]
    bubbles : 
        model_type : MapCubeSource
        versions : [v00, v01]
    iso_map : 
        model_type : MapCubeSource
        versions : [P8R3_SOURCE_V2]
    patches :
        model_type : MapCubeSource
        versions : [v09]
        selection_dependent : True
        no_psf : True
        edisp_disable : True
    unresolved :
        model_type : MapCubeSource
        versions : [strong]
    sun-ic : 
        model_type : MapCubeSource
        versions : [v2r0, P8R3-v2r0]
        moving : True    
        edisp_disable : True
    sun-disk : 
        model_type : MapCubeSource
        versions : [v3r1, P8R3-v3r1]
        moving : True    
        edisp_disable : True
    moon :
        model_type : MapCubeSource
        versions : [v3r2, P8R3-v3r2]
        moving : True    
        edisp_disable : True


* model_type: 'MapCubeSource' or 'catalog' or 'galprop_rings' or 'SpatialMap'
   Specifies how this model should be constructed.  See more below in
   the versions parameters. 

* versions: list of str
   Specifies different versions of this model component.  How this
   string is used depend on the model type.  for 'MapCubeSource' and
   'SpatialMap' sources it is used to construct the expected filename
   for the intensity template.   For 'catalog' and 'galprop_rings' it
   is used to construct the filename for the yaml file that defines
   the sub-components for that component.

* moving: bool
  If true, then will use source-specific livetime cubes to constuct
  source templates for each zenith angle cut.

* selection_dependent : bool
  If true, then will used different source templates for each binning
  component.

* no_psf : bool
  Turns of PSF convolution for this source.   Useful for data-driven components.

* edisp_disable : bool
  Turns off energy dispersion for the source.  Useful for data-driven components.




Spectral Model Configuration
----------------------------

We use a yaml file to define the spectral models and default
parameters.   This file is simply a dictionary mapping 
names to sub-dictionaries defining spectral models and default model parameters.


Model Defintion
---------------

We use a yaml file to define each overall model, which combine
library components and spectral models.

.. code-block:: yaml
   :caption: Sample *Model Definition* Configuration

    library : models/library.yaml
    spectral_models : models/spectral_models.yaml
    sources : 
        galprop_rings_p8-ref_IC_thin : 
            model_type : galprop_rings
            version : p8-ref_IC_150
            SpectrumType : 
                default : Constant_Correction
        galprop_rings-p8-ref_HI_300:
            model_type : galprop_rings
            version : p8-ref_HI_300
            SpectrumType : 
                default : Powerlaw_Correction
                merged_HI_2_p8-ref_HI_300 : BinByBin_5
                merged_HI_3_p8-ref_HI_300 : BinByBin_5
                merged_HI_4_p8-ref_HI_300 : BinByBin_9
                merged_HI_5_p8-ref_HI_300 : BinByBin_9
                merged_HI_6_p8-ref_HI_300 : BinByBin_9
                merged_HI_8_p8-ref_HI_300 : BinByBin_5
                merged_HI_9_p8-ref_HI_300 : BinByBin_5
        galprop_rings-p8-ref_CO_300:
            model_type : galprop_rings
            version : p8-ref_CO_300_mom
            SpectrumType : 
                default : Powerlaw_Correction
                merged_CO_2_p8-ref_CO_300_mom : BinByBin_5
                merged_CO_3_p8-ref_CO_300_mom : BinByBin_5
                merged_CO_4_p8-ref_CO_300_mom : BinByBin_9
                merged_CO_5_p8-ref_CO_300_mom : BinByBin_9
                merged_CO_6_p8-ref_CO_300_mom : BinByBin_9
                merged_CO_8_p8-ref_CO_300_mom : BinByBin_5
                merged_CO_9_p8-ref_CO_300_mom : BinByBin_5
        dnm_merged :
            version : like_4y_300K
            SpectrumType : BinByBin_5
        iso_map : 
            version : P8R3_SOURCE_V2
            SpectrumType : Iso
        sun-disk : 
            version : v3r1
            SpectrumType : Constant_Correction
            edisp_disable : True
        sun-ic : 
            version : v2r0
            SpectrumType : Constant_Correction
            edisp_disable : True
        moon :
            version : v3r2
            SpectrumType : Constant_Correction
            edisp_disable : True
        patches :
            version : v09
            SpectrumType : Patches
            edisp_disable : True
        unresolved :
            version : strong
            SpectrumType : Constant_Correction
        FL8Y : 
            model_type : Catalog
            version : v00
            SpectrumType : 
                default : Constant_Correction
                FL8Y_v00_remain : Catalog
                FL8Y_v00_faint : BinByBin_9


* model_type: 'MapCubeSource' or 'catalog' or 'galprop_rings' or 'SpatialMap'
   Specifies how this model should be constructed.  See more below.

* version: str
   Specifies version of this model component. 

* edisp_disable : bool
  Turns off energy dispersion for the source.  Useful for data-driven
  components.  Needed for model XML file construction.

* SpectrumType: str or dictionary 
  This specifies the Spectrum type to use for this model component.
  For 'catalog' and 'galprop_rings' model types it can be a
  dictionary mapping model sub-components to spectrum types.
  Note that the spectrum types should be defined in the spectral model
  configuration described above.
  


Module contents
---------------

.. automodule:: fermipy.diffuse
    :members:
    :undoc-members:
    :show-inheritance:


Configuration, binning, default options, etc...
-----------------------------------------------

.. automodule:: fermipy.diffuse.binning
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: fermipy.diffuse.defaults
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: fermipy.diffuse.name_policy
    :members:
    :undoc-members:
    :show-inheritance:


Utilities and tools
-------------------

 .. automodule:: fermipy.diffuse.spectral
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: fermipy.diffuse.timefilter
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: fermipy.diffuse.source_factory
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: fermipy.diffuse.utils
    :members:
    :undoc-members:
    :show-inheritance:

       
Helper classes to manage model building
---------------------------------------

    
.. autoclass:: fermipy.diffuse.model_component.ModelComponentInfo
    :members:
    :undoc-members:
    :show-inheritance:
   
.. autoclass:: fermipy.diffuse.model_component.CatalogInfo
    :members:
    :undoc-members:
    :show-inheritance:
    
.. autoclass:: fermipy.diffuse.model_component.GalpropMergedRingInfo
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: fermipy.diffuse.model_component.ModelComponentInfo
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: fermipy.diffuse.model_component.IsoComponentInfo
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: fermipy.diffuse.model_component.PointSourceInfo
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: fermipy.diffuse.model_component.CompositeSourceInfo
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: fermipy.diffuse.model_component.CatalogSourcesInfo
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: fermipy.diffuse.diffuse_src_manager.GalpropMapManager
    :members:
    :undoc-members:
    :show-inheritance:
    
.. autoclass:: fermipy.diffuse.diffuse_src_manager.DiffuseModelManager
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: fermipy.diffuse.catalog_src_manager.CatalogSourceManager
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: fermipy.diffuse.model_manager.ModelComponent
    :members:
    :undoc-members:
    :show-inheritance:
    
.. autoclass:: fermipy.diffuse.model_manager.ModelInfo
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: fermipy.diffuse.model_manager.ModelManager
    :members:
    :undoc-members:
    :show-inheritance:


Trivial Link Sub-classes
------------------------

.. autoclass:: fermipy.diffuse.job_library.Gtlink_select
    :members:
    :undoc-members:
    :show-inheritance:


.. autoclass:: fermipy.diffuse.job_library.Gtlink_bin
    :members:
    :undoc-members:
    :show-inheritance:


.. autoclass:: fermipy.diffuse.job_library.Gtlink_expcube2
    :members:
    :undoc-members:
    :show-inheritance:


.. autoclass:: fermipy.diffuse.job_library.Gtlink_scrmaps
    :members:
    :undoc-members:
    :show-inheritance:


.. autoclass:: fermipy.diffuse.job_library.Gtlink_ltsum
    :members:
    :undoc-members:
    :show-inheritance:


.. autoclass:: fermipy.diffuse.job_library.Gtlink_mktime
    :members:
    :undoc-members:
    :show-inheritance:


.. autoclass:: fermipy.diffuse.job_library.Gtlink_ltcube
    :members:
    :undoc-members:
    :show-inheritance:


.. autoclass:: fermipy.diffuse.solar.Gtlink_expcube2_wcs
    :members:
    :undoc-members:
    :show-inheritance:


.. autoclass:: fermipy.diffuse.solar.Gtlink_exphpsun
    :members:
    :undoc-members:
    :show-inheritance:

    
.. autoclass:: fermipy.diffuse.solar.Gtlink_suntemp
    :members:
    :undoc-members:
    :show-inheritance:

    
.. autoclass:: fermipy.diffuse.job_library.Link_FermipyCoadd
    :members:
    :undoc-members:
    :show-inheritance:

    
.. autoclass:: fermipy.diffuse.job_library.Link_FermipyGatherSrcmaps
    :members:
    :undoc-members:
    :show-inheritance:

    
.. autoclass:: fermipy.diffuse.job_library.Link_FermipyVstack
    :members:
    :undoc-members:
    :show-inheritance:

    
.. autoclass:: fermipy.diffuse.job_library.Link_FermipyHealview
    :members:
    :undoc-members:
    :show-inheritance:
    

Standalone Analysis Links
-------------------------
    
.. autoclass:: fermipy.diffuse.gt_merge_srcmaps.GtMergeSrcmaps
    :members:
    :undoc-members:
    :show-inheritance:
    
.. autoclass:: fermipy.diffuse.gt_srcmap_partial.GtSrcmapsDiffuse
    :members:
    :undoc-members:
    :show-inheritance:
    
.. autoclass:: fermipy.diffuse.gt_srcmaps_catalog.GtSrcmapsCatalog
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: fermipy.diffuse.gt_assemble_model.InitModel
    :members:
    :undoc-members:
    :show-inheritance:
    
.. autoclass:: fermipy.diffuse.gt_assemble_model.AssembleModel
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: fermipy.diffuse.residual_cr.ResidualCR
    :members:
    :undoc-members:
    :show-inheritance:


Batch job dispatch classes
--------------------------


.. autoclass:: fermipy.diffuse.job_library.Gtexpcube2_SG
    :members:
    :undoc-members:
    :show-inheritance:


.. autoclass:: fermipy.diffuse.job_library.Gtltsum_SG
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: fermipy.diffuse.solar.Gtexpcube2wcs_SG
    :members:
    :undoc-members:
    :show-inheritance:


.. autoclass:: fermipy.diffuse.solar.Gtexphpsun_SG
    :members:
    :undoc-members:
    :show-inheritance:

    
.. autoclass:: fermipy.diffuse.solar.Gtsuntemp_SG
    :members:
    :undoc-members:
    :show-inheritance:

    
.. autoclass:: fermipy.diffuse.gt_coadd_split.CoaddSplit_SG
    :members:
    :undoc-members:
    :show-inheritance:

    
.. autoclass:: fermipy.diffuse.job_library.GatherSrcmaps_SG
    :members:
    :undoc-members:
    :show-inheritance:

    
.. autoclass:: fermipy.diffuse.job_library.Vstack_SG
    :members:
    :undoc-members:
    :show-inheritance:

    
.. autoclass:: fermipy.diffuse.job_library.Healview_SG
    :members:
    :undoc-members:
    :show-inheritance:
    

.. autoclass:: fermipy.diffuse.job_library.SumRings_SG
    :members:
    :undoc-members:
    :show-inheritance:
    


.. autoclass:: fermipy.diffuse.job_library.SumRings_SG
    :members:
    :undoc-members:
    :show-inheritance:
    

.. autoclass:: fermipy.diffuse.residual_cr.ResidualCR_SG
    :members:
    :undoc-members:
    :show-inheritance:

    
.. autoclass:: fermipy.diffuse.gt_merge_srcmaps.MergeSrcmaps_SG
    :members:
    :undoc-members:
    :show-inheritance:

    
.. autoclass:: fermipy.diffuse.gt_srcmap_partial.SrcmapsDiffuse_SG
    :members:
    :undoc-members:
    :show-inheritance:

    
.. autoclass:: fermipy.diffuse.gt_srcmaps_catalog.SrcmapsCatalog_SG
    :members:
    :undoc-members:
    :show-inheritance:

    
.. autoclass:: fermipy.diffuse.gt_assemble_model.AssembleModel_SG
    :members:
    :undoc-members:
    :show-inheritance:

    
.. autoclass:: fermipy.diffuse.residual_cr.ResidualCR_SG
    :members:
    :undoc-members:
    :show-inheritance:

    
.. autoclass:: fermipy.diffuse.gt_split_and_bin.SplitAndBin_SG
    :members:
    :undoc-members:
    :show-inheritance:

    
.. autoclass:: fermipy.diffuse.gt_split_and_mktime.SplitAndMktime_SG
    :members:
    :undoc-members:
    :show-inheritance:

    
Analysis chain classes
----------------------

.. autoclass:: fermipy.diffuse.gt_coadd_split.CoaddSplit
    :members:
    :undoc-members:
    :show-inheritance:


.. autoclass:: fermipy.diffuse.gt_split_and_bin.SplitAndBin
    :members:
    :undoc-members:
    :show-inheritance:

    
.. autoclass:: fermipy.diffuse.gt_split_and_bin.SplitAndBinChain
    :members:
    :undoc-members:
    :show-inheritance:


.. autoclass:: fermipy.diffuse.gt_split_and_mktime.SplitAndMktime
    :members:
    :undoc-members:
    :show-inheritance:


.. autoclass:: fermipy.diffuse.gt_split_and_mktime.SplitAndMktimeChain
    :members:
    :undoc-members:
    :show-inheritance:


.. autoclass:: fermipy.diffuse.diffuse_analysis.DiffuseCompChain
    :members:
    :undoc-members:
    :show-inheritance:


.. autoclass:: fermipy.diffuse.diffuse_analysis.CatalogCompChain
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: fermipy.diffuse.gt_assemble_model.AssembleModelChain
    :members:
    :undoc-members:
    :show-inheritance:


.. autoclass:: fermipy.diffuse.diffuse_analysis.DiffuseAnalysisChain
    :members:
    :undoc-members:
    :show-inheritance:


.. autoclass:: fermipy.diffuse.residual_cr.ResidualCRChain
    :members:
    :undoc-members:
    :show-inheritance:


.. autoclass:: fermipy.diffuse.solar.SunMoonChain
    :members:
    :undoc-members:
    :show-inheritance:



