.. _fermipy_jobs_tools:


Link objects
------------

The basic building block of an analysis pipeline is a `Link` object.  In general
a `Link` is a single application that can be called from the command line.

The fermipy.jobs package imlements five types of `Link` objects, and the idea
is that users can make sub-classes to perform the steps of their analysis.

Every link sub-class has a small required header block, for example:

.. code-block:: python

    class AnalyzeROI(Link):
       """Small class that wraps an analysis script.

       This particular script does baseline fitting of an ROI.
       """
       appname = 'fermipy-analyze-roi'
       linkname_default = 'analyze-roi'
       usage = '%s [options]' % (appname)
       description = "Run analysis of a single ROI"

       default_options = dict(config=defaults.common['config'],
                              roi_baseline=defaults.common['roi_baseline'],
                              make_plots=defaults.common['make_plots'])

       __doc__ += Link.construct_docstring(default_options)



The various pieces of the header are:

* appname
  This is the unix command that will invoke this link.  

* linkname_default 
  This is the default name that links of this type will be given when then are put into
  analysis pipeline.

* usage, description
  These are passed to the argument parser and used to build the help string.

* default_options
  This is the set of options and default values for this link

* The  __doc__ += Link.construct_docstring(default_options) line ensures that the 
  default options will be included in the class's docstring.



Link sub-classes
----------------

There are five types of `Link` sub-classes implemented here.

* `Link`
  
  This is the sub-class to use for a user-defined function.  
  In this case in addition to providing the header material above, 
  the sub-class will need to implement the run_analysis() to perform
  that function.

  .. code-block:: python
 
      def run_analysis(self, argv):
          """Run this analysis"""
          args = self._parser.parse_args(argv)

          do stuff


* `Gtlink`
  
  This is the sub-class to use to invoke a Fermi ScienceTools
  gt-tool, such as gtsrcmaps or gtexcube2.   In this case the user only needs
  to provide the header content to make the options they want 
  availble to the interface.


* `AppLink`

  This is the sub-class to use to invoke a pre-existing unix
  command. In this case the user only needs to provide the header content
  to make the options they want availble to the interface.


* `ScatterGather`

   This is the sub-class to use to send a set of similar jobs to 
   a computing batch farm.   In this case the user needs to provide the 
   standard header content and a couple of addtional things.   Here is 
   an example:

   .. code-block:: python
 
       class AnalyzeROI_SG(ScatterGather):
           """Small class to generate configurations for the `AnalyzeROI` class.

           This loops over all the targets defined in the target list.
           """
           appname = 'fermipy-analyze-roi-sg'
           usage = "%s [options]" % (appname)
           description = "Run analyses on a series of ROIs"
           clientclass = AnalyzeROI

           job_time = 1500

           default_options = dict(ttype=defaults.common['ttype'],
                                  targetlist=defaults.common['targetlist'],
                                  config=defaults.common['config'],
                                  roi_baseline=defaults.common['roi_baseline'],
                                  make_plots=defaults.common['make_plots'])

           __doc__ += Link.construct_docstring(default_options)

           def build_job_configs(self, args):
               """Hook to build job configurations
               """
               job_configs = {}

               ttype = args['ttype']

               do stuff

	       return job_configs


     The job_time class parameter should be an estimate of the time the average
     job managed by this class will take.  That is used to decided which batch 
     farm resources to use to run the job, and how often to check from job completion.

     The user defined function build_job_configs() function should build a dictionary of 
     dictionaries that contains the parameters to use for each instance of the command that 
     will run.  E.g., if you want to analyze a set of 3 ROIs, using different
     config files and making different roi_baseline output files, build_job_configs
     should return a dictionary of 3 dictionaries, something like this:

     .. code-block:: python
  
         job_configs = {"ROI_000000" : {config="ROI_000000/config.yaml", 
	                                roi_baseline="baseline",
					make_plts=True},
		        "ROI_000000" : {config="ROI_000000/config.yaml", 
	                                roi_baseline="baseline",
					make_plts=True},
			"ROI_000000" : {config="ROI_000000/config.yaml", 
	                                roi_baseline="baseline",
					make_plts=True}}
 

* `Chain`

   This is the sub-class to use to run multiple `Link` objects in sequence.
 
   For `Chain` sub-classes, in addtion to the standard header material, the user
   should profile a map_arguments() method that builds up the chain and sets the 
   options of the component `Link` objects using the _set_link() method.   Here is an example:

   .. code-block:: python
   
       def _map_arguments(self, input_dict):
           """Map from the top-level arguments to the arguments provided to
           the indiviudal links """

           config_yaml = input_dict['config']
           config_dict = load_yaml(config_yaml)

           data = config_dict.get('data')
           comp = config_dict.get('comp')
           sourcekeys = config_dict.get('sourcekeys')

           mktimefilter = config_dict.get('mktimefilter')

           self._set_link('expcube2', Gtexpcube2wcs_SG,
                          comp=comp, data=data,
                          mktimefilter=mktimefilter)

           self._set_link('exphpsun', Gtexphpsun_SG,
                          comp=comp, data=data,
                          mktimefilter=mktimefilter)

           self._set_link('suntemp', Gtsuntemp_SG,
                          comp=comp, data=data,
                          mktimefilter=mktimefilter,
                          sourcekeys=sourcekeys)



Using Links and sub-classes in python
-------------------------------------

The main aspect of the `Link` python interface are:

* Building the `Link` and setting the parameters.  By way of example we will 
build a `Link` of type `AnalyzeROI` and configure it do a standard analysis of
the ROI using the file 'config.yaml' write the resulting 
ROI snapshot to 'baseline' and make the standard validation plots.

   .. code-block:: python

       link = AnalyzeROI.create()
       link.update_args(dict(config='config.yaml',
                             roi_baseline='baseline',
			     make_plots=True))


* Seeing the equivalent command line task

   .. code-block:: python

        link.formatted_command()


* Running the `Link`:

   .. code-block:: python

       link.run()


* Seeing the status of the `Link`:

   .. code-block:: python

       link.check_job_status()


* Seeing the jobs associated to this `Link`:

    .. code-block:: python

       link.jobs
   

* Setting the arguments used to run this `Link`:

    .. code-block:: python

       link.update_args(dict=(option_name=option_value,
                              option_name2=option_value2, 
			      ...))


