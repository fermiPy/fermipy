# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import logging
from fermipy.config import ConfigSchema
from fermipy.timing import Timer
import fermipy.utils as utils
import fermipy.gtpsmap as gtpsmap
import fermipy.plotting as plotting

from gammapy.maps import WcsNDMap, WcsGeom


class PSMapGenerator(object):
    """Mixin class for `~fermipy.gtanalysis.GTAnalysis` that
    generates PS maps."""

    def psmap(self, prefix='', **kwargs):
        ''' Generate a spatial PS map for a source component with
        properties defined by the `model` argument.  The PS map will
        have the same geometry as the ROI.  The output of this method
        is a dictionary containing `~fermipy.skymap.Map` objects with
        the PS and amplitude of the best-fit test source.  By default
        this method will also save maps to FITS files and render them
        as image files.

        This method uses a simplified likelihood fitting
        implementation that only fits for the normalization of the
        test source.  Before running this method it is recommended to
        first optimize the ROI model (e.g. by running
        :py:meth:`~fermipy.gtanalysis.GTAnalysis.optimize`


        Parameters
        ----------
        prefix : str
           Optional string that will be prepended to all output files.

        {options}

        Returns
        -------
        psmap : dict
           A dictionary containing the `~fermipy.skymap.Map` objects
           for PS and source amplitude.
        '''

        timer = Timer.create(start=True)
        schema = ConfigSchema(self.defaults['psmap'])


        if 'model_name' in kwargs.keys():
            model_name = kwargs['model_name']
        else:
            model_name = 'model00'

        mcube = os.path.join(self.workdir, 'mcube_%s.fits' % (model_name))

        if os.path.exists(mcube):
            self.logger.info('Using source model: %s' % mcube)
        else:
            raise FileExistsError('You must first generate a source model map using gta.write_model_map(model_name="%s")' % model_name)
            #gta.write_model_map(model_name=model_name)

        schema.add_option('loglevel', logging.INFO)

        schema.add_option('cmap', self.files['ccube'])
        schema.add_option('mmap', mcube)

        config = schema.create_config(self.config['psmap'], **kwargs)

        self.logger.info('Generating PS map')

        o = gtpsmap.run(config)
        map_geom = self._geom.to_image()

        ps_map      = WcsNDMap(map_geom, o['psmap'])
        pssigma_map = WcsNDMap(map_geom, o['psmapsigma'])

        o['name']  = utils.join_strings([prefix, model_name])
        o['ps_map']= ps_map
        o['pssigma_map']=pssigma_map
        o['config']=kwargs

        o['file'] = None
        if config['write_fits']:
            outfile = config.get('outfile', None)
            if outfile is None:
                outfile = utils.format_filename(self.workdir, 'psmap',
                                                prefix=[o['name']])
            else:
                outfile = os.path.join(self.workdir,
                                       os.path.splitext(outfile)[0])

            o['file'] = os.path.basename(outfile) + '.fits'
            gtpsmap.make_psmap_fits(o,outfile + '.fits')

        if config['make_plots']:

            plotter = plotting.AnalysisPlotter(self.config['plotting'],
                                               fileio=self.config['fileio'],
                                               logging=self.config['logging'])
            plotter.make_psmap_plots(o, self.roi)

        self.logger.log(config['loglevel'], 'Finished PS map')
        return o