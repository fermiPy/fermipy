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
        ''' Generate a spatial PS map for evaluate the data/model comparison
        The PS map will have the same geometry as the ROI.  The output of this method
        is a dictionary containing `~fermipy.skymap.Map` objects with
        the PS amplitude and sigma equivalent. By default
        this method will also save maps to FITS files and render them
        as image files.
        psmap requires a model map to be computed, therefore the user must run
        gta.write_model_map(model_name="model01") before running
        psmap = gta.psmap(cmap='ccube_00.fits',mmap='mcube_model01_00.fits',emin=100,emax=100000,nbinloge=15)

        Parameters
        ----------
        prefix : str
           Optional string that will be prepended to all output files.
        kwargs : Any
            these will override the psmap configuration file

        {options}

        Returns
        -------
        psmap : dict
           A dictionary containing the `~fermipy.skymap.Map` objects
           for PS amplitude and sigma equivalent.
        '''

        timer = Timer.create(start=True)
        schema = ConfigSchema(self.defaults['psmap'])

        #for i, c in enumerate(self.components):
        #    print('psmap: component %s : %s' %(c.name,c.files['ccube']))

        schema.add_option('loglevel', logging.INFO)

        config = schema.create_config(self.config['psmap'], **kwargs)
        print(config['loglevel'])

        self.logger.log(config['loglevel'], 'Generating PS map')

        o = gtpsmap.run(config)
        map_geom = self._geom.to_image()

        mydatcounts = o['datcounts']

        ps_map = WcsNDMap(map_geom, o['psmap'])
        pssigma_map = WcsNDMap(map_geom, o['psmapsigma'])

        o['name'] = utils.join_strings([prefix, 'PSmap'])
        o['ps_map'] = ps_map
        o['pssigma_map'] = pssigma_map
        o['config'] = kwargs

        outfile = config.get('outfile', '')
        if outfile == '':
            outfile = utils.format_filename(self.workdir, 'PSmap',
                                            prefix=[prefix])
        else:
            outfile = os.path.join(self.workdir,
                                   os.path.splitext(outfile)[0])

        o['file'] = None
        if config['write_fits']:
            o['file'] = os.path.basename(outfile) + '.fits'
            gtpsmap.make_psmap_fits(o,outfile + '.fits')
            self.logger.log(config['loglevel'], 'Writing output file %s.',
                            outfile + '.fits')

        if config['make_plots']:

            plotter = plotting.AnalysisPlotter(self.config['plotting'],
                                               fileio=self.config['fileio'],
                                               logging=self.config['logging'])
            plotter.make_psmap_plots(o, self.roi)

        self.logger.log(config['loglevel'], 'Finished PS map')
        return o
