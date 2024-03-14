# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import logging
from fermipy.config import ConfigSchema
from fermipy.timing import Timer
import fermipy.gtpsmap as gtpsmap


class PSMapGenerator(object):
    """Mixin class for `~fermipy.gtanalysis.GTAnalysis` that
    generates PS maps."""

    def psmap(self, prefix='', **kwargs):
        ''' Add the description here!'''

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
            self.logger.info('Generate source model: %s' % mcube)
            gta.write_model_map(model_name=model_name)

        schema.add_option('loglevel', logging.INFO)

        schema.add_option('cmap', self.files['ccube'])
        schema.add_option('mmap', mcube)

        config = schema.create_config(self.config['psmap'], **kwargs)

        self.logger.info('Generating PS map')
        print (config)

        gtpsmap.run(config)

        # python ../scripts/WCSview.py -i psmap.fits --zscale linear