
import os
import logging
import logging.config
import yaml
import fermipy

def logLevel(level):
    '''This is a function that returns a python like
    level from a HEASOFT like level.'''

    levels_dict = {0:50,
                   1:40,
                   2:30,
                   3:20,
                   4:10}

    if not isinstance(level,int):
        level = int(level)

    if level > 4:
        level = 4

    return levels_dict[level]


class Logger(object):
    """This class provides helper functions which facilitate creating
    instances of the built-in logger class."""
            
    @staticmethod
    def setup(config=None,logfile=None):
        """This method sets up the default configuration of the
        logger.  Once this method is called all subsequent instances
        Logger instances will inherit this configuration."""

        if config is None:
            configpath = os.path.join(fermipy.PACKAGE_ROOT,'config','logging.yaml')
            with open(configpath,'r') as f:
                config = yaml.load(f)

        # Update configuration
        if logfile:
            for name, h in config['handlers'].items():
                if 'file_handler' in name:
                    config['handlers'][name]['filename'] = logfile
            
        logging.config.dictConfig(config)

        # get the root logger
#        logger = logging.getLogger()
#        for h in logger.handlers:
#            if 'file_handler' in h.name:
#                print h.name
        
    @staticmethod
    def get(name, logfile, loglevel=logging.DEBUG):

#        logging.config.dictConfig({
#                'version': 1,              
#                'disable_existing_loggers': False})
        
        if logfile is None:
            raise Exception('Invalid log file: ' + logfile)
        
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logfile = logfile.replace('.log','') + '.log'
        
        if not logger.handlers:
            fh = logging.FileHandler(logfile)
            fh.setLevel(logging.DEBUG)
            ch = logging.StreamHandler()
            ch.setLevel(loglevel)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            logger.addHandler(fh)
            logger.addHandler(ch)
        else:
            logger.handlers[1].setLevel(loglevel)
            
        return logger
