
import os
import sys
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
            configpath = os.path.join(fermipy.PACKAGE_ROOT,'config',
                                      'logging.yaml')
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
        
        if logfile is not None:
            logfile = logfile.replace('.log','') + '.log'
        
        logger = logging.getLogger(name)

        # Don't propagate to root logger
        logger.propagate = False
        logger.setLevel(logging.DEBUG)

        datefmt = '%Y-%m-%d %H:%M:%S'
        format_stream = '%(asctime)s %(levelname)-8s %(name)s.%(funcName)s(): %(message)s'
        format_file = '%(asctime)s %(levelname)-8s %(name)s.%(funcName)s(): %(message)s' 
#        format_file = '%(asctime)s %(levelname)-8s %(name)s.%(funcName)s() [%(filename)s:%(lineno)d]: %(message)s' 
        
        if not logger.handlers:

            formatter = logging.Formatter(format,datefmt)

            # Add a file handler
            if logfile is not None:
                fh = logging.FileHandler(logfile)
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(logging.Formatter(format_file,datefmt))
                logger.addHandler(fh)
            
            # Add a stream handler
            ch = logging.StreamHandler()
            ch.setLevel(loglevel)
            ch.setFormatter(logging.Formatter(format_stream,datefmt))
            logger.addHandler(ch)
        else:
            logger.handlers[-1].setLevel(loglevel)
            
        return logger

class StreamLogger(object):
    """File-like object to log stdout/stderr using the `logging` module."""

    def __init__(self, name='stdout', logfile=None, quiet=True):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.format = '%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s'
        self.datefmt='%Y-%m-%d %H:%M:%S'
        self.stdout = sys.stdout
#        logging.basicConfig(level=logging.DEBUG, 
#                            format=self.format, filename=file)

        self.formatter = logging.Formatter(self.format,self.datefmt)
        fhdlr = logging.FileHandler(logfile)
        fhdlr.setFormatter(self.formatter)
        self.logger.addHandler(fhdlr)

        if not quiet and name == 'stdout':
            shdlr = logging.StreamHandler(sys.stdout)
            shdlr.setFormatter(self.formatter)
            self.logger.addHandler(shdlr)
        elif not quiet and name == 'stderr':
            shdlr = logging.StreamHandler(sys.stderr)
            shdlr.setFormatter(self.formatter)
            self.logger.addHandler(shdlr)

        sys.stdout = self

    def __del__(self):
        self.close()

    def close(self):
        self.flush()
        sys.stdout = self.stdout
        self.logger.handlers = []
        
    def write(self, msg, level=logging.DEBUG):
        msg = msg.rstrip('\n')
        if len(msg) > 0:
            self.logger.log(level, msg)

    def flush(self):
        for handler in self.logger.handlers:
            handler.flush()
