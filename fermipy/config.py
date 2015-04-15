import fermipy
import yaml
from utils import *
import defaults

class ConfigManager(object):

    @staticmethod
    def create(configfile):
        """Create a configuration dictionary from a yaml config file.
        This function will first populate the dictionary with defaults
        taken from pre-defined configuration files.  The configuration
        dictionary is then updated with the user-defined configuration
        file.  Any settings defined by the user will take precedence
        over the default settings."""
        
        # populate config dictionary with an initial set of values
        config_logging = ConfigManager.load('logging.yaml')

        config = ConfigManager.load(os.path.join(fermipy.PACKAGE_ROOT,'config',
                                                 'base.yaml'))
        config['fileio'].setdefault('outdir',os.path.dirname(configfile))

        if config['fileio']['outdir'] is None:
            config['fileio']['outdir'] = os.path.dirname(configfile)
        #        config['logging'] = config_logging
        
        user_config = ConfigManager.load(configfile)
        config = merge_dict(config,user_config,True)
        
        return config        

    @staticmethod
    def load(path):

        if not os.path.isfile(path):        
            path = os.path.join(fermipy.PACKAGE_ROOT,'config',path)

        with open(path,'r') as f: config = yaml.load(f)
        return config
