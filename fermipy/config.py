import fermipy
import yaml
from utils import *
import defaults

def create_default_config(defaults):
    """Create a configuration dictionary from a defaults dictionary.
    The defaults dictionary defines valid configuration keys with
    default values and docstrings.  Each dictionary element should be
    a tuple or list containing (default value,docstring,type)."""

    o = {}
    for key, item in defaults.items():

        if isinstance(item,dict):
            o[key] = create_default_config(item)
        elif isinstance(item,tuple):

            value, comment, item_type = item

            if isinstance(item_type,tuple): 
                item_type = item_type[0]

            if value is None and (item_type == list or item_type == dict):
                value = item_type()
            
            if key in o: raise Exception('Duplicate key.')
                
            o[key] = value            
        else:
            print key, item, type(item)            
            raise Exception('Unrecognized type for default dict element.')

    return o

def cast_config(config,defaults):

    for key, item in config.items():

        if isinstance(item,dict): 
            cast_config(config[key],defaults[key])
        elif item is None: continue
        else: 
            value, comment, item_type = defaults[key]
            if item_type is None or isinstance(item_type,tuple): 
                continue

            if isinstance(item,str) and item_type == list:
                config[key] = [item]
            else:
                config[key] = item_type(config[key])            

def validate_config(config,defaults,block='root'):

    for key, item in config.items():
        
        if not key in defaults:
            raise Exception('Invalid key in \'%s\' block of configuration: %s'%
                            (block,key))
        
        if isinstance(item,dict):
            validate_config(config[key],defaults[key],key)

class Configurable(object):
    """The base class provides common facilities like loading and saving
    configuration state. """
    def __init__(self,config,**kwargs):

        self._config = self.get_config()

        if isinstance(config,dict) or config is None: pass
        elif os.path.isfile(config) and 'fileio' in self._config:
            self._config['fileio']['outdir'] = os.path.abspath(os.path.dirname(config))
            config = yaml.load(open(config))

        self.configure(config,**kwargs)

    def configure(self,config,**kwargs):

        config = merge_dict(config,kwargs,add_new_keys=True)
        validate_config(config,self.defaults) 
        cast_config(config,self.defaults)
        self._config = merge_dict(self._config,config)
        
    @classmethod
    def get_config(cls):
        # Load defaults
        return create_default_config(cls.defaults)

    @property
    def config(self):
        """Return the configuration dictionary of this class."""
        return self._config

    def write_config(self,outfile):
        """Write the configuration dictionary to an output file."""
        yaml.dump(self.config,open(outfile,'w'),default_flow_style=False)
    
    def print_config(self,logger,loglevel=None):

        if loglevel is None:
            logger.debug('Configuration:\n'+ yaml.dump(self.config,
                                                       default_flow_style=False))
        else:
            logger.log(loglevel,'Configuration:\n'+ yaml.dump(self.config,
                                                              default_flow_style=False))

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

        config = {}
        if config['fileio']['outdir'] is None:
            config['fileio']['outdir'] = os.path.abspath(os.path.dirname(configfile))
        
        user_config = ConfigManager.load(configfile)
        config = merge_dict(config,user_config,True)

        config['fileio']['outdir'] = os.path.abspath(config['fileio']['outdir'])
        
        return config        

    @staticmethod
    def load(path):

        if not os.path.isfile(path):        
            path = os.path.join(fermipy.PACKAGE_ROOT,'config',path)

        with open(path,'r') as f: config = yaml.load(f)
        return config
