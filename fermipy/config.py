from __future__ import absolute_import, division, print_function, \
    unicode_literals

import os
import yaml
import fermipy
import fermipy.utils as utils


def create_default_config(defaults):
    """Create a configuration dictionary from a defaults dictionary.
    The defaults dictionary defines valid configuration keys with
    default values and docstrings.  Each dictionary element should be
    a tuple or list containing (default value,docstring,type)."""

    o = {}
    for key, item in defaults.items():

        if isinstance(item, dict):
            o[key] = create_default_config(item)
        elif isinstance(item, tuple):

            value, comment, item_type = item

            if isinstance(item_type, tuple):
                item_type = item_type[0]

            if value is None and (item_type == list or item_type == dict):
                value = item_type()

            if key in o: raise Exception('Duplicate key.')

            o[key] = value
        else:
            print(key, item, type(item))
            raise Exception('Unrecognized type for default dict element.')

    return o


def cast_config(config, defaults):
    for key, item in config.items():

        if key not in defaults:
            continue
        
        if isinstance(item, dict):
            cast_config(config[key], defaults[key])
        elif item is None:
            continue
        else:
            value, comment, item_type = defaults[key]
            if item_type is None or isinstance(item_type, tuple):
                continue

            if isinstance(item, str) and item_type == list:
                config[key] = [item]
            else:
                config[key] = item_type(config[key])


def validate_config(config, defaults, block='root'):
    for key, item in config.items():

        if key not in defaults:
            raise Exception('Invalid key in \'%s\' block of configuration: %s' %
                            (block, key))

        if isinstance(item, dict):
            validate_config(config[key], defaults[key], key)


class Configurable(object):
    """The base class provides common facilities like loading and saving
    configuration state. """

    def __init__(self, config, **kwargs):

        self._config = self.get_config()
        self._configdir = None

        if isinstance(config,str) and os.path.isfile(config):
            self._configdir = os.path.abspath(os.path.dirname(config))
            config_dict = yaml.load(open(config))            
        elif isinstance(config, dict) or config is None:
            config_dict = config
        elif isinstance(config,str) and not os.path.isfile(config):
            raise Exception('Invalid path to configuration file: %s'%config)
        else:
            raise Exception('Invalid config argument.')
            
        self.configure(config_dict, **kwargs)

        if self.configdir and 'fileio' in self.config and \
                self.config['fileio']['outdir'] is None:
            self.config['fileio']['outdir'] = self.configdir

    def configure(self, config, **kwargs):

        validate = kwargs.pop('validate',False)        
        config = utils.merge_dict(config, kwargs, add_new_keys=True)
        if validate:
            validate_config(config, self.defaults)
        cast_config(config, self.defaults)
        self._config = utils.merge_dict(self._config, config)

    @classmethod
    def get_config(cls):
        """Return a default configuration dictionary for this class."""
        return create_default_config(cls.defaults)

    @property
    def config(self):
        """Return the configuration dictionary of this class."""
        return self._config

    @property
    def configdir(self):
        return self._configdir
    
    def write_config(self, outfile):
        """Write the configuration dictionary to an output file."""
        utils.write_yaml(self.config, outfile, default_flow_style=False)

    def print_config(self, logger, loglevel=None):

        cfgstr = yaml.dump(self.config,default_flow_style=False)

        if loglevel is None:
            logger.debug('Configuration:\n' + cfgstr)
        else:
            logger.log(loglevel, 'Configuration:\n' + cfgstr)

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
        # config_logging = ConfigManager.load('logging.yaml')

        config = {}
        if config['fileio']['outdir'] is None:
            config['fileio']['outdir'] = os.path.abspath(
                os.path.dirname(configfile))

        user_config = ConfigManager.load(configfile)
        config = utils.merge_dict(config, user_config, True)

        config['fileio']['outdir'] = os.path.abspath(config['fileio']['outdir'])

        return config

    @staticmethod
    def load(path):

        if not os.path.isfile(path):
            path = os.path.join(fermipy.PACKAGE_ROOT, 'config', path)

        with open(path, 'r') as f: config = yaml.load(f)
        return config
