# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import os
import copy
import fermipy
from fermipy import utils


def create_default_config(schema):
    """Create a configuration dictionary from a schema dictionary.
    The schema defines the valid configuration keys and their default
    values.  Each element of ``schema`` should be a tuple/list
    containing (default value,docstring,type) or a dict containing a
    nested schema."""

    o = {}
    for key, item in schema.items():

        if isinstance(item, dict):
            o[key] = create_default_config(item)
        elif isinstance(item, tuple):

            value, comment, item_type = item

            if isinstance(item_type, tuple):
                item_type = item_type[0]

            if value is None and (item_type == list or item_type == dict):
                value = item_type()

            if key in o:
                raise KeyError('Duplicate key in schema.')

            o[key] = value
        else:
            raise TypeError('Unrecognized type for schema dict element: %s %s' %
                            (key, type(item)))

    return o


def validate_from_schema(cfg, schema, section=None):
    for k, v in cfg.items():

        if k not in schema:
            if section is None:
                raise KeyError('Invalid configuration key: %s' % k)
            else:
                raise KeyError('Invalid configuration key: %s (section : %s)'
                               % (k, section))

        # This is a section
        if isinstance(schema[k], dict):

            if not isinstance(cfg[k], dict):
                raise TypeError('')

            validate_from_schema(cfg[k], schema[k], k)
        else:
            validate_option(k, cfg[k], schema[k][2])


def validate_option(opt_name, opt_val, schema_type):

    if opt_val is None:
        return

    type_match = type(opt_val) is schema_type
    type_checks = (schema_type in [list, dict, bool] or
                   type(opt_val) in [list, dict, bool])
    if type_checks and not type_match:
        raise TypeError('Wrong type for %s %s %s' %
                        (opt_name, type(opt_val), schema_type))


def update_from_schema(cfg, cfgin, schema):
    """Update configuration dictionary ``cfg`` with the contents of
    ``cfgin`` using the ``schema`` dictionary to determine the valid
    input keys.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary to be updated.

    cfgin : dict
        New configuration dictionary that will be merged with ``cfg``.

    schema : dict
        Configuration schema defining the valid configuration keys and
        their types.

    Returns
    -------
    cfgout : dict

    """
    cfgout = copy.deepcopy(cfg)
    for k, v in schema.items():

        if k not in cfgin:
            continue
        if isinstance(v, dict):
            cfgout.setdefault(k, {})
            cfgout[k] = update_from_schema(cfg[k], cfgin[k], v)
        elif v[2] is dict:
            cfgout[k] = utils.merge_dict(cfg[k], cfgin[k], add_new_keys=True)
        else:
            cfgout[k] = cfgin[k]

    return cfgout


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

            if utils.isstr(item) and item_type == list:
                config[key] = [item]
            else:
                config[key] = item_type(config[key])


def validate_config(config, defaults, section=None):
    for key, item in config.items():

        if (key in defaults and isinstance(defaults[key], dict)
                and not isinstance(item, dict)):
            type0 = type(defaults[key])
            type1 = type(item)

            raise TypeError('Wrong type for configuration key: '
                            '%s\ntype: %s required type: %s' % (key, type1, type0))

        if key not in defaults:

            if section is None:
                raise KeyError('Invalid configuration key: %s' % key)
            else:
                raise KeyError('Invalid configuration key: %s (section : %s)'
                               % (key, section))

        if isinstance(item, dict) and isinstance(defaults[key], dict):
            validate_config(config[key], defaults[key], key)


class ConfigSchema(object):
    """Class encapsulating a configuration schema."""

    def __init__(self, options=None, **kwargs):
        self._options = {} if options is None else options
        self._options = utils.merge_dict(self._options, kwargs,
                                         add_new_keys=True)

    def add_option(self, name, default_value, helpstr='', otype=None):

        if otype is None:
            otype = type(default_value)
        self._options[name] = (default_value, helpstr, otype)

    def add_section(self, name, section):
        self._options[name] = section

    def create_config(self, config=None, validate=True, **kwargs):

        config = {} if config is None else config

        o = create_default_config(self)
        config = utils.merge_dict(config, kwargs, add_new_keys=True)
        cast_config(config, self)
        if validate:
            validate_from_schema(config, self)

        o = update_from_schema(o, config, self)
        return o

    def items(self):
        return self._options.items()

    def __contains__(self, key):
        return key in self._options

    def __getitem__(self, key):
        return self._options[key]

    def __setitem__(self, key, value):
        self._options[key] = value


class Configurable(object):
    """The base class provides common facilities like loading and saving
    configuration state. """

    def __init__(self, config, **kwargs):
        import yaml

        self._config = self.get_config()
        self._configdir = None

        if utils.isstr(config) and os.path.isfile(config):
            self._configdir = os.path.abspath(os.path.dirname(config))
            config_dict = yaml.load(open(config))
        elif isinstance(config, dict) or config is None:
            config_dict = config
        elif utils.isstr(config) and not os.path.isfile(config):
            raise Exception('Invalid path to configuration file: %s' % config)
        else:
            raise Exception('Invalid config argument.')

        self.configure(config_dict, **kwargs)

        if self.configdir and 'fileio' in self.config and \
                self.config['fileio']['outdir'] is None:
            self.config['fileio']['outdir'] = self.configdir

    def configure(self, config, **kwargs):
        schema = ConfigSchema(self.defaults)
        config = schema.create_config(config, **kwargs)
        cast_config(config, schema)
        self._config = config

    @classmethod
    def get_config(cls):
        """Return a default configuration dictionary for this class."""
        return create_default_config(cls.defaults)

    @property
    def config(self):
        """Return the configuration dictionary of this class."""
        return self._config

    @property
    def schema(self):
        """Return the configuration schema of this class."""
        return ConfigSchema(self.defaults)

    @property
    def configdir(self):
        return self._configdir

    def write_config(self, outfile):
        """Write the configuration dictionary to an output file."""
        utils.write_yaml(self.config, outfile, default_flow_style=False)

    def print_config(self, logger, loglevel=None):
        import yaml
        cfgstr = yaml.dump(self.config, default_flow_style=False)

        if loglevel is None:
            logger.debug('Configuration:\n' + cfgstr)
        else:
            logger.log(loglevel, 'Configuration:\n' + cfgstr)


class ConfigManager(object):

    @classmethod
    def create(cls, configfile):
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

        user_config = cls.load(configfile)
        config = utils.merge_dict(config, user_config, True)

        config['fileio']['outdir'] = os.path.abspath(
            config['fileio']['outdir'])

        return config

    @staticmethod
    def load(path):
        import yaml
        if not os.path.isfile(path):
            path = os.path.join(fermipy.PACKAGE_ROOT, 'config', path)

        with open(path, 'r') as f:
            config = yaml.load(f)

        return config
