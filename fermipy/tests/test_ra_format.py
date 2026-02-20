# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for RA format configuration in plotting"""
from __future__ import absolute_import, division, print_function
try:
    import pytest
except ImportError:
    pytest = None
from fermipy import defaults
from fermipy import config


def test_plotting_ra_format_default():
    """Test that ra_format has correct default value in plotting config"""
    plotting_defaults = defaults.plotting
    assert 'ra_format' in plotting_defaults
    assert plotting_defaults['ra_format'][0] == 'hour'


def test_psmap_ra_format_default():
    """Test that ra_format has correct default value in psmap config"""
    psmap_defaults = defaults.psmap
    assert 'ra_format' in psmap_defaults
    assert psmap_defaults['ra_format'][0] == 'hour'


def test_ra_format_config_values():
    """Test that ra_format accepts valid values"""
    plotting_defaults = defaults.plotting
    
    # Check default
    default_value = plotting_defaults['ra_format'][0]
    assert default_value in ['hour', 'deg']
    
    # Check description exists
    description = plotting_defaults['ra_format'][1]
    assert len(description) > 0
    assert 'hour' in description.lower() or 'deg' in description.lower()
    
    # Check type
    value_type = plotting_defaults['ra_format'][2]
    assert value_type == str


def test_ra_format_in_roiplotter_defaults():
    """Test that ROIPlotter has ra_format in its defaults"""
    from fermipy.plotting import ROIPlotter
    
    assert 'ra_format' in ROIPlotter.defaults
    default_value = ROIPlotter.defaults['ra_format'][0]
    assert default_value == 'hour'


def test_plotting_config_creation():
    """Test that plotting configuration can be created with ra_format"""
    schema = config.ConfigSchema(defaults.plotting)
    cfg = schema.create_config({'ra_format': 'deg'})
    assert cfg['ra_format'] == 'deg'
    
    # Test with default value
    cfg_default = schema.create_config({})
    assert cfg_default['ra_format'] == 'hour'


def test_psmap_config_creation():
    """Test that psmap configuration can be created with ra_format"""
    schema = config.ConfigSchema(defaults.psmap)
    cfg = schema.create_config({'ra_format': 'deg'})
    assert cfg['ra_format'] == 'deg'
    
    # Test with default value
    cfg_default = schema.create_config({})
    assert cfg_default['ra_format'] == 'hour'


def test_ra_format_override():
    """Test that ra_format can be overridden in configuration"""
    # Create base config
    schema = config.ConfigSchema(defaults.plotting)
    base_cfg = schema.create_config({'ra_format': 'hour'})
    assert base_cfg['ra_format'] == 'hour'
    
    # Override with deg
    override_cfg = schema.create_config(base_cfg, ra_format='deg')
    assert override_cfg['ra_format'] == 'deg'


if __name__ == '__main__':
    # Run tests
    test_plotting_ra_format_default()
    test_psmap_ra_format_default()
    test_ra_format_config_values()
    test_ra_format_in_roiplotter_defaults()
    test_plotting_config_creation()
    test_psmap_config_creation()
    test_ra_format_override()
    print("All tests passed!")
