from __future__ import absolute_import, division, print_function

import numpy as np

from fermipy import config
from fermipy import defaults
from fermipy.plotting import ROIPlotter


class _DummySrc(dict):
    def __init__(self, name, ts):
        super(_DummySrc, self).__init__(ts=ts)
        self.name = name


class _DummyROI(object):
    def __init__(self):
        self.point_sources = [
            _DummySrc('srcA', 10.0),
            _DummySrc('srcB', 20.0),
            _DummySrc('srcC', 5.0),
        ]
        self._src_skydir = None


class _MaskRecorder(object):
    def __init__(self):
        self.label_mask = None

    def plot_sources(self, skydir, labels, plot_kwargs, text_kwargs, **kwargs):
        self.label_mask = kwargs.get('label_mask')


def test_plot_roi_label_source_and_ts_threshold():
    """Verify source labeling by explicit names and TS threshold."""
    roi = _DummyROI()

    # label_source path: only listed sources are labeled.
    recorder = _MaskRecorder()
    ROIPlotter.plot_roi(recorder, roi, label_source=['srcA', 'srcC'])
    assert np.array_equal(recorder.label_mask, np.array([True, False, True]))

    # label_ts_threshold path: only sources above threshold are labeled.
    recorder = _MaskRecorder()
    ROIPlotter.plot_roi(recorder, roi, label_ts_threshold=12.0)
    assert np.array_equal(recorder.label_mask, np.array([False, True, False]))


def test_plotting_ra_format_default():
    """Test that ra_format has correct default value in plotting config."""
    plotting_defaults = defaults.plotting
    assert 'ra_format' in plotting_defaults
    assert plotting_defaults['ra_format'][0] == 'hour'


def test_psmap_ra_format_default():
    """Test that ra_format has correct default value in psmap config."""
    psmap_defaults = defaults.psmap
    assert 'ra_format' in psmap_defaults
    assert psmap_defaults['ra_format'][0] == 'hour'


def test_ra_format_config_values():
    """Test that ra_format defaults and schema metadata are valid."""
    plotting_defaults = defaults.plotting

    # Check default.
    default_value = plotting_defaults['ra_format'][0]
    assert default_value in ['hour', 'deg']

    # Check description exists.
    description = plotting_defaults['ra_format'][1]
    assert len(description) > 0
    assert 'hour' in description.lower() or 'deg' in description.lower()

    # Check type.
    value_type = plotting_defaults['ra_format'][2]
    assert value_type == str


def test_ra_format_in_roiplotter_defaults():
    """Test that ROIPlotter has ra_format in its defaults."""
    assert 'ra_format' in ROIPlotter.defaults
    default_value = ROIPlotter.defaults['ra_format'][0]
    assert default_value == 'hour'


def test_plotting_config_creation():
    """Test that plotting configuration can be created with ra_format."""
    schema = config.ConfigSchema(defaults.plotting)
    cfg = schema.create_config({'ra_format': 'deg'})
    assert cfg['ra_format'] == 'deg'

    # Test with default value.
    cfg_default = schema.create_config({})
    assert cfg_default['ra_format'] == 'hour'


def test_psmap_config_creation():
    """Test that psmap configuration can be created with ra_format."""
    schema = config.ConfigSchema(defaults.psmap)
    cfg = schema.create_config({'ra_format': 'deg'})
    assert cfg['ra_format'] == 'deg'

    # Test with default value.
    cfg_default = schema.create_config({})
    assert cfg_default['ra_format'] == 'hour'


def test_ra_format_override():
    """Test that ra_format can be overridden in configuration."""
    schema = config.ConfigSchema(defaults.plotting)
    base_cfg = schema.create_config({'ra_format': 'hour'})
    assert base_cfg['ra_format'] == 'hour'

    override_cfg = schema.create_config(base_cfg, ra_format='deg')
    assert override_cfg['ra_format'] == 'deg'
