from __future__ import absolute_import, division, print_function

import numpy as np

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
