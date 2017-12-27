# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import time


class Timer(object):
    """Lightweight timer class."""

    def __init__(self):
        self._t0 = None
        self._time = 0

    @property
    def elapsed_time(self):
        """Get the elapsed time."""

        # Timer is running
        if self._t0 is not None:
            return self._time + self._get_time()
        else:
            return self._time

    @classmethod
    def create(cls, start=False):
        o = cls()
        if start:
            o.start()
        return o

    def clear(self):
        self._time = 0
        self._t0 = None

    def start(self):
        """Start the timer."""

        if self._t0 is not None:
            raise RuntimeError('Timer already started.')

        self._t0 = time.time()

    def stop(self):
        """Stop the timer."""

        if self._t0 is None:
            raise RuntimeError('Timer not started.')

        self._time += self._get_time()
        self._t0 = None

    def _get_time(self):
        return time.time() - self._t0
