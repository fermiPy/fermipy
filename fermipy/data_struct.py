# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
from collections import OrderedDict


class MutableNamedTuple(OrderedDict):
    """Light-weight class for representing data structures.  Internal
    data members can be accessed using both attribute- and
    dictionary-style syntax.  The constructor accepts an ordered dict
    defining the names and default values of all data members.  Once
    an instance is initialized no new data members may be added."""

    def __init__(self, *args, **kwargs):
        super(MutableNamedTuple, self).__init__(*args, **kwargs)
        self._initialized = True

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if hasattr(self, '_initialized'):
            if not name in self:
                raise AttributeError(name)
            super(MutableNamedTuple, self).__setitem__(name, value)
        else:
            super(MutableNamedTuple, self).__setattr__(name, value)

    def __setitem__(self, name, value):
        if hasattr(self, '_initialized'):
            if not name in self:
                raise KeyError(name)
            super(MutableNamedTuple, self).__setitem__(name, value)
        else:
            super(MutableNamedTuple, self).__setitem__(name, value)

    def update(self, d, strict=False):
        for k, v in d.items():
            if k in self:
                self[k] = v
            elif strict:
                raise KeyError('Missing key {}'.format(k))
