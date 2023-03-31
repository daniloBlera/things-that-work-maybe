#!/usr/bin/env python3
"""Return default values on missing keys

Maybe useful for implementing a index <-> token mapping that
returns "unknown" values for out of vocabulary keys.
"""
from collections import UserDict


# %% the definition
class dyct(UserDict):
    def __init__(self, default_factory=None, /, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not callable(default_factory) and default_factory is not None:
            raise TypeError('First argument must be callable or None')

        self.default_factory = default_factory

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)

        if key not in self:
            return self.default_factory()


# %% test missing key with factory
xs = dyct(lambda: -1)
print(f'{xs[1]=}')  # => 'xs[-1]=-1'

# %% test missing key plus dict args
xs = dyct(lambda: -1, {1: 1, 2: 2})
print(f'{xs=}')      # => 'xs={1: 1, 2: 2}'
print(f'{xs[-1]=}')  # => 'xs[-1]=-1'

# %% test missing key without factory
xs = dyct()
print(f'{xs[1]=}')   # raises KeyError
