#!/usr/bin/env python3
"""Trying to mimick function currying

Yes, this is a bad idea. Anyway...

This code wraps a function with a `functools.partial` object and calls it if it
doesn't have remaining/unbound parameters, for example, assume `function`
has the signature `(x: int, y: int) -> int`:

    func = Curry(function)   => f(x: int, y: int) -> int
    func(1)                  => f(y: int) -> int
    func(1)(2) == func(1, 2) => 2
"""
from functools import partial
from inspect import signature as sig


# %% our function wrapper
class Curry():
    def __init__(self, func):
        self.func = partial(func)

    def __repr__(self):
        return f'f{sig(self.func)}'

    def __str__(self):
        return self.__repr__()

    def __call__(self, *args):
        output = partial(self.func, *args)

        if len(sig(output).parameters) <= 0:
            return output()
        else:
            return Curry(output)


# %% the function to be curried
def foonction(x: int, y: int, z: int) -> int:
    return x + y + z


# %% testing argument consumption
func = Curry(foonction)
print(func)  # => f(x: int, y: int, z: int) -> int

f1 = func(1)
print(f1)    # => f(y: int, z: int) -> int

f2 = f1(2)
print(f2)    # => f(z: int) -> int

f3 = f2(3)
print(f3)    # => 6
