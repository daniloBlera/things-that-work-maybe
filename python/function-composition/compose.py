#!/usr/bin/env python3
"""Exercising bad ideas for the sake of learning"""
from collections import Counter
from itertools import pairwise
from functools import reduce


def composeN(*funcs):
    """Compose an arbitrary number of functions"""
    def compose2(f, g):
        return lambda *args, **kwargs: f(g(*args, **kwargs))

    return reduce(compose2, funcs)


def compose(*funcs):
    """compose(f1, f2, ..., fN) => f1 . f2 . ... . fN

    Same as the composeN function but more compact and harder to read
    """
    return reduce(lambda f, g: lambda *args, **kwargs: f(g(*args, **kwargs)),
                  funcs)


comp = compose(int, float, str, float, int)
print(f'{comp(1) == 1 = }')


# Some examples that you should probably avoid
def flatten(xss):
    """Flatten an arbitrarily nested list

    Implementation based on the `flatten` function from the
    chapter 4 of the book "On Lisp".
    """
    def rec(x, acc):
        if not isinstance(x, list):
            return [x] + acc
        elif len(x) == 0:
            return acc
        else:
            [head, *tail] = x
            return rec(head, rec(tail, acc))

    return rec(xss, [])


# List-fying a pairwise generator because why not?
pairwise_l = compose(list, pairwise)

# Flatten nested list then count elements
flatten_and_count = compose(Counter, flatten)
