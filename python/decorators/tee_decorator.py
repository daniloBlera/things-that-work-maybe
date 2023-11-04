#!/usr/bin/env python3
"""Mimick the behaviour of `tee` when calling functions """
from typing import Sequence, TypeVar


def tee(func):
    """Print the output of func to STDOUT then return it"""
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        print(output)
        return output

    return wrapper


# our generic type `T` until `mypy` adds support for PEP695
T = TypeVar('T')

@tee
def butlast(iterable: Sequence[T]) -> Sequence[T]:
    return iterable[:-1]


trigram = [1, 2, 3]
bigram = butlast(trigram)
unigram = butlast(bigram)
empty = butlast([1])
