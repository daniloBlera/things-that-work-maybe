#!/usr/bin/env python3
"""Mimick the behaviour of `tee` when calling functions"""


def tee(func):
    """Print the output of func to STDOUT then return it"""
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        print(output)
        return output

    return wrapper


@tee
def butlast(iterable):
    return iterable[:-1]


trigram = [1, 2, 3]
bigram = butlast(trigram)
unigram = butlast(bigram)
