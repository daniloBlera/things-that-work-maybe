#!/usr/bin/env python3
from functools import reduce


def partition(predicate, iterable):
    """Everything in a single beautifully unreadable code"""
    return reduce(lambda acc, x: acc[not predicate(x)].append(x) or acc,
                  iterable,
                  ([], []))


def partitionv2(predicate, iterable):
    """An uglier but more readable version of the same above"""
    def accumulate(acc, x):
        if predicate(x):
            acc[0].append(x)
        else:
            acc[1].append(x)
        return acc

    (true_cases, false_cases) = ([], [])
    return reduce(accumulate,
                  iterable,
                  (true_cases, false_cases))


def evenp(x: int) -> bool:
    return x % 2 == 0


(even_nums, odd_nums) = partitionv2(evenp, range(10))
print(list(even_nums))
print(list(odd_nums))
