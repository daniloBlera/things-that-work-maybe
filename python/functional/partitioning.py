#!/usr/bin/env python3
"""Group elements from iterable in two groups, according to a predicate

For learning purposes only. Use `more_itertools.partition` instead
"""
from functools import reduce


# Everything in a single, beautifully unreadable code
def partition(predicate, iterable):
    """Partition the iterable according to the predicate"""
    return reduce(lambda acc, x: acc[not predicate(x)].append(x) or acc,
                  iterable,
                  ([], []))


# An uglier but more readable version of the previous function
def partition_alt(predicate, iterable):
    """Partition the iterable according to the predicate"""
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


# (true_cases, false_cases) = partition(predicate, iterable)
(even_nums, odd_nums) = partition(evenp, range(10))
print(even_nums)
print(odd_nums)
