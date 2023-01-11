#!/usr/bin/env python3
from functools import reduce


# Everything in a single beautifully unreadable code
def partition(predicate, iterable):
    """Partition the iterable according to the predicate"""
    return reduce(lambda acc, x: acc[not predicate(x)].append(x) or acc,
                  iterable,
                  ([], []))


# An uglier but more readable version of the same above
def partitionv2(predicate, iterable):
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


(even_nums, odd_nums) = partitionv2(evenp, range(10))
print(list(even_nums))
print(list(odd_nums))
