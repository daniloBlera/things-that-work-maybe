#!/usr/bin/env python3
from more_itertools.recipes import partition


def evenp(x):
    return x % 2 == 0


# (false cases, true cases) = partition(pred, iterable)
(odd_nums, even_nums) = partition(evenp, range(10))
print(list(odd_nums))
print(list(even_nums))
