#!/usr/bin/env python3
from more_itertools import first


# To desambiguate a default argument from a user-provided one
__SENTINEL = object()


def find_if(predicate, iterable, default=__SENTINEL, /):
    """Return the first element where `predicate(item)` is True

    Usage examples:
        # find the first even number
        find_if(lambda x: x % 2 == 0, [1, 2, 3, 4, 5]) => 2

        # find the first even number (raises ValueError)
        find_if(lambda x: x % 2 == 0, [1, 3, 5]) # raises ValueError

        # find the first even number, otherwise return None
        find_if(lambda x: x % 2 == 0, [1, 3, 5], None) => None

    Arguments:
        predicate:
            A function of one argument that returns a generalized boolean.

        iterable:
            An iterable of things.

        default:
            If unset, this function will raise a ValueError if no element
            from `iterable` passes the `predicate`. Otherwise, return the
            value given without raising ValueError.

    Return
        The first element from `iterable` that passes the `predicate`, if
        it exists.
    """
    if default is __SENTINEL:
        return first(filter(predicate, iterable))
    else:
        return first(filter(predicate, iterable), default)


print(f'{find_if(lambda x: x % 2 == 0, [1, 2, 3, 4, 5])}')  # prints `2`
print(f'{find_if(lambda x: x % 2 == 0, [1, 3, 5], None)}')  # prints `None`
