#!/usr/bin/env python3
from more_itertools import first


# To desambiguate a default argument from a user-provided one
__SENTINEL = object()


def find_if(predicate, iterable, default=__SENTINEL, /):
    """Return the first item where `predicate(item)` is True

    Example: find the first tokenized sentence that has a '<unk>' token
        find_if(lambda s: '<unk>' in s, tokenized_sentences)

        find_if(lambda x: x % 2 == 0, [1, 2, 3, 4, 5]) => 2
        find_if(lambda x: x % 2 == 0, [1, 3, 5]) # raises ValueError
        find_if(lambda x: x % 2 == 0, [1, 3, 5], None) => None

    Arguments:
        predicate:  A function to test every element from iterable.
        iterable:   An iterable of things.
        default:    If unset, this function will raise a ValueError if no
                    element from `iterable` passes the `predicate`. Otherwise,
                    return the value given without raising ValueError.
    """
    if default is __SENTINEL:
        return first(filter(predicate, iterable))
    else:
        return first(filter(predicate, iterable), default)


print(f'{find_if(lambda x: x % 2 == 0, [1, 2, 3, 4, 5])}')  # prints `2`
print(f'{find_if(lambda x: x % 2 == 0, [1, 3, 5], None)}')  # prints `None`