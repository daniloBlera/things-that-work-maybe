#!/usr/bin/env python3
"""Exercising bad ideas for the sake of learning"""
# %% imports
from functools import reduce
from typing import Callable


# %% to display the function application order
def tee(func: Callable) -> Callable:
    """Print the input args and output of `func`, kinda like `tee`"""
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        print(f'* {func.__name__}({args=}, {kwargs=}) => {output}')
        return output
    return wrapper


# %% composition implementation
def compose(
    f: Callable, g: Callable, /, *rest: Callable, left_to_right: bool = False
) -> Callable:
    """Compose two or more functions

    By default, return a function composition that applies from right to left.
    Optionally, if `left_to_right` is `True`, the composition applies the
    functions from left to right like in a pipeline.
    """
    funcs: list[Callable] = [f, g, *rest]
    return reduce(lambda f, g: lambda *args, **kwargs: f(g(*args, **kwargs)),
                  funcs if not left_to_right else reversed(funcs))


# %% run tests
if __name__ == '__main__':
    intt = tee(int)
    floatt = tee(float)

    pipeline = compose(intt, floatt)
    pipeline(1.9)

    pipeline = compose(intt, floatt, left_to_right=True)
    pipeline(1.9)


# %% learn you a python for great good
def negate(x):
    return -x


if __name__ == '__main__':
    # shorthand to `list(map(fn, *args))` for printing
    mapl = compose(list, map)

    # apply the absolute value then multiply by -1
    print(mapl(compose(negate, abs), [5, -3, -6, 7, -3, 2, -19, 24]))

    # sum the elements for each sublist then multiply by -1
    print(mapl(compose(negate, sum), [[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
