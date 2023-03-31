#!/usr/bin/env python3
"""Exercising bad ideas for the sake of learning"""
from functools import reduce


# %% to display the function application order
def tee(func):
    """Print the input args and output of `func`, kinda like `tee`"""
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        print(f'* {func.__name__}({args=}, {kwargs=}) => {output}')
        return output
    return wrapper


# %% composition implementation
def compose(f, g, /, *rest, left_to_right=False):
    """Compose two or more functions

    By default, this returns a function composition that applies
    functions from the right to the left. Optionally, if `left_to_right` is
    set to `True`, the composition applies function from left to right like
    in a pipeline
    """
    funcs = [f, g, *rest]
    return reduce(lambda f, g: lambda *args, **kwargs: f(g(*args, **kwargs)),
                  funcs if not left_to_right else reversed(funcs))


if __name__ == '__main__':
    # %% tee-wrapped functions
    intt = tee(int)
    floatt = tee(float)

    # %% testing compose(f1, f2, f3)(x) == f1(f2(f3(x)))
    pipeline = compose(intt, floatt)
    pipeline(1.9)

    # %% testing compose(f1, f2, f3)(x) == f3(f2(f1(x)))
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
