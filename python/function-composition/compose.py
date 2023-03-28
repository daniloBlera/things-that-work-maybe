#!/usr/bin/env python3
"""Exercising bad ideas for the sake of learning"""
from functools import reduce


# %% debugging function order
def tee(func):
    """Print the input args and output of `func`, kinda like `tee`"""
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        print(f'* {func.__name__}({args=}, {kwargs=}) => {output}')
        return output
    return wrapper


# %% composition implementation
def compose(f, g, *rest, left_to_right=False):
    funcs = [f, g, *rest]

    return reduce(lambda f, g: lambda *args, **kwargs: f(g(*args, **kwargs)),
                  funcs if not left_to_right else reversed(funcs))


# %% tee-wrapped functions
intt = tee(int)
floatt = tee(float)

# %% testing compose(f1, f2, f3)(x) == f1(f2(f3(x)))
pipeline = compose(intt, floatt)
pipeline(1.9)

# %% testing compose(f1, f2, f3)(x) == f3(f2(f1(x)))
pipeline = compose(intt, floatt, left_to_right=True)
pipeline(1.9)
