#!/usr/bin/env python3
"""Learn You a Python for Great Good

Trying to compose partially applied functions with currying

This was based off of the example from the chapter 6 of the book
"Learn You a Haskell for Great Good"
"""
from itertools import takewhile

# the modules in the current directory
from currying import Curry
from composition import compose


# %% defining the required functions and arguments
def oddp(x):
    return x % 2 != 0


def square(x):
    return pow(x, 2)


# reversing the order of x and y so we can set y before x with currying
lt = Curry(lambda x, y: y < x)
takewhilec = Curry(takewhile)
filterc = Curry(lambda p, xs: filter(p, xs))
mapc = Curry(lambda f, xs: map(f, xs))
numbers = range(1, 10_000)

# %% defining the function composition
# what this composition is meant to do:
#
#   1. square every element;
#   2. filter the odd values;
#   3. take all values before the first greater then or equal to 50;
#   4. sum all values from the previous step
pipeline = compose(sum, takewhilec(lt(50)), filterc(oddp), mapc(square))
result = pipeline(numbers)
print(f'* sum of values: {result}')

# %% using more descriptive names
filter_odds = filterc(oddp)
map_square = mapc(square)
takewhile_lt50 = takewhilec(lt(50))

pipeline = compose(sum, takewhile_lt50, filter_odds, map_square)
result = pipeline(numbers)
print(f'* sum of values: {result}')

# %% evaluating each step sequentially
squares = list(map(square, numbers))
print(f'* {squares[:10]}...')

odd_squares = list(filter(oddp, squares))
print(f'* {odd_squares[:10]}...')

filtered = list(takewhile(lambda x: x < 50, odd_squares))
print(f'* {filtered[:10]}')

result = sum(filtered)
print(f'* sum of values: {result}')
