#!/usr/bin/env python3
"""Testing unreachable code for python v3.11 or later

see also: https://typing.readthedocs.io/en/latest/source/unreachable.html
"""
from typing import Never


# Testing how pyright warns about unreachable code. Hover the mouse cursor over
# the symbol `something` below inside every if-else block to see the tooltips
# for possible values that pass the type narrowing, if they exist.
something: str | None = "hello"
if something:
    something  # (variable) something: Literal['hello']
else:
    something  # (variable) something: Never

if not something:
    something  # (variable) something: Never
else:
    something  # (variable) something: Literal['hello']

something: str | None = ""
if something:
    something  # (variable) something: Never
else:
    something  # (variable) something: Literal['']

if not something:
    something  # (variable) something: Literal['']
else:
    something  # (variable) something: Never


# Testing functions that either should never return or should never be called
# todo: make it explicit that returning `typing.Never` marks statements after
# it as unreachable.
def should_not_return() -> Never:
    """This function should never return"""
    ...


def should_not_be_called(arg: Never):
    """This function should not be called"""
    ...


def assert_never(arg: Never) -> Never:
    raise AssertionError("Expected code to be unreachable")


# No value will ever satisfy the bottom-type of `arg`
should_not_be_called(arg=None)
# this function can be used to perform exaustiveness checking

myvar = 1
if myvar:
    # the language server should complain about the call below as `myvar`
    # doesn't satisfies the function's argument type
    # assert_never(myvar)
    myvar
else:
    # the statement below should be ok since it's unreachable
    assert_never(myvar)

if myvar:
    myvar
else:
    should_not_return()

result = should_not_return()
# code below the previous statement should be marked unreachable
myvar = "hello"
myvar = "world"
