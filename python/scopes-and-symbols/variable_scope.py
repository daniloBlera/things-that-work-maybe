#!/usr/bin/env python3
"""Testing the effects of functions that rely on externally defined symbols"""

# Example 1 - Dynamic binding effects
# creating a global variable
var = 1


# defining two functions that print the value of some symbol `var`
def func1() -> None:
    print(f"{var=}")


def func2(var=var) -> None:
    print(f"{var=}")


# reassigning the value of our global symbol after evaluating the
# function definitions
var = 2

# while `func1` uses the current value of `var`, `func2` uses the value at the
# time of the function definition
func1()  # prints 'var=2' to stdout
func2()  # prints 'var=1' to stdout


# Example 2 - functions with variables defined in a listcomp
for func in [(lambda: i) for i in range(3)]:
    print(f"{func()=}")

# the statement above prints (maybe unexpectedly)
#   func()=2
#   func()=2
#   func()=2

# in the statement below, doing `i = i` in the lambda parameters create a local
# binding with name `i` and assigns it to the value of the outer `i` at the time
# of the iteration
for func in [(lambda i=i: i) for i in range(3)]:
    print(f"{func()=}")

# differently, the statement above prints (probably the intended result)
#    func()=0
#    func()=1
#    func()=2

# to make it a little easier to understand
for func in [(lambda x=i: x) for i in range(3)]:
    print(f"{func()=}")

# again, the statement above prints the same result as the statement before it
#    func()=0
#    func()=1
#    func()=2
