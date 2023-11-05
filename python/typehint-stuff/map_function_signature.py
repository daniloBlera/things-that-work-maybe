#!/usr/bin/env python3
"""Testing function cardinality vs number of iterables from `map`

To see how the typeshed for the built-in `map` was implemented, see:
https://github.com/python/typeshed/blob/2ce9dcd5fbfede964483b43abc5918dda9c7ef4f/stdlib/builtins.pyi#L1473
tested under python 3.12.0 and mypy 1.6.1
"""

ns = [1, 2, 3]

# correctly matching the callable's cardinality and the number of iterables
map(lambda x: (x,), ns)
map(lambda x, y: (x, y), ns, ns)
map(lambda x, y, z: (x, y, z), ns, ns, ns)

# mismatched numbers -- mypy/pyright should complain about the following lines
map(lambda x, y: (x, y), ns)
map(lambda x: (x,), ns, ns)

# testing with more than six iterables, as `map` is hardcoded with annotations
# to handle up to six iterables
map(lambda x1, x2, x3, x4, x5, x6, x7: (x1, x2, x3, x4, x5, x6, x7),
                                        ns, ns, ns, ns, ns, ns)

map(lambda x1, x2, x3, x4, x5, x6, x7: (x1, x2, x3, x4, x5, x6, x7),
                                        ns, ns, ns, ns, ns, ns, ns)

map(lambda x1, x2, x3, x4, x5, x6, x7: (x1, x2, x3, x4, x5, x6, x7),
                                        ns, ns, ns, ns, ns, ns, ns, ns)
