#!/usr/bin/env pytho3
from dataclasses import dataclass
from typing import Any, Final
"""Indicating that symbols should't be reassigned

An editor equipped with a linter (e.g.: pyright) should warn you of the issues
below.

Note, preventing reassignment of symbols is not the
same as immutability.
"""

# Indicating a symbol shouldn't be reassigned
DEFAULT_PATH: Final = './resources'
DEFAULT_PATH = 'newvalue'       # not ok


# Indicate members should't be reassigned
@dataclass(frozen=True)
class Something:
    items: list[Any]


something = Something([1, 2, 3])
something.items = []            # not ok
something.items.append(999)     # ok

