#!/usr/bin/env python3
"""Using python's match-case to handle different objects"""
import io
import os
import pathlib
import typing
from dataclasses import dataclass


# %% matching string, pathlib.Path or file object
def type_matcher(filespec: str | os.PathLike | typing.TextIO) -> None:
    match filespec:
        case str():
            print('* matched str')
        case os.PathLike():
            print('* matched Path')
        case io.IOBase():
            print('* matched file object')
        case _:
            print('* no match')


spec = 'some_file_path'
type_matcher(spec)
type_matcher(pathlib.Path(spec))
with open(spec, 'w') as fd:
    type_matcher(fd)
os.remove(spec)


# %% matching class instance
@dataclass
class Point:
    x: int
    y: int
    z: int


point = Point(1, 2, 3)
match point:
    case Point(x=1, y=_, z=_):
        print('* matched specific Point object configuration')
    case Point():
        print('* matched Point object')
    case _:
        print('* no match')
