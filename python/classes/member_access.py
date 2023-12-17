#!/usr/bin/env python3
"""Implementing javascript-ish member access"""
from collections import UserDict


class dyct(UserDict):
    def __getattr__(self, name):
        return self[name]


obj = dyct({"hello": "world", "something": "fishy"})
print(f"{obj=}")

# using __getitem__
print(f"{obj['hello']=}")

# using __getattr__
print(f"{obj.hello=}")
