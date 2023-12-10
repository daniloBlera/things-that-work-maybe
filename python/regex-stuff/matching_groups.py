#!/usr/bin/env python3
"""Parsing a string with regex for its groups"""
import re


# A standoff (entity) annotation file has the
# following EBNF-kind-of syntax:
#
#     <entity> ::= '<id>\t<type> <boundary>\t<text>'
#         <id> ::= 'T[0-9]+'
#       <type> ::= '<str>'
#   <boundary> ::= '<int> <int>' | '<int> <int>;<boundary>'
#       <text> ::= '<str>'
#
# For example
line = "T2\tHabitat 77 98;119 129;144 151\tsurface microbiota of Gorgonzola cheeses"

# Specifying the above capture groups with
# parenthesis-delimited expressions
pattern = re.compile(r"(\w+)\t(\w+) ([0-9 ;]+)\t(.+)")

# Display the groups if the string is fully matched
if result := pattern.fullmatch(line):
    print(f"* Line: {repr(line)}")
    print(f"* Groups: {result.groups()}")
else:
    print("Entity matching failed")
