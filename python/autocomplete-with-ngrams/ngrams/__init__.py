#!/usr/bin/env python3
"""This module provides a simple word-based autocompletion using
n-gram language models.

The implementation was based on the contents from the second week of
coursera's "Natural Language Processing with Probabilistic Models"
course, as well as the chapter three of the "Speech and Language
Processing" book.

To use this module, import the four functions below from its top
module namespace:

    * get_tokenized_sentences_from
    * preprocess_data
    * count_ngrams
    * get_suggestions

The rest of the symbols and kludges found in this module are for
internal usage mostly because avoiding namespace taxonomies in general
is a good thing.
"""
from ngrams.tools import (
        get_tokenized_sentences_from,
        preprocess_data,
        count_ngrams,
        get_suggestions
)
