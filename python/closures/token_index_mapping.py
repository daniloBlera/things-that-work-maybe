#!/usr/bin/env python3
"""Using function closures to configure token-index mapping functions"""
from typing import Callable, TypeVar


# %% definitions
T = TypeVar('T', str, int)
U = TypeVar('U', int, str)


# Using `str.split` instead of a more robust solution like `nltk`
def tokenize_text(text: str) -> list[str]:
    """Convert a string to tokens into a list of tokens"""
    return text.split()


def get_vocab(corpus: str) -> list[str]:
    """Get a sorted list of the corpus' unique tokens"""
    return sorted(set(tokenize_text(corpus)))


def get_token_index_maps(
        vocab: list[str]
) -> tuple[dict[str, int], dict[int, str]]:
    """Get the token->index and index->token dictionaries from the vocab

    Given a list of unique tokens, return two dictionaries. The first one
    maps tokens to integer indexes while the second does the inverse, mapping
    integers to tokens. These mappings are used to transform back and forth
    lists of string tokens to lists of integers.

    Arguments:
        vocab: list[str]
            A list of unique words.

    Return: (dict[str, int], dict[int, str])
        A tuple containing two dictionaries, the first maps tokens to integer
        indexes while the second does the opposite.
    """
    idx2token = dict(enumerate(['<PAD>', '<UNK>'] + vocab))
    token2idx = {token: idx for (idx, token) in idx2token.items()}
    return (token2idx, idx2token)


def create_converter(
        mapping: dict[T, U],
        default: U
) -> Callable[[list[T]], list[U]]:
    """Configure a function to convert list[T] -> list[U]

    Given a dictionary that maps from a source type T to a destination type U,
    and a default destination value (with type U), return a function that
    converts a list of elements from the source type into a list of elements
    from the destination type.

    Arguments:
        mapping: dict[T, U]
            A dictionary that maps keys from one type to values from another.
        default: U
            The default value for missing keys.

    Return: func: list[T] -> list[U]
        A function that convert list[T] -> list[U]
    """
    mapping = mapping.copy()

    def convert(keys: list[T]) -> list[U]:
        return [mapping.get(key, default) for key in keys]

    return convert


# For learning purposes only!
def create_unified_converter(
        mapping1: dict[T, U],
        default1: U,
        mapping2: dict[U, T],
        default2: T
) -> Callable:
    """Configure a function to convert between lists of tokens and indexes.

    Configure a single function closure that can convert a list of tokens into
    a list of integer indexes, as well as a list of integer indexes into a list
    of tokens:

        * convert(list[str]) -> list[int]
        * convert(list[int]) -> list[str]

    Arguments:
        mapping1: dict[T, U]
            A dictionary that maps in one direction.
        default1: U
            The default value for missing keys from `mapping1`.
        mapping2: dict[U, T]
            A dictionary that maps in the opposite direction to `mapping1`.
        default1: T
            The default value for missing keys from `mapping2`.

    Return: Callable[[?], ?]
        A function that can convert both ways:

            * func: list[str] -> list[int]
            * func: list[int] -> list[str]
    """
    t2u = create_converter(mapping1, default1)  # func: list[T] -> list[U]
    u2t = create_converter(mapping2, default2)  # func: list[U] -> list[T]

    # return `None` when converting invalid inputs to avoid exceptions
    def type_converter(inputs):
        if inputs:
            if isinstance(inputs[0], str):
                return t2u(inputs)              # list[U]
            elif isinstance(inputs[0], int):
                return u2t(inputs)              # list[T]

    return type_converter


# %% test body
# Defining the text corpus
corpus = '''Thousands of demonstrators have marched through London to protest
the war in Iraq and demand withdrawal British troops from that country'''

# Getting a sorted list of all unique words from the corpus
vocab = get_vocab(corpus)

# Get the token and index maps
(tok2idx, idx2tok) = get_token_index_maps(vocab)

# The default values for missing tokens
unk_tok = '<UNK>'
unk_idx = tok2idx[unk_tok]

# Configuring and assigning the conversion functions
encode_tokens = create_converter(tok2idx, unk_idx)
encode_indexes = create_converter(idx2tok, unk_tok)

# Converting a list of tokens into a list of indexes and back
tokens = tokenize_text(corpus)            #       str -> list[str]
indexes = encode_tokens(tokens)           # list[str] -> list[int]

# inversion should be valid if no out-of-vocabulary tokens are found
assert encode_indexes(indexes) == tokens

# %% testing the unified converter
convert = create_unified_converter(tok2idx, unk_idx, idx2tok, unk_tok)

tokens = ['Thousands', 'of', 'demonstrators', '<some_OOV_token>']
print(f'* {tokens=}')

indexes = convert(tokens)
print(f'* {indexes=}')

tokens = convert(indexes)
print(f'* {tokens=}')
