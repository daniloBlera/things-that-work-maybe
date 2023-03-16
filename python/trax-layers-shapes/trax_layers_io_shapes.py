#!/usr/bin/env python3
"""Checking input/output shapes of trax layers"""
import jax
import jax.numpy as jnp
import trax.layers as tl
from trax.shapes import signature


# %% define l2 normalization
def normalize(x: jax.Array) -> jax.Array:
    """Apply L2 normalization to array (on the last axis)"""
    return x / jnp.linalg.norm(x, axis=-1, keepdims=True)


# %% shape exploration
vocab_size = 500
embedding_dim = 128     # The number of embedding features per word

# The changes in input shapes
#                        x => (batch_size, max_len)
#     a1 = tl.Embedding(x) => (batch_size, max_len, embedding_dim)
#         a2 = tl.LSTM(a1) => (batch_size, max_len, embedding_dim)
#         a3 = tl.Mean(a2) => (batch_size, embedding_dim)
#           a4 = tl.Fn(a3) => (batch_size, embedding_dim)
#
# where `batch_size` is the number of tokenized sentences and
# `max_len` is the length of all tokenized sentences (shorter
# sentences were padded to fit the input ndarray).

# x := [<tokenized-sentence-1>,
#                 ⋮           ,
#       <tokenized-sentence-N>]
x = jnp.array([[1, 2, 3],
               [4, 5, 6]], # two sentences, both three tokens long
              dtype=jnp.int32)

print(f'* {x.shape=}')

# %% embedding
emb = tl.Embedding(vocab_size=vocab_size, d_feature=embedding_dim)
emb.init(signature(x))
a1 = emb(x)
print(f'* Emb({x.shape=}) -> {a1.shape=}')

# %% lstm
lstm = tl.LSTM(n_units=embedding_dim)
lstm.init(signature(a1))
a2 = lstm(a1)
print(f'* LSTM({a1.shape=}) -> {a2.shape=}')

# %% mean
# Combine all tokens from a sentence into a single vector using the mean (for
# question duplicates with siamese networks, for example).
mean = tl.Mean(axis=1)
mean.init(signature(a2))
a3 = mean(a2)
print(f'* Mean({a2.shape=}) -> {a3.shape=}')

# %% l2-norm
fn = tl.Fn('Normalize', lambda x: normalize(x))
fn.init(signature(a3))
a4 = fn(a3)
print(f'* Norm({a3.shape=}) -> {a4.shape=}')
