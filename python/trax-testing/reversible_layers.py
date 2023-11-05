#!/usr/bin/env python3
"""Experimenting with trax layers

Tested with:
 * python v3.11.5 (from pyenv)
 * trax v1.4.1

note: the reversible layers are included in the `trax.layers` namespace with the
current version of `trax` so no need to import the `trax.layers.reversible`.
"""
# %% 0. the imports
from trax import layers, shapes
from trax.fastmath import numpy as np

# %% 1.0 Residual Networks
# running a single layer pipeline
add_1 = layers.Fn(name="λ.x.x+1", f=lambda x: x + 1, n_out=1)
add_2 = layers.Fn(name="λ.x.x+2", f=lambda x: x + 2, n_out=1)
add_3 = layers.Fn(name="λ.x.x+3", f=lambda x: x + 3, n_out=1)

stack = np.array([1], dtype=np.int32)
print(f"{add_1(stack)=}")
print(f"{add_2(stack)=}")
print(f"{add_3(stack)=}")

# %% 1.1 branching
add_1_3 = layers.Branch(add_1, add_2, add_3)
print(add_1_3)

# %% 1.2 operating on a stack with a Branch layer
stack = (1, "hello", "world")
print(f"{stack=}")
print(f"{add_1_3(stack)=}")

# %% 1.3 branching with functions of different cardinalities
# note that `n_out` from `layers.Fn` can be used to unpack the output of the layer
add_x_y = layers.Fn(name="F: R2 -> R1", f=lambda x, y: x + y, n_out=1)
repeat_x = layers.Fn(name="F: R1 -> R3", f=lambda x: (x, x, x), n_out=3)
branch_3_ops = layers.Branch(add_1, add_x_y, repeat_x)

stack = (np.array([1]), np.array([3]), "hello", "world")
print(f"{stack=}")
print(f"{branch_3_ops(stack)=}")

# %% 1.4 residual behaviour from Branch
branch_2_ops = layers.Branch(add_1, None)

stack = (np.array([1]), "hello")
print(f"{stack=}")
print(f"{branch_2_ops(stack)=}")


# %% 1.5 a bootleg residual implementation
# this is just for learning purposes, use `trax.layers.Residual` instead
def residual(layer: layers.PureLayer | layers.Serial) -> layers.Serial:
    return layers.Serial(layers.Branch(layer, None), layers.Add())


residual_pipeline = residual(add_1)
stack = (np.array([1]), "hello", "world")
print(f"{stack=}")
print(f"{residual_pipeline(stack)=}")

# %% 1.6 building a pipeline of chained residual layers
f = layers.Fn(name="f:R1->R1", f=lambda x: 2 * x, n_out=1)
g = layers.Fn(name="g:R1->R1", f=lambda x: 10 * x, n_out=1)

pipeline = layers.Serial(
    layers.Residual(f), layers.Residual(g), layers.Residual(f), layers.Residual(g)
)

stack = [np.array([1]), "hello", "world"]
print(f"{stack=}")
print(f"{pipeline(stack)=}")

# %% 2.0 Reversible Residual Networks
# basic reversible operations - `Dup` and `Swap`
duplicate = layers.Dup()
print(f"{duplicate(np.array([1]))=}")

swap = layers.ReversibleSwap()
print(f"{swap([np.array([1]), np.array([5])])=}")

# %% 2.1 using a ReversibleSwap layer's "reverse" pass
stack = (np.array([1]), np.array([5]))
print(f"{stack=}")

reversed_stack = swap.reverse(stack)
print(f"{reversed_stack=}")

print(f"{swap.reverse(reversed_stack)=}")

# %% 2.2 building a reversible model
# using the function layers `f` and `g` defined in section `1.6`
block = [
    layers.ReversibleHalfResidual(f),
    layers.ReversibleSwap(),
    layers.ReversibleHalfResidual(g),
    layers.ReversibleSwap(),
]

blocks = [block, block]

model = layers.Serial(
    layers.Dup(),
    blocks,
    layers.Concatenate(),
)

# %% 2.3 testing the model pipeline
x1 = np.array([1])
model.init(shapes.signature(x1))
y1 = model(x1)
print(y1)
