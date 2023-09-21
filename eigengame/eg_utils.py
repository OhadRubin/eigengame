# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Helper functions for the distributed TPU implementation of EigenGame."""
import collections
import enum
import functools
import os
from typing import Callable, Dict, NamedTuple, Optional, Tuple, Union
from absl import logging

import chex
import jax
import jax.numpy as jnp


import numpy as np
from flax import struct





class SplitVector(struct.PyTreeNode):
  """Generalized Eigenvector object used for CCA, PLS, etc.

  Concatenations and splits are some of the slowest things you can do on a TPU,
  so we want to keep the portions of the eigenvectors responsible for each data
  source separate.
  """
  x: chex.ArrayTree
  y: chex.ArrayTree


class AuxiliaryParams(struct.PyTreeNode):
  r"""Container for auxiliary variables for $\gamma$-EigenGame."""
  b_vector_product: chex.ArrayTree
  b_inner_product_diag: chex.Array


EigenGameGradientFunction = Callable[..., Tuple[chex.ArrayTree,
                                                AuxiliaryParams]]

EigenGameQuotientFunction = Callable[..., Tuple[chex.ArrayTree, chex.ArrayTree]]



def get_spherical_gradients(
    gradient: chex.ArrayTree,
    eigenvectors: chex.ArrayTree,
) -> chex.ArrayTree:
  """Project gradients to a perpendicular to each of the eigenvectors."""
  tangential_component_scale = tree_einsum(
      'l..., l...-> l',
      gradient,
      eigenvectors,
      reduce_f=lambda x, y: x + y,
  )
  tangential_component = tree_einsum_broadcast(
      'l..., l -> l... ',
      eigenvectors,
      tangential_component_scale,
  )

  return jax.tree_map(
      lambda x, y: x - y,
      gradient,
      tangential_component,
  )


def normalize_eigenvectors(eigenvectors: chex.ArrayTree) -> chex.ArrayTree:
  """Normalize all eigenvectors."""
  squared_norm = jax.tree_util.tree_reduce(
      lambda x, y: x + y,
      jax.tree_map(
          lambda x: jnp.einsum('k..., k... -> k', x, x),
          eigenvectors,
      ))
  return jax.tree_map(
      lambda x: jnp.einsum('k..., k -> k...', x, 1 / jnp.sqrt(squared_norm)),
      eigenvectors,
  )



def get_local_slice(
    local_identity_slice: chex.Array,
    input_vector_tree: chex.ArrayTree,
) -> chex.ArrayTree:
  """Get the local portion from all the eigenvectors.

  Multiplying by a matrix here to select the vectors that we care about locally.
  This is significantly faster than using jnp.take.

  Args:
    local_identity_slice: A slice of the identity matrix denoting the vectors
      which we care about locally
    input_vector_tree: An array tree of data, with the first index querying all
      the vectors.

  Returns:
      A pytree of the same structure as input_vector_tree, but with only the
      relevant vectors specified in the identity.
  """

  def get_slice(all_vectors):
    print(all_vectors.shape, local_identity_slice.shape)
    return jnp.einsum(
        'k..., lk -> l...',
        all_vectors,
        local_identity_slice,
    )

  return jax.tree_map(get_slice, input_vector_tree)


def per_vector_metric_log(
    metric_name: str,
    metric: chex.Array,
) -> Dict[str, float]:
  """Creates logs for each vector in a sortable way."""
  # get the biggest index length
  max_index_length = len(str(len(metric)))

  def get_key(index: int) -> str:
    """Adds metrix prefix and pads the index so the key is sortable."""
    padded_index = str(index).rjust(max_index_length, '0')
    return metric_name + '_' + padded_index

  return {get_key(i): value for i, value in enumerate(metric)}


def tree_einsum(
    subscripts: str,
    *operands: chex.ArrayTree,
    reduce_f: Optional[Callable[[chex.Array, chex.Array], chex.Array]] = None
) -> Union[chex.ArrayTree, chex.Array]:
  """Applies an leaf wise einsum to a list of trees.

  Args:
    subscripts: subscript string denoting the einsum operation.
    *operands: a list of pytrees with the same structure. The einsum will be
      applied leafwise.
    reduce_f: Function denoting a reduction. If not left empty, this calls a
      tree reduce on the resulting tree after the einsum.

  Returns:
      A pytree with the same structure as the input operands if reduce_f.
      Otherwise an array which is the result of the reduction.
  """
  einsum_function = functools.partial(jnp.einsum, subscripts)
  mapped_tree = jax.tree_map(einsum_function, *operands)
  if reduce_f is None:
    return mapped_tree
  else:
    return jax.tree_util.tree_reduce(reduce_f, mapped_tree)


def tree_einsum_broadcast(
    subscripts: str,
    tree: chex.ArrayTree,
    *array_operands: chex.Array,
    reduce_f: Optional[Callable[[chex.Array, chex.Array], chex.Array]] = None
) -> Union[chex.ArrayTree, chex.Array]:
  """Applies an einsum operation on a list of arrays to all leaves of a tree.

  Args:
    subscripts: subscript string denoting the einsum operation. The first
      argument must denote the tree leaf, followed by the list of arrays.
    tree: A pytree. The einsum will be applied with the leaves of this tree in
      the first argument.
    *array_operands: A list of arrays. The sinsum with these arrays will be
      mapped to each leaf in the tree.
    reduce_f: Function denoting a reduction. If not left empty, this calls a
      tree reduce on the resulting tree after the einsum.

  Returns:
      A pytree with the same structure as the input tree. if reduce_f.
      Otherwise an array which is the result of the reduction.
  """
  einsum_function = lambda leaf: jnp.einsum(subscripts, leaf, *array_operands)
  mapped_tree = jax.tree_map(einsum_function, tree)
  if reduce_f is None:
    return mapped_tree
  else:
    return jax.tree_util.tree_reduce(reduce_f, mapped_tree)



