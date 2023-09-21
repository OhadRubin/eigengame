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

"""Different Eigengame gradients and their associated helpers.


Einsum legend:
...: the massive data dimension of the eigenvector which we're trying to do
PCA/CCA/etc. on. The input data may not be contiguious, and may consist of
multiple shapes in a pytree (e.g. images, activations of multiple layers of a
neural net). Hence we use ellipses to denote it.
l: local eigenvector index -- the number of eigenvectors per machine.
k: global eigenvector index -- number of eigenvectors over all machines.
b: batch dimension, batch size per machine.

"""
from typing import Tuple, Optional

import chex
from eigengame import eg_utils
import jax
import jax.numpy as jnp
import numpy as np

SplitVector = eg_utils.SplitVector


def pca_unloaded_gradients(
    local_eigenvectors: chex.ArrayTree,
    sharded_data: chex.ArrayTree,
    mask: chex.Array,
    sliced_identity: chex.Array,
) -> chex.ArrayTree:
  """Calculates the gradients for each of the eigenvectors for EG unloaded.

  Calculates the gradients of eigengame unloaded (see Algorithm 1. of
  https://arxiv.org/pdf/2102.04152.pdf)

  This is done in a distributed manner. Each TPU is responsible for updating
  the whole of a subset of eigenvectors, and get a different batch of the data.
  The training data is applied and then aggregated to increase the effective
  batch size.

  Args:
    local_eigenvectors: eigenvectors sharded across machines in
      a ShardedDeviceArray. ArrayTree with leaves of shape:
      [eigenvectors_per_device, ...]
    sharded_data: Training data sharded across machines. ArrayTree with leaves
      of shape: [batch_size_per_device, ...]
    mask: Mask with 1s below the diagonal and then sharded across devices, used
      to calculate the penalty. Shape of: [eigenvectors_per_device,
        total_eigenvector_count]
    sliced_identity: Sharded copy of the identity matrix used to calculate the
      reward. Shape of: [eigenvectors_per_device, total_eigenvector_count]

  Returns:
    Gradients for the local eigenvectors. ArrayTree with leaves of shape:
   [eigenvectors_per_device, ...] on each device.
  """
  # Collect the matrices from all the other machines.
  all_eigenvectors = jax.lax.all_gather(
      local_eigenvectors,
      axis_name='devices',
      axis=0,
      tiled=True,
  )
  # Calculate X^TXv/b for all v. This and v are all you need to construct the
  # gradient. This is done on each machine with a different batch.
  data_vector_product = eg_utils.tree_einsum(
      'b..., k... -> bk',
      sharded_data,
      all_eigenvectors,
      reduce_f=lambda x, y: x + y,
  )
  # divide by the batch size
  data_vector_product /= data_vector_product.shape[0]  # pytype: disable=attribute-error  # numpy-scalars
  sharded_gram_vector = eg_utils.tree_einsum_broadcast(
      'b..., bk -> k...',
      sharded_data,
      data_vector_product,
  )
  # Aggregate the gram matrix products across machines.
  gram_vector = jax.lax.pmean(
      sharded_gram_vector,
      axis_name='devices',
  )
  # Calculate <Xv_i, Xv_j> for all vectors on each machine.
  scale = eg_utils.tree_einsum(
      'l..., k... -> lk',
      local_eigenvectors,
      gram_vector,
      reduce_f=lambda x, y: x + y,
  )

  # The weights is 1 below diagonals
  penalty = eg_utils.tree_einsum_broadcast(
      'k..., lk, lk -> l...',
      all_eigenvectors,
      scale,
      mask,
  )

  reward = eg_utils.get_local_slice(sliced_identity, gram_vector)  # XtXv_i
  return jax.tree_map(lambda x, y: x - y, reward, penalty)




def _generalized_eg_matrix_inner_products(
    local_eigenvectors: chex.ArrayTree,
    all_eigenvectors: chex.ArrayTree,
    b_vector_product: chex.ArrayTree,
    a_vector_product: chex.ArrayTree,
    aux_b_vector_product: chex.ArrayTree,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array,]:
  """Compute various inner product quantities used in the gradient.

  In particular, the loss requires the various forms of inner product
  <v_i, Av_j> in order to function. This function calculates all of them to keep
  the gradient function less cluttered.


  Args:
    local_eigenvectors: ArrayTree with local copies of generalised eigen
      vectors with leaves of shape [l, ...] denoting v_l.
    all_eigenvectors: ArrayTree with all generalised eigen vectors with leaves
      of shape [k, ...] denoting v_k
    b_vector_product: ArrayTree with all B matrix eigenvector products with
      leaves of shape [k, ...] denoting Bv_k
    a_vector_product: ArrayTree with all A matrix eigenvector products with
      leaves of shape [k, ...] denoting Av_k
    aux_b_vector_product: ArrayTree with all B matrix eigenvector products from
      the axuiliary variable with leaves of shape [k, ...] denoting Bv_k

  Returns:
    Tuple of arrays containing:
      local_b_inner_product: <v_l, Bv_k>
      local_a_inner_product: <v_l, Av_k>
      b_inner_product_diag: <v_k, Bv_k>
      a_inner_product_diag: : <v_k, Bv_k>
  """

  # Evaluate the various forms of the inner products used in the gradient
  local_aux_b_inner_product = eg_utils.tree_einsum(
      'l... , k... -> lk',
      local_eigenvectors,
      aux_b_vector_product,
      reduce_f=lambda x, y: x + y,
  )  # v_l B v_k, used in the penalty term
  local_a_inner_product = eg_utils.tree_einsum(
      'l..., k... -> lk',
      local_eigenvectors,
      a_vector_product,
      reduce_f=lambda x, y: x + y,
  )  # v_l A v_k, used in the penalty term
  b_inner_product_diag = eg_utils.tree_einsum(
      'k..., k... -> k',
      all_eigenvectors,
      b_vector_product,
      reduce_f=lambda x, y: x + y,
  )  # v_k B v_k, used in the penalty and reward terms
  a_inner_product_diag = eg_utils.tree_einsum(
      'k..., k... -> k',
      all_eigenvectors,
      a_vector_product,
      reduce_f=lambda x, y: x + y,
  )  # v_k A v_k, used in the reward term
  return (
      local_aux_b_inner_product,
      local_a_inner_product,
      b_inner_product_diag,
      a_inner_product_diag,
  )


def _generalized_eg_gradient_reward(
    local_b_inner_product_diag: chex.Array,
    local_a_inner_product_diag: chex.Array,
    local_b_vector_product: chex.ArrayTree,
    local_a_vector_product: chex.ArrayTree,
) -> chex.ArrayTree:
  """Evaluates the reward term for the eigengame gradient for local vectors.

  This attempts to maximise the rayleigh quotient for each eigenvector, and
  by itself would find the eigenvector with the largest generalized eigenvalue.

  The output corresponds to the equation for all l:
    <v_l,Bv_l>Av_l - <v_l,Av_l>Bv_l

  Args:
    local_b_inner_product_diag: Array of shape [l] corresponding to <v_l,Bv_l>.
    local_a_inner_product_diag: Array of shape [l] corresponding to <v_l,Av_l>.
    local_b_vector_product: ArrayTree with local eigen vectors products with
      B with leaves of shape [l, ...] denoting Bv_l.
    local_a_vector_product: ArrayTree with local eigen vectors products with
      A with leaves of shape [l, ...] denoting Av_l.

  Returns:
    The reward gradient for the eigenvectors living on the current machine.
    Array tree with leaves of shape [l, ...]
  """
  # Evaluates <v_l,Bv_l>Av_l
  scaled_a_vectors = eg_utils.tree_einsum_broadcast(
      'l..., l -> l...',
      local_a_vector_product,
      local_b_inner_product_diag,
  )
  # Evaluates <v_l,Av_l>Bv_l
  scaled_b_vectors = eg_utils.tree_einsum_broadcast(
      'l..., l -> l...',
      local_b_vector_product,
      local_a_inner_product_diag,
  )
  return jax.tree_map(
      lambda x, y: x - y,
      scaled_a_vectors,
      scaled_b_vectors,
  )


def _generalized_eg_gradient_penalty(
    local_b_inner_product: chex.Array,
    local_a_inner_product: chex.Array,
    local_b_inner_product_diag: chex.Array,
    b_inner_product_diag: chex.Array,
    local_b_vector_product: chex.ArrayTree,
    b_vector_product: chex.ArrayTree,
    mask: chex.Array,
    b_diag_min: float = 1e-6) -> chex.ArrayTree:
  r"""Evaluates the penalty term for the eigengame gradient for local vectors.

  This attempts to force each eigenvalue to be B orthogonal (i.e. <v,Bw> = 0) to
  all its parents. Combining this with the reward terms means each vector
  learns to maximise the eigenvalue whilst staying orthogonal, giving us the top
  k generalized eigenvectors.


  The output corresponds to the equation for all l on the local machine:
   \\sum_{k<l}(<v_l,Bv_l>Bv_k - <v_l,Bv_k>Bv_l) <v_l,Av_k>/<v_k,Bv_k>

  Note: If the gradient returned must be unbiased, then any estimated quantities
  in the formula below must be independent and unbiased (.e.g, the numerator of
  the first term (<v_l,Av_k>/<v_k,Bv_k>)<v_l,Bv_l>Bv_k must use independent,
  unbiase estimates for each element of this product, otherwise the estimates
  are correlated and will bias the computation). Furthermore, any terms in the
  denominator must be deterministic (i.e., <v_k,Bv_k> must be computed without
  using sample estimates which can be accomplished by introducing an auxiliary
  learning variable).

  Args:
    local_b_inner_product: Array of shape [l,k] denoting <v_l, Bv_k>
    local_a_inner_product: Array of shape [l,k] denoting <v_l, Av_k>
    local_b_inner_product_diag: Array of shape [l] denoting <v_l, Bv_l>
    b_inner_product_diag: Array of shape [k] denoting <v_k, Bv_k>. Insert an
      auxiliary variable here for in order to debias the receprocal.
    local_b_vector_product: ArrayTree with local eigen vectors products with
      B with leaves of shape [l, ...] denoting Bv_l.
    b_vector_product: ArrayTree with all eigen vectors products with
      B with leaves of shape [k, ...] denoting Bv_k.
    mask: Slice of a k x k matrix which is 1's under the diagonals and 0's
      everywhere else. This is used to denote the parents of each vector. Array
      of shape [l, k].
    b_diag_min: Minimum value for the b_inner_product_diag. This value is
      divided, so we use this to ensure we don't get a division by zero.

  Returns:
    The penalty gradient for the eigenvectors living on the current machine.
  """
  # Calculate <v_l,Av_k>/<v_k,Bv_k> with mask
  scale = jnp.einsum(
      'lk, lk, k -> lk',
      mask,
      local_a_inner_product,
      1 / jnp.maximum(b_inner_product_diag, b_diag_min),
  )
  # Calculate scale * <v_l,Bv_l>Bv_k  term
  global_term = eg_utils.tree_einsum_broadcast(
      'k..., lk, l -> l...',
      b_vector_product,
      scale,
      local_b_inner_product_diag,
  )
  # Calculate scale *  <v_l,Bv_k>Bv_l term
  local_term = eg_utils.tree_einsum_broadcast(
      'l..., lk, lk -> l...',
      local_b_vector_product,
      scale,
      local_b_inner_product,
  )
  return jax.tree_map(lambda x, y: x - y, global_term, local_term)


def generalized_eigengame_gradients(
    *,
    local_eigenvectors: chex.ArrayTree,
    all_eigenvectors: chex.ArrayTree,
    a_vector_product: chex.ArrayTree,
    b_vector_product: chex.ArrayTree,
    auxiliary_variables: eg_utils.AuxiliaryParams,
    mask: chex.Array,
    sliced_identity: chex.Array,
    epsilon: Optional[float] = None,
    maximize: bool = True,
) -> Tuple[chex.ArrayTree, eg_utils.AuxiliaryParams,]:
  """Solves for Av = lambda Bv using eigengame in a data parallel manner.

  Algorithm pseudocode can be found in Algorithm 1 of overleaf doc:
  https://ol.deepmind.host/read/kxdfdtfbsdxc

  For moderately sized models this is fine, but for really large models (
  moderate number of eigenvectors of >1m params) the memory overhead might be
  strained in the parallel case.

  Args:
    local_eigenvectors: ArrayTree with local eigen vectors with leaves
      of shape [l, ...] denoting v_l.
    all_eigenvectors: ArrayTree with all eigen vectors with leaves
      of shape [k, ...] denoting v_k.
    a_vector_product: ArrayTree with all eigen vectors products with
      A with leaves of shape [k, ...] denoting Av_k.
    b_vector_product: ArrayTree with all eigen vectors products with
      B with leaves of shape [k, ...] denoting Bv_k.
    auxiliary_variables: AuxiliaryParams object which holds all the variables
      which we want to update separately in order to avoid bias.
    mask: Mask with 1s below the diagonal and then sharded across devices, used
      to calculate the penalty. Shape of: [eigenvectors_per_device,
        total_eigenvector_count]
    sliced_identity: Sharded copy of the identity matrix used to calculate the
      reward. Shape of: [eigenvectors_per_device, total_eigenvector_count]
    epsilon: Add an isotropic term to the B matrix to make it
      positive semi-definite. In this case, we're solving for: Av =
        lambda*(B+epsilon*I)v.
    maximize: Whether to search for top-k eigenvectors of Av = lambda*Bv (True)
      or the top-k of (-A)v = lambda*Bv (False).

  Returns:
    returns the gradients for the eigenvectors and a new entry to update
    auxiliary variable estimates.
  """
  if not maximize:
    a_vector_product = -a_vector_product  # pytype: disable=unsupported-operands  # numpy-scalars

  # Add a small epsilon to the b vector product to make it positive definite.
  # This can help with convergence.
  if epsilon is not None:
    b_vector_product = jax.tree_map(
        lambda x, y: x + epsilon * y,
        b_vector_product,
        all_eigenvectors,
    )
  # Calculate the various matrix inner products.
  (
      # Get <v_l ,Bv_k> and <v_l, Av_k>.
      local_aux_b_inner_product,
      local_a_inner_product,
      # Get <v_k ,Bv_k> and <v_k, Av_k>.
      b_inner_product_diag,
      a_inner_product_diag,
  ) = _generalized_eg_matrix_inner_products(
      local_eigenvectors,
      all_eigenvectors,
      b_vector_product,
      a_vector_product,
      auxiliary_variables.b_vector_product,
  )

  # Get the local slices of these quantities. Going from k rows to l rows.
  (
      local_b_vector_product,
      local_a_vector_product,
      local_b_inner_product_diag,
      local_a_inner_product_diag,
  ) = eg_utils.get_local_slice(
      sliced_identity,
      (
          b_vector_product,
          a_vector_product,
          b_inner_product_diag,
          a_inner_product_diag,
      ),
  )
  # Calculate the reward
  reward = _generalized_eg_gradient_reward(
      local_b_inner_product_diag,
      local_a_inner_product_diag,
      local_b_vector_product,
      local_a_vector_product,
  )

  # Calculate the penalty using the associated auxiliary variables.
  penalty = _generalized_eg_gradient_penalty(
      local_aux_b_inner_product,
      local_a_inner_product,
      local_b_inner_product_diag,
      auxiliary_variables.b_inner_product_diag,
      local_b_vector_product,
      auxiliary_variables.b_vector_product,
      mask,
  )

  gradient = jax.tree_map(lambda x, y: x - y, reward, penalty)

  # Propagate the existing auxiliary variables
  new_auxiliary_variable = eg_utils.AuxiliaryParams(
      b_vector_product=b_vector_product,
      b_inner_product_diag=b_inner_product_diag,
  )
  return gradient, new_auxiliary_variable


import functools
@functools.partial(jax.pmap, axis_name='devices', in_axes=0, out_axes=0)
def pca_generalized_eigengame_gradients(
    local_eigenvectors: chex.ArrayTree,
    sharded_data: chex.ArrayTree,
    auxiliary_variables: eg_utils.AuxiliaryParams,
    mask: chex.Array,
    sliced_identity: chex.Array,
    mean_estimate: Optional[chex.ArrayTree] = None,
    epsilon: Optional[float] = 1e-4,
    maximize: bool = True,
    )->Tuple[chex.ArrayTree, eg_utils.AuxiliaryParams,]:
  """Evaluates PCA gradients for two data sources with local data parallelism.

  Implements PCA. In this case, we simply set the B matrix as I, which means
  the problem is solving for Av=lambda v and we're back to the classic eigen
  value problem.

  This is not as lightweight as eigengame unloaded due to additional terms in
  the calculation and handling of the auxiliary variables, and is mostly here to
  demonstrate the flexibility of the generalized eigenvalue method. However,
  adaptive optimisers may have an easier time with this since the gradients
  for the generalized eigengame are naturally tangential to the unit sphere.

  einsum legend:
    l: local eigenvector index -- the number of eigenvectors per machine.
    k: global eigenvector index -- number of eigenvectors over all machines.

  Args:
    local_eigenvectors: SplitVector with local eigen vectors products. Array
      tree with leaves of shape [l, ...] denoting v_l.
    sharded_data: SplitVector containing a batch of data from our two data
      sources. Array tree with leaves of shape [b, ...].
    auxiliary_variables: AuxiliaryParams object which holds all the variables
      which we want to update separately in order to avoid bias.
    mask: Mask with 1s below the diagonal and then sharded across devices, used
      to calculate the penalty. Shape of: [eigenvectors_per_device,
        total_eigenvector_count]
    sliced_identity: Sharded copy of the identity matrix used to calculate the
      reward. Shape of: [eigenvectors_per_device, total_eigenvector_count]
    mean_estimate: SplitVector containing an estimate of the mean of the input
      data if it is not centered by default. This is used to calculate the
      covariances. Array tree with leaves of shape [...].
    epsilon: Add an isotropic term to the A matrix to make it
      positive semi-definite. In this case, we're solving for: (A+epsilon*I)v =
        lambda*v.
    maximize: unused- Solving Av = lambda * v is the only sensible approach for
      PCA. We do not foresee a use case for (-Av) = lambda * v. Please contact
      authors if you have a need for it.

  Returns:
    returns the gradients for the eigenvectors and a new entry to update
    auxiliary variance estimates.
  """
  del maximize

  # First, collect all the eigenvectors v_k
  all_eigenvectors = jax.lax.all_gather(
      local_eigenvectors,
      axis_name='devices',
      axis=0,
      tiled=True,
  )


  # Calculate X^TXv/b for all v. This and v are all you need to construct the
  # gradient. This is done on each machine with a different batch.
  data_vector_product = eg_utils.tree_einsum(
      'b..., k... -> bk',
      sharded_data,
      all_eigenvectors,
      reduce_f=lambda x, y: x + y,
  )
  # divide by the batch size
  data_vector_product /= data_vector_product.shape[0]  # pytype: disable=attribute-error  # numpy-scalars
  sharded_gram_vector = eg_utils.tree_einsum_broadcast(
      'b..., bk -> k...',
      sharded_data,
      data_vector_product,
  )
  gram_vector = jax.lax.pmean(
      sharded_gram_vector,
      axis_name='devices',
  )

  # Add a small epsilon to the gram vector (Av) to make it positive definite.
  if epsilon is not None:
    gram_vector = jax.tree_map(
        lambda x, y: x + epsilon * y,
        gram_vector,
        all_eigenvectors,
    )
  return generalized_eigengame_gradients(
      local_eigenvectors=local_eigenvectors,
      all_eigenvectors=all_eigenvectors,
      a_vector_product=gram_vector,
      b_vector_product=all_eigenvectors,  # Just return V here.
      auxiliary_variables=auxiliary_variables,
      mask=mask,
      sliced_identity=sliced_identity,
      epsilon=None,
  )

