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

"""Boilerplate needed to do a jaxline experiment with Eigengame."""
import abc
import functools
from typing import Callable, Dict, Iterator, Optional, Tuple

import chex
from eigengame import eg_gradients
from eigengame import eg_utils
import jax
import jax.numpy as jnp
import numpy as np
from typing import Any
import chex
import optax
from flax import core
from flax import struct
from flax.jax_utils import replicate,unreplicate
from einops import rearrange
def add_device_dim(x, num_processes=None):
    if num_processes is None:
        num_processes = jax.local_device_count()
    """ Add a process dimension to a tensor. """
    if x is None or not hasattr(x, 'shape'):
        return x
    return rearrange(x, '(p l)  ... -> p l ... ', p=num_processes)

class TrainState(struct.PyTreeNode):
  step: int
  params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  aux_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  aux_tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState = struct.field(pytree_node=True)
  aux_opt_state: optax.OptState = struct.field(pytree_node=True)

  def apply_gradients(self, *, grads, aux_grads, **kwargs):
    
    updates, new_opt_state = self.tx.update(jax.tree_map(lambda x: -x, grads), self.opt_state)
    aux_updates, aux_new_opt_state = self.aux_tx.update(aux_grads, self.aux_opt_state)
    
    new_params = optax.apply_updates(self.params, updates)
    aux_new_params = optax.apply_updates(self.params, aux_updates)
    return self.replace(
        step=self.step + 1,
        params=new_params,
        aux_params=aux_new_params,
        opt_state=new_opt_state,
        aux_opt_state=aux_new_opt_state,
        
        **kwargs,
    )

  @classmethod
  def create(cls, *, params, aux_params, tx, aux_tx, **kwargs):
    """Creates a new instance with `step=0` and initialized `opt_state`."""
    opt_state = tx.init(params)
    aux_opt_state = tx.init(aux_params)
    return cls(
        step=0,
        params=params,
        aux_params=aux_params,
        tx=tx,
        aux_tx=aux_tx,
        opt_state=opt_state,
        aux_opt_state=aux_opt_state,
        **kwargs,
    )



@functools.partial(
    jax.pmap,
    in_axes=0,
    out_axes=0,
    axis_name='devices',
    static_broadcasted_argnums=0,
)
def init_aux_variables(
    device_count: int,
    init_vectors: chex.ArrayTree,
) -> eg_utils.AuxiliaryParams:
  """Initializes the auxiliary variables from the eigenvalues."""
  leaves, _ = jax.tree_util.tree_flatten(init_vectors)
  per_device_eigenvectors = leaves[0].shape[0]
  total_eigenvectors = per_device_eigenvectors * device_count
  return eg_utils.AuxiliaryParams(
      b_vector_product=jax.tree_map(
          lambda leaf: jnp.zeros((total_eigenvectors, *leaf.shape[1:])),
          init_vectors,
      ),
      b_inner_product_diag=jnp.zeros(total_eigenvectors),
  )
@functools.partial(jax.pmap, axis_name='devices', in_axes=0, out_axes=0, static_broadcasted_argnums=0)
def initialize_eigenvectors(
    eigenvector_count: int,
    batch: chex.ArrayTree,
    rng_key: chex.PRNGKey,
) -> chex.ArrayTree:
  """Initialize the eigenvectors on a unit sphere and shards it.

  Args:
    eigenvector_count: Total number of eigenvectors (i.e. k)
    batch: A batch of the data we're trying to find the eigenvalues of. The
      initialized vectors will take the shape and tree structure of this data.
      Array tree with leaves of shape [b, ...].
    rng_key: jax rng seed. For multihost, each host should have a different
      seed in order to initialize correctly

  Returns:
    A pytree of initialized, normalized vectors in the same structure as the
    input batch. Array tree with leaves of shape [num_devices, l, ...].
  """
  device_count = jax.device_count()
  if eigenvector_count % device_count != 0:
    raise ValueError(f'Number of devices ({device_count}) must divide number of'
                     'eigenvectors ({eigenvector_count}).')
  per_device_count = eigenvector_count // device_count
  leaves, treedef = jax.tree_flatten(batch)
  shapes = [(per_device_count, *leaf.shape[1:]) for leaf in leaves]
  
  per_leaf_keys = jax.random.split(rng_key, len(leaves))

  vector_leaves = [
      jax.random.normal(key, shape)
      for key, shape in zip(per_leaf_keys, shapes)
  ]
  eigenvector_tree = jax.tree_unflatten(treedef, vector_leaves)
  normalized_eigenvector = eg_utils.normalize_eigenvectors(eigenvector_tree)

  return normalized_eigenvector
  # return jax.device_put_sharded(eigenvectors, jax.local_devices())

def decaying_schedule_with_warmup(
    step: int,
    warm_up_step=10_000,
    end_step=1_000_000,
    base_lr=2e-4,
    end_lr=1e-6
) -> float:

  warmup_lr = step * base_lr / warm_up_step
  decay_shift = (warm_up_step * base_lr - end_step * end_lr) / (
      end_lr - base_lr)
  decay_scale = base_lr * (warm_up_step + decay_shift)
  decay_lr = decay_scale / (step + decay_shift)
  return jnp.where(step < warm_up_step, warmup_lr, decay_lr)



def get_first(xs):
  """Gets values from the first device."""
  return jax.tree_util.tree_map(lambda x: x[0], xs)

# @functools.partial(jax.pmap, axis_name='devices', static_broadcasted_argnums=0)
def create_sharded_mask(eigenvector_count) -> chex.ArraySharded:
  """Defines a mask of 1s under the diagonal and shards it."""
  mask = np.ones((eigenvector_count, eigenvector_count))
  r, c = np.triu_indices(eigenvector_count)
  mask[..., r, c] = 0
  start_index = jax.process_index() * eigenvector_count // jax.process_count()
  end_index = (jax.process_index()+1) * eigenvector_count // jax.process_count()
  mask = mask[start_index:end_index]
  mask = mask.reshape((
      jax.local_device_count(),
      eigenvector_count // jax.device_count(),
      eigenvector_count,
  ))
  return mask

# @functools.partial(jax.pmap, axis_name='devices', static_broadcasted_argnums=0)
def create_sharded_identity(eigenvector_count) -> chex.ArraySharded:
  """Create identity matrix which is then split across devices."""
  identity = np.eye(eigenvector_count)

  start_index = jax.process_index() * eigenvector_count // jax.process_count()
  end_index = (jax.process_index() +
              1) * eigenvector_count // jax.process_count()
  identity = identity[start_index:end_index]
  identity = identity.reshape((
      jax.local_device_count(),
      eigenvector_count // jax.device_count(),
      eigenvector_count,
  ))
  return identity

class EigenGame:
  """Jaxline object for running Eigengame Experiments."""


  def __init__(self,init_rng, n_components = 2):
    self.eigenvector_count = n_components
    self.init_rng  = init_rng
    self.mask = create_sharded_mask(n_components)
    self.sliced_identity = create_sharded_identity(n_components)



  
  # @functools.partial(jax.pmap, axis_name='devices')
  # @functools.partial(jax.pmap, axis_name='devices', in_axes=0, out_axes=0)
  def update_eigenvectors(
      self,
      tstate,
      batch: chex.Array
  ) -> Tuple[chex.ArrayTree, chex.ArrayTree, eg_utils.AuxiliaryParams,
             eg_utils.AuxiliaryParams, Optional[chex.ArrayTree],
             Optional[chex.ArrayTree],]:
    # mask = create_sharded_mask(self.eigenvector_count)
    # print(mask.shape)
    # [jax.lax.axis_index("devices")]
    # sliced_identity = create_sharded_identity(self.eigenvector_count)
    # print(sliced_identity.shape)
    
    """Calculates the new vectors, applies update and then renormalize."""
    # pca_generalized_eigengame_gradients = functools.partial(jax.pmap, axis_name='devices', in_axes=0, out_axes=0)
    el = dict(local_eigenvectors=tstate.params,
        sharded_data=batch,
        auxiliary_variables=tstate.aux_params,
        mask=self.mask,
        sliced_identity=self.sliced_identity,)
    print(jax.tree_map(lambda x: x.shape, el))
    gradient, new_aux = eg_gradients.pca_generalized_eigengame_gradients(
        local_eigenvectors=tstate.params,
        sharded_data=batch,
        auxiliary_variables=tstate.aux_params,
        mask=self.mask,
        sliced_identity=self.sliced_identity,
        # epsilon=epsilon,
    )
    # Update and normalize the eigenvectors variables.
    neg_grads = jax.tree_map(lambda x: -x, gradient)
    auxiliary_error = jax.tree_map(
        lambda x, y: x - y,
        tstate.aux_params,
        new_aux,
    )
    tstate = tstate.apply_gradients(grads=neg_grads, aux_grads=auxiliary_error)
    tstate = tstate.replace(params=eg_utils.normalize_eigenvectors(tstate.params))
    return tstate
    
  def fit(self, data_stream):
    batch = next(data_stream)
    # print(jax.tree_map(lambda x: x.shape, batch))
    batch = jax.tree_map(lambda x: add_device_dim(x), batch)
    # batch = add_device_dim(batch, jax.device_count())
    # print(jax.tree_map(lambda x: x.shape, batch))
    
    eigenvectors = initialize_eigenvectors(
        self.eigenvector_count,
        batch,
        self.init_rng,
    )
    auxiliary_variables = init_aux_variables(
        jax.device_count(),
        eigenvectors,
    )


    # Initialize the data generators and optimizers
    optimizer = optax.adam(
                  learning_rate=decaying_schedule_with_warmup,
                  b1=0.9,
                  b2=0.999,
                  eps=1e-8,
                  )
    aux_optimizer = optax.sgd(learning_rate=1e-3)
    
    tstate = TrainState.create(
          params=eigenvectors,
          aux_params=auxiliary_variables,
          tx=optimizer,
          aux_tx=aux_optimizer
          )
    
    # tstate = tstate.replace(step=replicate(tstate.step))
    def expand_zero_shape(x):
      if isinstance(x, float) or isinstance(x, int) or x.shape == ():
        return jnp.broadcast_to(x,jax.local_device_count())
      return x
    tstate = jax.device_put(tstate)
    tstate = jax.tree_map(expand_zero_shape, tstate)
    print(jax.tree_map(lambda x: x.shape, tstate))
    for batch in data_stream:
      # print(jax.tree_map(lambda x: x.shape, batch))
      # batch = add_device_dim(batch)
      batch = jax.tree_map(lambda x: add_device_dim(x), batch)
      # print(jax.tree_map(lambda x: x.shape, batch))
      # return tstate,batch
      # print(batch.shape)
      # print(jax.tree_map(lambda x: x.ndim, batch))
      # print(batch.shape)
      # batch = replicate(batch)
      
      tstate = self.update_eigenvectors(tstate, batch)
    return jax.device_get(tstate.params)
        