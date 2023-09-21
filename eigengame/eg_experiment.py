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
from torch.utils.data import DataLoader
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
from torch.utils.data import DataLoader, Dataset, Sampler
import random
from sklearn.decomposition import PCA

from einops import rearrange
def add_device_dim(x, num_processes=None):
    if num_processes is None:
        num_processes = jax.local_device_count()
    """ Add a process dimension to a tensor. """
    if x is None or not hasattr(x, 'shape'):
        return x
    return rearrange(x, '(p l)  ... -> p l ... ', p=num_processes)
  

      
class customSampler(Sampler) :
    def __init__(self, size):
        assert size > 0
        self.size = size
        self.order = list(range(size))
    
    def __iter__(self):
        idx = 0
        while True:
            yield self.order[idx]
            idx += 1
            if idx == len(self.order):
                random.shuffle(self.order)
                idx = 0
                
def format_batch(batch):
  batch = np.array(batch)
  batch = {"params":batch}
  batch = jax.tree_map(lambda x: add_device_dim(x), batch)
  return batch

class TrainState(struct.PyTreeNode):
  step: int
  params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  aux_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  aux_tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState = struct.field(pytree_node=True)
  aux_opt_state: optax.OptState = struct.field(pytree_node=True)

  def apply_gradients(self, *, grads, aux_grads, **kwargs):
    grads = jax.tree_map(lambda x: -x, grads)
    aux_grads = jax.tree_map(
        lambda x, y: x - y,
        self.aux_params,
        aux_grads,
    )
    print(jax.tree_map(lambda x:x.shape,grads))
    print(jax.tree_map(lambda x:x.shape,self.opt_state))
    updates, new_opt_state = self.tx.update(grads, self.opt_state)
    aux_updates, aux_new_opt_state = self.aux_tx.update(aux_grads, self.aux_opt_state)
    
    new_params = optax.apply_updates(self.params, updates)
    new_params = eg_utils.normalize_eigenvectors(new_params)

    aux_new_params = optax.apply_updates(self.aux_params, aux_updates)
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




def tree_unstack(tree):
    """Takes a tree and turns it into a list of trees. Inverse of tree_stack.
    For example, given a tree ((a, b), c), where a, b, and c all have first
    dimension k, will make k trees
    [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]
    Useful for turning the output of a vmapped function into normal objects.
    """
    leaves, treedef = jax.tree_flatten(tree)
    n_trees = leaves[0].shape[0]
    new_leaves = [[] for _ in range(n_trees)]
    for leaf in leaves:
        for i in range(n_trees):
            new_leaves[i].append(leaf[i])
    new_trees = [treedef.unflatten(l) for l in new_leaves]
    return new_trees

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

# pca_generalized_eigengame_gradients = functools.partial(jax.pmap, axis_name='devices', in_axes=0, out_axes=0)
class EigenGame:
  """Jaxline object for running Eigengame Experiments."""


  def __init__(self,init_rng, n_components = 2):
    self.eigenvector_count = n_components
    self.init_rng  = init_rng
    self.mask = create_sharded_mask(n_components)
    self.sliced_identity = create_sharded_identity(n_components)
    self.groundtruth = None
  def create_update_eigenvectors(self):
    @functools.partial(jax.pmap, axis_name='devices', in_axes=0, out_axes=0,donate_argnums=(0,))
    def update_eigenvectors(
        tstate,
        batch: chex.Array,
        mask,
        sliced_identity
        
    ) -> Tuple[chex.ArrayTree, chex.ArrayTree, eg_utils.AuxiliaryParams,
              eg_utils.AuxiliaryParams, Optional[chex.ArrayTree],
              Optional[chex.ArrayTree],]:
      """Calculates the new vectors, applies update and then renormalize."""
      gradient, new_aux = eg_gradients.pca_generalized_eigengame_gradients(
          local_eigenvectors=tstate.params,
          sharded_data=batch,
          auxiliary_variables=tstate.aux_params,
          mask=mask,
          sliced_identity=sliced_identity,
          # epsilon=epsilon,
      )
      tstate = tstate.apply_gradients(grads=gradient, aux_grads=new_aux)
      return tstate
    return update_eigenvectors
    
  def fit(self, data,learning_rate=2e-4,steps=1000000,warmup_ratio=0.9,batch_size=512):
    update_eigenvectors = self.create_update_eigenvectors()
    groundtruth = PCA(n_components=self.eigenvector_count).fit(data).components_
    #normalize the groundtruth to have unit norm
    groundtruth = groundtruth/np.linalg.norm(groundtruth,axis=1,keepdims=True)
    def decaying_schedule_with_warmup(
        step: int,
    ) -> float:
      end_lr = 0.1*(learning_rate)
      warm_up_step = int(steps*warmup_ratio)
      warmup_lr = step * learning_rate / warm_up_step
      decay_shift = (warm_up_step * learning_rate - steps * end_lr) / (
          end_lr - learning_rate)
      decay_scale = learning_rate * (warm_up_step + decay_shift)
      decay_lr = decay_scale / (step + decay_shift)
      return jnp.where(step < warm_up_step, warmup_lr, decay_lr)

    data_stream = DataLoader(data, batch_size=batch_size,sampler=customSampler(data.shape[0]))
    batch = next(iter(data_stream))
    batch = format_batch(batch)

    
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
    # aux_optimizer = optax.sgd(learning_rate=1e-1)
    aux_optimizer = optax.sgd(learning_rate=1e-3)
    
    tstate = TrainState.create(
          params=eigenvectors,
          aux_params=auxiliary_variables,
          tx=optimizer,
          aux_tx=aux_optimizer
          )
      
    def calculate_dist(tstate):
      eigenvectors = jax.device_get(tstate.params)['params']
      eigenvectors = np.squeeze(eigenvectors).reshape((self.eigenvector_count,-1))
      
      return np.square(eigenvectors-groundtruth).sum()
    def calculate_dot(tstate):
      eigenvectors = jax.device_get(tstate.params)['params']
      eigenvectors = np.squeeze(eigenvectors).reshape((self.eigenvector_count,-1))
      return eigenvectors@eigenvectors.T
    
    def expand_zero_shape(x):
      if isinstance(x, float) or isinstance(x, int) or x.shape == ():
        return jnp.broadcast_to(x,jax.local_device_count())
      return x
    tstate = jax.tree_map(expand_zero_shape, tstate)
    tstate = jax.device_put_sharded(tree_unstack(tstate), jax.local_devices())
    print(jax.tree_map(lambda x:x.shape,tstate))
    
    # tstate = jax.device_put(tstate)
    step = 0
    for batch in data_stream:
      if batch.shape[0]!=batch_size:
        continue
      batch = format_batch(batch)
      
      tstate = update_eigenvectors(tstate, batch, mask=self.mask, sliced_identity=self.sliced_identity)
      # print(i)
      if (step%1000) ==0:
        print(f"{step} {calculate_dot(tstate)=}")
        
      if step==steps:
        break
      step +=1
    return jax.device_get(tstate.params)['params']
