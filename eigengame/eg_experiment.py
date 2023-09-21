
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

import abc
import functools
from typing import Callable, Dict, Iterator, Optional, Tuple
from torch.utils.data import DataLoader
import chex
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

from typing import Tuple, Optional

import chex
import jax
import jax.numpy as jnp
import numpy as np





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

  def get_slice(all_vectors):
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
  einsum_function = lambda leaf: jnp.einsum(subscripts, leaf, *array_operands)
  mapped_tree = jax.tree_map(einsum_function, tree)
  if reduce_f is None:
    return mapped_tree
  else:
    return jax.tree_util.tree_reduce(reduce_f, mapped_tree)





def pca_unloaded_gradients(
    local_eigenvectors: chex.ArrayTree,
    sharded_data: chex.ArrayTree,
    mask: chex.Array,
    sliced_identity: chex.Array,
) -> chex.ArrayTree:
  # Collect the matrices from all the other machines.
  all_eigenvectors = jax.lax.all_gather(
      local_eigenvectors,
      axis_name='devices',
      axis=0,
      tiled=True,
  )
  # Calculate X^TXv/b for all v. This and v are all you need to construct the
  # gradient. This is done on each machine with a different batch.
  data_vector_product = tree_einsum(
      'b..., k... -> bk',
      sharded_data,
      all_eigenvectors,
      reduce_f=lambda x, y: x + y,
  )
  # divide by the batch size
  data_vector_product /= data_vector_product.shape[0]  # pytype: disable=attribute-error  # numpy-scalars
  sharded_gram_vector = tree_einsum_broadcast(
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
  scale = tree_einsum(
      'l..., k... -> lk',
      local_eigenvectors,
      gram_vector,
      reduce_f=lambda x, y: x + y,
  )

  # The weights is 1 below diagonals
  penalty = tree_einsum_broadcast(
      'k..., lk, lk -> l...',
      all_eigenvectors,
      scale,
      mask,
  )

  reward = get_local_slice(sliced_identity, gram_vector)  # XtXv_i
  return jax.tree_map(lambda x, y: x - y, reward, penalty)




def _generalized_eg_matrix_inner_products(
    local_eigenvectors: chex.ArrayTree,
    all_eigenvectors: chex.ArrayTree,
    b_vector_product: chex.ArrayTree,
    a_vector_product: chex.ArrayTree,
    aux_b_vector_product: chex.ArrayTree,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array,]:

  # Evaluate the various forms of the inner products used in the gradient
  local_aux_b_inner_product = tree_einsum(
      'l... , k... -> lk',
      local_eigenvectors,
      aux_b_vector_product,
      reduce_f=lambda x, y: x + y,
  )  # v_l B v_k, used in the penalty term
  local_a_inner_product = tree_einsum(
      'l..., k... -> lk',
      local_eigenvectors,
      a_vector_product,
      reduce_f=lambda x, y: x + y,
  )  # v_l A v_k, used in the penalty term
  b_inner_product_diag = tree_einsum(
      'k..., k... -> k',
      all_eigenvectors,
      b_vector_product,
      reduce_f=lambda x, y: x + y,
  )  # v_k B v_k, used in the penalty and reward terms
  a_inner_product_diag = tree_einsum(
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
  # Evaluates <v_l,Bv_l>Av_l
  scaled_a_vectors = tree_einsum_broadcast(
      'l..., l -> l...',
      local_a_vector_product,
      local_b_inner_product_diag,
  )
  # Evaluates <v_l,Av_l>Bv_l
  scaled_b_vectors = tree_einsum_broadcast(
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
  # Calculate <v_l,Av_k>/<v_k,Bv_k> with mask
  scale = jnp.einsum(
      'lk, lk, k -> lk',
      mask,
      local_a_inner_product,
      1 / jnp.maximum(b_inner_product_diag, b_diag_min),
  )
  # Calculate scale * <v_l,Bv_l>Bv_k  term
  global_term = tree_einsum_broadcast(
      'k..., lk, l -> l...',
      b_vector_product,
      scale,
      local_b_inner_product_diag,
  )
  # Calculate scale *  <v_l,Bv_k>Bv_l term
  local_term = tree_einsum_broadcast(
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
    auxiliary_variables: AuxiliaryParams,
    mask: chex.Array,
    sliced_identity: chex.Array,
    epsilon: Optional[float] = None,
    maximize: bool = True,
) -> Tuple[chex.ArrayTree, AuxiliaryParams,]:
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
  ) = get_local_slice(
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
  new_auxiliary_variable = AuxiliaryParams(
      b_vector_product=b_vector_product,
      b_inner_product_diag=b_inner_product_diag,
  )
  return gradient, new_auxiliary_variable


import functools


def pca_generalized_eigengame_gradients(
    local_eigenvectors: chex.ArrayTree,
    sharded_data: chex.ArrayTree,
    auxiliary_variables: AuxiliaryParams,
    mask: chex.Array,
    sliced_identity: chex.Array,
    mean_estimate: Optional[chex.ArrayTree] = None,
    epsilon: Optional[float] = 1e-4,
    maximize: bool = True,
    )->Tuple[chex.ArrayTree, AuxiliaryParams,]:
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
  data_vector_product = tree_einsum(
      'b..., k... -> bk',
      sharded_data,
      all_eigenvectors,
      reduce_f=lambda x, y: x + y,
  )
  # divide by the batch size
  data_vector_product /= data_vector_product.shape[0]  # pytype: disable=attribute-error  # numpy-scalars
  sharded_gram_vector = tree_einsum_broadcast(
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

    updates, new_opt_state = self.tx.update(grads, self.opt_state)
    aux_updates, aux_new_opt_state = self.aux_tx.update(aux_grads, self.aux_opt_state)
    
    new_params = optax.apply_updates(self.params, updates)
    new_params = normalize_eigenvectors(new_params)

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
) -> AuxiliaryParams:
  """Initializes the auxiliary variables from the eigenvalues."""
  leaves, _ = jax.tree_util.tree_flatten(init_vectors)
  per_device_eigenvectors = leaves[0].shape[0]
  total_eigenvectors = per_device_eigenvectors * device_count
  return AuxiliaryParams(
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
  normalized_eigenvector = normalize_eigenvectors(eigenvector_tree)

  return normalized_eigenvector

def build_lr_schedule(learning_rate,steps, warmup_ratio):
  end_lr = 0.1*(learning_rate)
  warm_up_step = int(steps*warmup_ratio)
  def decaying_schedule_with_warmup(
        step: int,
    ) -> float:
      warmup_lr = step * learning_rate / warm_up_step
      decay_shift = (warm_up_step * learning_rate - steps * end_lr) / (
          end_lr - learning_rate)
      decay_scale = learning_rate * (warm_up_step + decay_shift)
      decay_lr = decay_scale / (step + decay_shift)
      return jnp.where(step < warm_up_step, warmup_lr, decay_lr)
  return decaying_schedule_with_warmup

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


  def __init__(self,init_rng, n_components = 2, eval_freq=None):
    self.eigenvector_count = n_components
    self.init_rng  = init_rng
    self.mask = create_sharded_mask(n_components)
    self.sliced_identity = create_sharded_identity(n_components)
    self.groundtruth = None
    self.eval_freq = eval_freq
  def create_update_eigenvectors(self):
    @functools.partial(jax.pmap, axis_name='devices', in_axes=0, out_axes=0,donate_argnums=(0,))
    def update_eigenvectors(
        tstate,
        batch: chex.Array,
        mask,
        sliced_identity
        
    ) -> Tuple[chex.ArrayTree, chex.ArrayTree, AuxiliaryParams,
              AuxiliaryParams, Optional[chex.ArrayTree],
              Optional[chex.ArrayTree],]:
      """Calculates the new vectors, applies update and then renormalize."""
      gradient, new_aux = pca_generalized_eigengame_gradients(
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
    
  def fit(self, data,learning_rate=2e-4,steps=1000000,warmup_ratio=0.9,batch_size=512,aux_lr=1e-3):
    update_eigenvectors = self.create_update_eigenvectors()


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
                  learning_rate=build_lr_schedule(learning_rate,steps, warmup_ratio),
                  b1=0.9,
                  b2=0.999,
                  eps=1e-8,
                  )
    aux_optimizer = optax.sgd(learning_rate=aux_lr)
    
    tstate = TrainState.create(
          params=eigenvectors,
          aux_params=auxiliary_variables,
          tx=optimizer,
          aux_tx=aux_optimizer
          )
    
    def expand_zero_shape(x):
      if isinstance(x, float) or isinstance(x, int) or x.shape == ():
        return jnp.broadcast_to(x,jax.local_device_count())
      return x
    tstate = jax.tree_map(expand_zero_shape, tstate)
    tstate = jax.device_put_sharded(tree_unstack(tstate), jax.local_devices())
    step = 0
    for batch in data_stream:
      if batch.shape[0]!=batch_size: #tmp hack..
        continue
      batch = format_batch(batch)
      
      tstate = update_eigenvectors(tstate, batch, mask=self.mask, sliced_identity=self.sliced_identity)
      if self.eval_freq is not None and (step%self.eval_freq) ==0:
        print(f"{step} {self.nondiagonal_mean(tstate)=}")
      if step==steps:
        break
      step +=1
    return jax.device_get(tstate.params)['params']
  
  def groundtruth_error(self, tstate, groundtruth):
    eigenvectors = jax.device_get(tstate.params)['params']
    eigenvectors = np.squeeze(eigenvectors).reshape((self.eigenvector_count,-1))
    
    return np.square(eigenvectors-groundtruth).sum()
  
  def nondiagonal_mean(self, tstate):
    indices = jnp.arange(self.eigenvector_count)
    eigenvectors = jax.device_get(tstate.params)['params']
    eigenvectors = np.squeeze(eigenvectors).reshape((self.eigenvector_count,-1))
    result = eigenvectors@eigenvectors.T
    result[indices,indices] = 0
    return np.abs(result).sum()/(self.eigenvector_count*(self.eigenvector_count-1))