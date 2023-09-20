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



class TrainState(struct.PyTreeNode):
  step: int
  params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState = struct.field(pytree_node=True)

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
        tx=tx,
        aux_tx=aux_tx,
        opt_state=opt_state,
        aux_opt_state=aux_opt_state,
        **kwargs,
    )



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

class EigenGame:
  """Jaxline object for running Eigengame Experiments."""


  def __init__(self, init_rng, n_components = 2):
    self.eigenvector_count = n_components
    self.epsilon = 1e-4
    self.init_rng  = init_rng



  def initialize_eigenvector_params(
      self,
      batch,
  ) -> chex.ArrayTree:
    """Initializes the eigenvalues, mean estimates and auxiliary variables."""
    local_activiation_batch = get_first(batch)
    initial_eigenvectors = eg_utils.initialize_eigenvectors(
        self.eigenvector_count,
        local_activiation_batch,
        self.init_rng,
    )
    
    auxiliary_variables = eg_utils.init_aux_variables(
        jax.device_count(),
        initial_eigenvectors,
    )
    return initial_eigenvectors, auxiliary_variables

  def update_eigenvectors(
      self,
      tstate,
      batch: chex.Array,
      mask: chex.Array,
      sliced_identity: chex.Array,

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
        epsilon=self.epsilon,
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

    # Initialize the eigenvalues and mean estimate from a batch of data
    eigenvectors, auxiliary_variables = self.initialize_eigenvector_params(self.init_rng)

    update = functools.partial(
        jax.pmap(
          self.update_eigenvectors,
            axis_name='devices',
            in_axes=0,
            out_axes=0,
        ),
        mask=eg_gradients.create_sharded_mask(self.eigenvector_count),
        sliced_identity=eg_gradients.create_sharded_identity(self.eigenvector_count))

    # Initialize the data generators and optimizers
    optimizer = optax.adam(
                  learning_rate=decaying_schedule_with_warmup,
                  b1=0.9,
                  b2=0.999,
                  eps=1e-8,
                  )
    aux_optimizer = optax.sgd(learning_rate=1e-3)
    
    
    tstate = jax.pmap(TrainState.create)(eigenvectors, auxiliary_variables, optimizer, aux_optimizer)
    
    for batch in data_stream:
      tstate = update(tstate, batch)
    return jax.device_get(tstate.params)
        