
import os
if "DEBUG" in os.environ:
  import debugpy
  debugpy.listen(5678)
  print("Waiting for debugger attach")
  debugpy.wait_for_client()
  
import sys
# sys.path.append("eigengame")
import numpy as np
import jax
from torch.utils.data import DataLoader
from sklearn.datasets import load_digits
from eigengame.eg_experiment import EigenGame, add_device_dim
import jax.numpy as jnp



data, labels = load_digits(return_X_y=True)


prng = jax.random.PRNGKey(0)
# prng, init_rng = jax.random.split(prng)

prng = jax.random.split(prng, 8)
eigenvectors = EigenGame(n_components=32,init_rng=prng).fit(data)
eigenvectors = np.squeeze(eigenvectors)
# eigenvectors = EigenGame(n_components=128,init_rng=init_rng).fit(dataloader)
from sklearn.decomposition import PCA
groundtruth = PCA(n_components=8).fit(data)
groundtruth.components_@groundtruth.components_.T