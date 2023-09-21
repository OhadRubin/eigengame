
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
prng = jax.random.split(prng, 8)
#We whiten the data before passing it to the EigenGame using scipy
data = data - data.mean(axis=0)
# data = data / data.std(axis=0)
eigenvectors = EigenGame(n_components=8,init_rng=prng,eval_freq=1000).fit(data)
eigenvectors = np.squeeze(eigenvectors)