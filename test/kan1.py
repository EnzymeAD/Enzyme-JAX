from test_utils import *
import functools
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from absl.testing import absltest
import enzyme_ad
from enzyme_ad.jax import JaXPipeline, hlo_opts
from typing import Tuple
from absl.testing import absltest
from test_utils import *

with open("test/exported_modules/kan1.mlir") as f:
    code = f.read()

@jax.jit
def foo(arg1, arg2, arg3, arg4, arg5):
    return enzyme_ad.jax.primitives.hlo_call(
        arg1, arg2, arg3, arg4, arg5,
        source=code
    )[0]

class KAN1(EnzymeJaxTest):
    def setUp(self):
        import jax.numpy as jnp
        import jax.random

        key = jax.random.PRNGKey(0)

        key, subkey = jax.random.split(key)
        arg1 = jax.random.uniform(subkey, shape=(100,1), dtype=jnp.float32)

        key, subkey = jax.random.split(key)
        arg2 = jax.random.uniform(subkey, shape=(18480,), dtype=jnp.float32)

        key, subkey = jax.random.split(key)
        arg3 = jax.random.uniform(subkey, shape=(10,), dtype=jnp.float32)

        key, subkey = jax.random.split(key)
        arg4 = jax.random.uniform(subkey, shape=(10,), dtype=jnp.float32)

        key, subkey = jax.random.split(key)
        arg5 = jax.random.uniform(subkey, shape=(10,), dtype=jnp.float32)

        args = [arg1, arg2, arg3, arg4, arg5]

        self.fn = foo
        self.name = "KAN1"
        self.count = 1000
        self.revprimal = False
        self.AllPipelines = pipelines()
        self.AllBackends = CurBackends

        self.ins = args
        self.dins = args
        self.douts = [
            jnp.ones((100,1), jnp.float32),
            # jnp.ones((10), jnp.float32),
            # jnp.ones((10), jnp.float32),
            # jnp.ones((10), jnp.float32),
            # jnp.ones((100,1), jnp.float32),
            # jnp.ones((18480), jnp.float32)
        ]
        self.tol = 5e-3


if __name__ == "__main__":
    from test_utils import fix_paths

    fix_paths()
    absltest.main()
