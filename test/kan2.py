from test_utils import *
import functools
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from absl.testing import absltest
import enzyme_ad
from enzyme_ad.jax import JaXPipeline, hlo_opts
from blackjax.util import run_inference_algorithm
import blackjax
from typing import Tuple
from absl.testing import absltest
from test_utils import *

pipelines = [
    ("JaX", None, CurBackends),
    ("DefOpt", JaXPipeline(hlo_opts()), CurBackends),
    (
        "EqSat",
        JaXPipeline(
            "inline{default-pipeline=canonicalize max-iterations=4},"
            + "equality-saturation-pass"
        ),
        CurBackends,
    ),
]

with open("test/exported_modules/kan2.mlir") as f:
    code = f.read()

@jax.jit
def foo(arg1, arg2, arg3, arg4, arg5):
    return enzyme_ad.jax.primitives.hlo_call(
        arg1, arg2, arg3, arg4, arg5,
        source=code
    )[0]

args = [
    jnp.ones((100,1),jnp.float32),
    jnp.ones((16800),jnp.float32),
    jnp.ones((10),jnp.float32),
    jnp.ones((10),jnp.float32),
    jnp.ones((10),jnp.float32)
]

output = foo(*args)

class KAN2(EnzymeJaxTest):
    def setUp(self):
        import jax.numpy as jnp
        import jax.random

        self.fn = foo
        self.name = "KAN2"
        self.count = 10
        self.revprimal = False
        self.AllPipelines = pipelines
        self.AllBackends = CurBackends

        self.ins = args
        self.dins = args
        self.douts = [
            jnp.ones((100,1), jnp.float32),
        ]
        self.tol = 5e-5


if __name__ == "__main__":
    from test_utils import fix_paths

    fix_paths()
    absltest.main()