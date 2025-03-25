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

with open("test/exported_modules/searchless_chess_9m.mlir") as f:
    code = f.read()

@jax.jit
def foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45, arg46, arg47, arg48, arg49, arg50, arg51, arg52, arg53, arg54, arg55, arg56, arg57, arg58, arg59, arg60, arg61, arg62, arg63, arg64, arg65, arg66, arg67, arg68, arg69, arg70, arg71, arg72, arg73, arg74, arg75, arg76, arg77, arg78, arg79, arg80, arg81, arg82, arg83, arg84, arg85, arg86, arg87, arg88, arg89, arg90, arg91, arg92, arg93, arg94):
    return enzyme_ad.jax.primitives.hlo_call(
        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45, arg46, arg47, arg48, arg49, arg50, arg51, arg52, arg53, arg54, arg55, arg56, arg57, arg58, arg59, arg60, arg61, arg62, arg63, arg64, arg65, arg66, arg67, arg68, arg69, arg70, arg71, arg72, arg73, arg74, arg75, arg76, arg77, arg78, arg79, arg80, arg81, arg82, arg83, arg84, arg85, arg86, arg87, arg88, arg89, arg90, arg91, arg92, arg93, arg94,
        source=code
    )[0]

args = [jnp.ones((1968, 256), jnp.float32),
jnp.ones((79, 256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256), jnp.float32),
jnp.ones((256, 1024), jnp.float32),
jnp.ones((256, 1024), jnp.float32),
jnp.ones((256, 1024), jnp.float32),
jnp.ones((1024, 256), jnp.float32),
jnp.ones((256, 1024), jnp.float32),
jnp.ones((256, 1024), jnp.float32),
jnp.ones((1024, 256), jnp.float32),
jnp.ones((256, 1024), jnp.float32),
jnp.ones((256, 1024), jnp.float32),
jnp.ones((1024, 256), jnp.float32),
jnp.ones((256, 1024), jnp.float32),
jnp.ones((256, 1024), jnp.float32),
jnp.ones((1024, 256), jnp.float32),
jnp.ones((1024, 256), jnp.float32),
jnp.ones((256, 1024), jnp.float32),
jnp.ones((256, 1024), jnp.float32),
jnp.ones((1024, 256), jnp.float32),
jnp.ones((128), jnp.float32),
jnp.ones((256, 128), jnp.float32),
jnp.ones((256, 1024), jnp.float32),
jnp.ones((256, 1024), jnp.float32),
jnp.ones((1024, 256), jnp.float32),
jnp.ones((256, 1024), jnp.float32),
jnp.ones((256, 1024), jnp.float32),
jnp.ones((1024, 256), jnp.float32),
jnp.ones((256, 1024), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((256, 256), jnp.float32),
jnp.ones((33, 79), jnp.int32)]

output = foo(*args)

class SearchlessChess9M(EnzymeJaxTest):
    def setUp(self):
        import jax.numpy as jnp
        import jax.random

        self.fn = foo
        self.name = "searchless_chess_9m"
        self.count = 100
        self.revprimal = False
        self.AllPipelines = pipelines()
        self.AllBackends = CurBackends

        self.ins = args
        self.dins = args
        self.douts = jnp.ones((33, 79, 128), jnp.float32)
        self.tol = 5e-5


if __name__ == "__main__":
    from test_utils import fix_paths

    fix_paths()
    absltest.main()
