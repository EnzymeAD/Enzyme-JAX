from absl.testing import absltest
from test_utils import EnzymeJaxTest, no_newxla, justjax

import jax
import jax.numpy as jnp


class AddOne(EnzymeJaxTest):
    def setUp(self):
        import jax.numpy as jnp

        self.ins = [
            jnp.array([1.0, 2.0, 3.0]),
            jnp.array([10.0, 20.0, 30.0]),
        ]
        self.dins = [
            jnp.array([0.1, 0.2, 0.3]),
            jnp.array([50.0, 70.0, 110.0]),
        ]
        self.douts = jnp.array([500.0, 700.0, 110.0])

        def add_one(x, y):
            return x + 1 + y

        self.primfilter = no_newxla
        self.fwdfilter = no_newxla
        self.revfilter = no_newxla

        self.fn = add_one

        self.name = "add_one"


class AddTwo(EnzymeJaxTest):
    def setUp(self):
        import jax.numpy as jnp

        self.ins = [
            jnp.array([1.0, 2.0, 3.0]),
            jnp.array([10.0, 20.0, 30.0]),
            jnp.array([100.0, 200.0, 300.0]),
        ]
        self.dins = [
            jnp.array([0.1, 0.2, 0.3]),
            jnp.array([50.0, 70.0, 110.0]),
            jnp.array([1300.0, 1700.0, 1900.0]),
        ]
        self.douts = jnp.array([500.0, 700.0, 110.0])

        def add_two(x, z, y):
            return x + y

        self.fn = add_two
        self.name = "add_two"


class Sum(EnzymeJaxTest):
    def setUp(self):
        import jax.numpy as jnp

        self.ins = [jnp.array(range(50), dtype=jnp.float32)]
        self.dins = [jnp.array([i * i for i in range(50)], dtype=jnp.float32)]
        self.douts = jnp.array(1.0)

        def sum(x):
            return jnp.sum(x)

        self.fn = sum
        self.name = "sum   "


class Cache(EnzymeJaxTest):
    def setUp(self):
        import jax.numpy as jnp

        dim = 288
        self.ins = [jnp.array(range(dim), dtype=jnp.float32)]
        self.dins = [jnp.array([i * i for i in range(dim)], dtype=jnp.float32)]
        self.douts = jnp.array([i * i for i in range(dim)], dtype=jnp.float32)

        self.primfilter = no_newxla
        self.fwdfilter = no_newxla
        self.revfilter = no_newxla

        def cache(x):
            return x * x[0]

        self.fn = cache
        self.name = "cache"
        self.rtol = 1e-5  # comparing values ~1e7 magnitudes


class Slicing(EnzymeJaxTest):
    def setUp(self):
        import jax.numpy as jnp

        dim = 3
        self.ins = [jnp.array(range(dim), dtype=jnp.float32).reshape(1, dim, 1)]
        self.dins = [
            jnp.array([i * i for i in range(dim)], dtype=jnp.float32).reshape(1, dim, 1)
        ]
        self.douts = jnp.array([i * i for i in range(dim)], dtype=jnp.float32)

        self.primfilter = no_newxla
        self.fwdfilter = no_newxla
        self.revfilter = no_newxla

        def slicing(x):
            return x[0, 0:1, 0] * jnp.ones((3,))

        self.fn = slicing
        self.name = "slicing"


class ActivityMismatch(EnzymeJaxTest):
    def setUp(self):
        import jax.numpy as jnp

        dim = 12
        self.ins = [jnp.array(range(dim), dtype=jnp.float32)]
        self.dins = [jnp.array([i * i for i in range(dim)], dtype=jnp.float32)]
        self.douts = jnp.array(
            [i * i for i in range(2 * dim)], dtype=jnp.float32
        ).reshape((2, dim))

        self.primfilter = no_newxla
        self.fwdfilter = no_newxla
        self.revfilter = justjax

        def f(x):
            toconv2 = jnp.ones((dim, dim))
            k = jnp.einsum("jk,k->j", toconv2, x)
            kcl = jnp.zeros((1, dim))
            h = jnp.reshape(k, (1, dim))
            kcl = jnp.append(kcl, h, axis=0)
            return kcl

        self.fn = f
        self.name = "actmtch"


class GenDot(EnzymeJaxTest):
    def setUp(self):
        import jax.numpy as jnp

        dim = 12
        self.ins = [jnp.array(range(dim), dtype=jnp.float32)]
        self.dins = [jnp.array([i * i for i in range(dim)], dtype=jnp.float32)]
        self.douts = jnp.array(
            [i * i for i in range(2 * dim)], dtype=jnp.float32
        ).reshape((2, dim))

        self.primfilter = no_newxla
        self.fwdfilter = no_newxla
        # No new xla runs but gets wrong answer
        # self.revfilter = no_newxla
        self.revfilter = justjax

        def f(x):
            k = jnp.ones((dim, dim)) @ x
            k_tmp = jnp.reshape(k, (2, dim // 2))

            toconv2 = jnp.ones((2, dim // 2, dim // 2))
            k = jnp.reshape(jnp.einsum("ijk,ik -> ij", toconv2, k_tmp), (dim,))

            kcl = jnp.zeros((1, dim))

            h = jnp.reshape(k, (1, dim))
            kcl = jnp.append(kcl, h, axis=0)
            return kcl

        self.fn = f
        self.name = "GenDot"


class Concat(EnzymeJaxTest):
    def setUp(self):
        import jax.numpy as jnp

        dim = 12
        self.ins = [
            jnp.array(range(dim), dtype=jnp.float32),
            10 * jnp.array(range(dim), dtype=jnp.float32),
        ]
        self.dins = [
            jnp.array([i * i for i in range(dim)], dtype=jnp.float32),
            jnp.array([i * i * i / 3.0 for i in range(dim)], dtype=jnp.float32),
        ]
        self.douts = jnp.array([i * i for i in range(2 * dim)], dtype=jnp.float32)

        self.revfilter = justjax

        def f(x, y):
            return jnp.concat([x, y], axis=None)

        self.fn = f
        self.name = "Concat"


class ValueAndGrad(EnzymeJaxTest):
    def setUp(self):
        def f(x, y):
            return (jnp.sum(x * y[0] + y[1]), y)

        self.mlirad_fwd = False
        self.mlirad_rev = False
        self.revfilter = lambda _: []
        self.fwdfilter = lambda _: []

        self.ins = [
            jnp.full((1,), 3, dtype=jnp.float32),
            (
                jnp.full((1,), 5, dtype=jnp.float64),
                jnp.full((1,), 7, dtype=jnp.int32),
            ),
        ]
        self.dins = []
        self.douts = []

        self.fn = jax.value_and_grad(f, allow_int=True, has_aux=True)
        self.name = "value_and_grad"


class ConstScatter(EnzymeJaxTest):
    def setUp(self):
        def forward(c_tau):
            Q = c_tau
            Q = Q.at[0].multiply(3)
            return (Q[0],)

        self.fn = forward
        self.name = "const_scatter"

        N = 1024**2
        self.ins = [jnp.full(N, 2.7)]
        self.dins = [jnp.full(N, 3.1)]
        self.douts = (self.dins[0],)

        # TODO: support multiply for scatter reverse mode
        self.revfilter = lambda _: []
        # self.revfilter = justjax


class ScatterSum(EnzymeJaxTest):
    def setUp(self):
        def energy_fn(R, neighbor):
            dR = R[neighbor[0]]
            return jnp.sum(jnp.sin(dR))

        nbrs = jnp.array([[2, 2]])

        def forward(position):
            e = jax.grad(energy_fn)(position, neighbor=nbrs)
            return (e,)

        self.fn = forward
        self.name = "scatter_sum"

        self.ins = [jnp.array([2.0, 4.0, 6.0, 8.0])]
        self.dins = [jnp.array([2.7, 3.1, 5.9, 4.2])]
        self.douts = self.fn(*self.ins)


if __name__ == "__main__":
    from test_utils import fix_paths

    fix_paths()

    absltest.main()
