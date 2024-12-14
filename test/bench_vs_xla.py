from absl.testing import absltest
from test_utils import *


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


class ValueAndGrad(absltest.TestCase):
    def setUp(self):
        pass

    def test(self):
        from enzyme_ad.jax import enzyme_jax_ir
        import jax.numpy as jnp

        def f(x, y):
            return (jnp.sum(x * y[0] + y[1]), y)

        filt = justjax

        for pname, pipeline, backends in AllPipelines():
            prevres = None
            for backend in backends:
                if (pname, pipeline) in filt(AllPipelines()):
                    args = (
                        to_backend(3 * jnp.ones((1,), dtype=jnp.float32), backend),
                        (
                            to_backend(5 * jnp.ones((1,), dtype=jnp.float64), backend),
                            to_backend(7 * jnp.ones((1,), dtype=jnp.int32), backend),
                        ),
                    )

                    g = jax.value_and_grad(
                        (
                            f
                            if pipeline is None
                            else jax.jit(
                                enzyme_jax_ir(pipeline_options=pipeline, argv=argv)(f),
                                # backend=backend
                            )
                        ),
                        has_aux=True,
                        allow_int=True,
                    )

                    res = g(*args)
                    if prevres is None:
                        prevres = res
                    else:
                        name = "valueandgrad"
                        print(name + " JaX(", pname, "): ", prevres)
                        print(name + " EnzymeMLIR(", pname, "): ", res)
                        self.assertTrue(
                            (
                                jnp.abs(res[0][0] - to_backend(prevres[0][0], backend))
                                < 1e-6
                            ).all()
                        )
                        self.assertTrue(
                            (
                                jnp.abs(
                                    res[0][1][0] - to_backend(prevres[0][1][0], backend)
                                )
                                < 1e-6
                            ).all()
                        )
                        self.assertTrue(
                            (
                                jnp.abs(
                                    res[0][1][1] - to_backend(prevres[0][1][1], backend)
                                )
                                < 1e-6
                            ).all()
                        )

                        self.assertTrue(
                            (
                                jnp.abs(res[1] - to_backend(prevres[1], backend)) < 1e-6
                            ).all()
                        )


class ConstScatter(EnzymeJaxTest):
    def setUp(self):
        import jax.numpy as jnp

        def forward(c_tau):
            Q = c_tau
            Q = Q.at[0].multiply(3)
            chain = (Q,)
            return (chain[0],)

        self.fn = forward
        self.name = "const_scatter"
        self.count = 10

        self.ins = [
            jnp.array([2.7, 2.7, 2.7]),
        ]
        self.dins = [jnp.array([3.1, 3.1, 3.1])]
        self.douts = (self.dins[0],)
        self.revfilter = lambda _: []
        # No support for stablehlo.while atm
        # self.revfilter = justjax
        self.mlirad_rev = False


class ScatterSum(EnzymeJaxTest):
    def setUp(self):
        import jax
        import jax.numpy as jnp

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
        self.revfilter = lambda _: []
        # No support for stablehlo.scatter atm
        self.mlirad_rev = False


if __name__ == "__main__":
    from test_utils import fix_paths

    fix_paths()

    absltest.main()
