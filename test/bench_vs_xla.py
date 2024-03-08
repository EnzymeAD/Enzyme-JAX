import jax
import jax.numpy as jnp
from enzyme_ad.jax import enzyme_jax_ir, NewXLAPipeline, OldXLAPipeline, JaXPipeline
from absl.testing import absltest
import timeit

argv = ("-I/usr/include/c++/11", "-I/usr/include/x86_64-linux-gnu/c++/11")
number = 1000

AllPipelines = [
    ("JaXPipeline", JaXPipeline()),
    ("NewXLAMLIR", NewXLAPipeline(mlirad=True)),
    ("NewXLA", NewXLAPipeline()),
    ("OldXLA", OldXLAPipeline()),
]
PrimalPipelines = AllPipelines
FwdPipelines = AllPipelines
RevPipelines = AllPipelines


# @jax.jit
# def fwd_jax(in0, in1, din0, din1):
# .  return jax.jvp(add_one_jax, (in0, in1), (din0, din1))
def splatjvp(in_fn):
    def fwd(*args):
        assert len(args) % 2 == 0
        return jax.jvp(
            in_fn, tuple(args[: len(args) // 2]), tuple(args[len(args) // 2 :])
        )

    return fwd


# @jax.jit
# def rev_jax(dout, in0, in1):
# primals, f_vjp = jax.vjp(add_one_jax, in0, in1)
# grads = f_vjp(dout)
# return primals, grads
def splatvjp(in_fn):
    def rev(dout, *args):
        primals, f_vjp = jax.vjp(in_fn, *args)
        grads = f_vjp(dout)
        return primals, grads

    return rev


class EnzymeJaxTest(absltest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.revfilter = lambda x: x

    def setUp(self):
        self.name = None

    def test(self):
        if self.name is None:
            return
        self.harness(self.name, self.fn, self.ins, self.dins, self.douts)

    def harness(self, name, in_fn, ins, dins, douts):
        assert len(ins) == len(dins)
        rfn_jax = jax.jit(in_fn)

        aop = rfn_jax(*ins)
        assert 1 == len(douts)

        primalstr = "fn(" + (", ".join(["in" + str(i) for i in range(len(ins))])) + ")"
        primalins = {("in" + str(i)): ins[0] for i in range(len(ins))}

        print(
            name + " JaX Primal: ",
            timeit.Timer(
                primalstr,
                globals={
                    "fn": rfn_jax,
                }
                | primalins,
            ).timeit(number)
            / number,
        )

        fwd_jax = jax.jit(splatjvp(rfn_jax))

        primals_p, tangents_p = fwd_jax(*(ins + dins))
        print(primals_p)
        print((jnp.abs(aop - primals_p) < 1e-6).all())
        self.assertTrue((jnp.abs(aop - primals_p) < 1e-6).all())

        fwdstr = (
            "fwd("
            + (", ".join(["in" + str(i) for i in range(len(ins))]))
            + ", "
            + (", ".join(["din" + str(i) for i in range(len(dins))]))
            + ")"
        )
        fwdins = primalins | {("din" + str(i)): dins[0] for i in range(len(dins))}
        print(
            name + " JaX Fwd: ",
            timeit.Timer(
                fwdstr,
                globals={
                    "fwd": fwd_jax,
                }
                | fwdins,
            ).timeit(number)
            / number,
        )

        assert len(douts) == 1

        rev_jax = jax.jit(splatvjp(rfn_jax))

        primals_p, grads_p = rev_jax(*douts, *ins)

        print(primals_p)
        print((jnp.abs(aop - primals_p) < 1e-6).all())
        self.assertTrue((jnp.abs(aop - primals_p) < 1e-6).all())

        revstr = (
            "rev(dout, " + (", ".join(["in" + str(i) for i in range(len(ins))])) + ")"
        )
        revins = primalins | {"dout": douts[0]}

        print(
            name + " JaX Rev: ",
            timeit.Timer(
                revstr,
                globals={
                    "rev": rev_jax,
                }
                | revins,
            ).timeit(number)
            / number,
        )

        for name, pipeline in AllPipelines:
            rfn_enzyme = enzyme_jax_ir(pipeline_options=pipeline, argv=argv)(in_fn)

            if (name, pipeline) in PrimalPipelines:
                ao = rfn_enzyme(*ins)
                print(aop)
                print((jnp.abs(aop - aop) < 1e-6).all())
                self.assertTrue((jnp.abs(ao - aop) < 1e-6).all())

                print(
                    name + " EnzymeMLIR(",
                    name,
                    ") Primal: ",
                    timeit.Timer(
                        primalstr,
                        globals={
                            "fn": rfn_enzyme,
                        }
                        | primalins,
                    ).timeit(number)
                    / number,
                )

            if (name, pipeline) in FwdPipelines:
                fwd_enzyme = jax.jit(splatjvp(rfn_enzyme))

                primals, tangents = fwd_jax(*(ins + dins))

                self.assertTrue((jnp.abs(primals - primals_p) < 1e-6).all())

                if len(tangents.shape) == 0:
                    self.assertTrue((jnp.abs(tangents - tangents_p) < 1e-6).all())
                else:
                    for t, t_p in zip(tangents, tangents_p):
                        self.assertTrue((jnp.abs(t - t_p) < 1e-6).all())

                print(
                    name + " EnzymeMLIR(",
                    name,
                    ") Fwd: ",
                    timeit.Timer(
                        fwdstr,
                        globals={
                            "fwd": fwd_enzyme,
                        }
                        | fwdins,
                    ).timeit(number)
                    / number,
                )

            if (name, pipeline) in self.revfilter(RevPipelines):
                rev_enzyme = jax.jit(splatvjp(rfn_enzyme))

                primals, grads = rev_enzyme(*douts, *ins)
                self.assertTrue((jnp.abs(primals - primals_p) < 1e-6).all())

                for i, (g, g_p) in enumerate(zip(grads, grads_p)):
                    print(i, g, g_p)
                    self.assertTrue((jnp.abs(g - g_p) < 1e-6).all())

                print(
                    name + " EnzymeMLIR(",
                    name,
                    ") Rev: ",
                    timeit.Timer(
                        revstr,
                        globals={
                            "rev": rev_enzyme,
                        }
                        | revins,
                    ).timeit(number)
                    / number,
                )


class AddOne(EnzymeJaxTest):
    def setUp(self):
        self.ins = [
            jnp.array([1.0, 2.0, 3.0]),
            jnp.array([10.0, 20.0, 30.0]),
        ]
        self.dins = [
            jnp.array([0.1, 0.2, 0.3]),
            jnp.array([50.0, 70.0, 110.0]),
        ]
        self.douts = [jnp.array([500.0, 700.0, 110.0])]

        def add_one(x, y):
            return x + 1 + y

        self.fn = add_one

        self.name = "add_one"


class AddTwo(EnzymeJaxTest):
    def setUp(self):
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
        self.douts = [jnp.array([500.0, 700.0, 110.0])]

        def add_two(x, z, y):
            return x + y

        self.fn = add_two
        self.name = "add_two"


class Sum(EnzymeJaxTest):
    def setUp(self):
        self.ins = [jnp.array(range(50), dtype=jnp.float32)]
        self.dins = [jnp.array([i * i for i in range(50)], dtype=jnp.float32)]
        self.douts = [1.0]

        def nomlir(x):
            return [(name, a) for (name, a) in x if name != "NewXLAMLIR"]

        self.revfilter = nomlir

        def sum(x):
            return jnp.sum(x)

        self.fn = sum
        self.name = "sum"


class Cache(EnzymeJaxTest):
    def setUp(self):
        dim = 288
        self.ins = [jnp.array(range(dim), dtype=jnp.float32)]
        self.dins = [jnp.array([i * i for i in range(dim)], dtype=jnp.float32)]
        self.douts = [jnp.array([i * i for i in range(dim)], dtype=jnp.float32)]

        def nomlir(x):
            return [(name, a) for (name, a) in x if name != "NewXLAMLIR"]

        self.revfilter = nomlir

        def cache(x):
            return x * x[0]

        self.fn = cache
        self.name = "cache"


class Slicing(EnzymeJaxTest):
    def setUp(self):
        dim = 3
        self.ins = [jnp.array(range(dim), dtype=jnp.float32).reshape(1, dim, 1)]
        self.dins = [jnp.array([i * i for i in range(dim)], dtype=jnp.float32).reshape(1, dim, 1)]
        self.douts = [jnp.array([i * i for i in range(dim)], dtype=jnp.float32)]

        def nomlir(x):
            return [(name, a) for (name, a) in x if name != "NewXLAMLIR"]

        self.revfilter = nomlir

        def slicing(x):
            return x[0, 0:1, 0] * jnp.ones((3,))

        self.fn = slicing
        self.name = "slicing"


class ActivityMismatch(EnzymeJaxTest):
    def setUp(self):
        dim = 12
        self.ins = [jnp.array(range(dim), dtype=jnp.float32)]
        self.dins = [jnp.array([i * i for i in range(dim)], dtype=jnp.float32)]
        self.douts = [jnp.array([i * i for i in range(2*dim)], dtype=jnp.float32).reshape((2, dim))]

        def nomlir(x):
            return [(name, a) for (name, a) in x if name != "NewXLAMLIR" and name != "NewXLA" and name != "OldXLA"]

        self.revfilter = nomlir

        def f(x):
            toconv2 = jnp.ones((dim, dim))
            k = jnp.einsum('jk,k->j', toconv2, x)
            kcl = jnp.zeros((1, dim))
            h = jnp.reshape(k, (1, dim))
            kcl = jnp.append(kcl, h, axis=0)
            return kcl

        self.fn = f
        self.name = "activitymismatch"

class GenDot(EnzymeJaxTest):
    def setUp(self):
        dim = 12
        self.ins = [jnp.array(range(dim), dtype=jnp.float32)]
        self.dins = [jnp.array([i * i for i in range(dim)], dtype=jnp.float32)]
        self.douts = [jnp.array([i * i for i in range(2*dim)], dtype=jnp.float32).reshape((2, dim))]

        def nomlir(x):
            return [(name, a) for (name, a) in x if name != "NewXLAMLIR" and name != "NewXLA" and name != "OldXLA"]

        self.revfilter = nomlir

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
        dim = 12
        self.ins = [jnp.array(range(dim), dtype=jnp.float32), 10*jnp.array(range(dim), dtype=jnp.float32)]
        self.dins = [jnp.array([i * i for i in range(dim)], dtype=jnp.float32), jnp.array([i * i *i / 3. for i in range(dim)], dtype=jnp.float32)]
        self.douts = [jnp.array([i * i for i in range(2*dim)], dtype=jnp.float32)]

        def nomlir(x):
            return [(name, a) for (name, a) in x if name != "NewXLAMLIR" and name != "NewXLA" and name != "OldXLA"]

        self.revfilter = nomlir

        def f(x, y):
            return jnp.concat([x, y], axis=None)

        self.fn = f
        self.name = "Concat"

if __name__ == "__main__":
    absltest.main()
