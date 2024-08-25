import jax
import jax.numpy as jnp
from enzyme_ad.jax import enzyme_jax_ir, NewXLAPipeline, OldXLAPipeline, JaXPipeline
from absl.testing import absltest
import timeit

argv = ("-I/usr/include/c++/11", "-I/usr/include/x86_64-linux-gnu/c++/11")
number = 1000

devices = []
if jax.default_backend() != "cpu":
    devices = [jax.default_backend()]

AllBackends = ["cpu"] + devices

AllPipelines = [
    ("JaX", None, AllBackends),
    ("JaXPipeline", JaXPipeline(), AllBackends),
    # ("NewXLAMLIR", NewXLAPipeline(mlirad=True)),
    # ("NewXLA", NewXLAPipeline()),
    ("OldXLA", OldXLAPipeline(), ["cpu"]),
]
PrimalPipelines = AllPipelines[:]
FwdPipelines = AllPipelines[:-1]
RevPipelines = AllPipelines[:-1]


def no_newxla(x):
    return [(name, a, b) for (name, a, b) in x if name != "NewXLAMLIR" and name != "NewXLA"]


def no_newxlamlir(x):
    return [(name, a, b) for (name, a, b) in x if name != "NewXLAMLIR"]

def nomlir(x):
    return [
        (name, a, b)
        for (name, a, b) in x
        if name != "NewXLAMLIR" and name != "NewXLA" # and name != "OldXLA"
    ]

def justjax(x):
    return [
        (name, a, b)
        for (name, a, b) in x
        if name != "NewXLAMLIR" and name != "NewXLA" and name != "OldXLA"
    ]


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

def to_backend(x, backend):
    return jax.device_put(x, jax.local_devices(backend=backend)[0])

class EnzymeJaxTest(absltest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.primfilter = lambda x: x
        self.fwdfilter = lambda x: x
        self.revfilter = lambda x: x

    def setUp(self):
        self.name = None

    def test(self):
        if self.name is None:
            return
        self.harness(self.name, self.fn, self.ins, self.dins, self.douts)

    def harness(self, name, in_fn, ins, dins, douts):
        assert len(ins) == len(dins)

        assert 1 == len(douts)

        primalstr = "fn(" + (", ".join(["in" + str(i) for i in range(len(ins))])) + ")"

        fwdstr = (
            "fwd("
            + (", ".join(["in" + str(i) for i in range(len(ins))]))
            + ", "
            + (", ".join(["din" + str(i) for i in range(len(dins))]))
            + ")"
        )

        revstr = (
            "rev(dout, " + (", ".join(["in" + str(i) for i in range(len(ins))])) + ")"
        )

        for backend in AllBackends:
            ins_backend = [to_backend(x, backend) for x in ins]
            dins_backend = [to_backend(x, backend) for x in dins]
            douts_backend = [to_backend(x, backend) for x in douts]
    
            primalins = {("in" + str(i)): ins_backend[i] for i in range(len(ins))}
            fwdins = primalins | {("din" + str(i)): dins_backend[i] for i in range(len(dins))}
            revins = primalins | {"dout": douts_backend[0]}

            primres = None

            for (pname, pipeline, pbackends) in self.primfilter(PrimalPipelines):
                if backend in pbackends:
                    rfn_enzyme = jax.jit(
                        in_fn if pipeline is None else enzyme_jax_ir(pipeline_options=pipeline, argv=argv)(in_fn),
                        #backend=backend
                    )
                    ao = rfn_enzyme(*ins_backend)
                    if primres is None:
                        primres = ao
                    else:
                        self.assertTrue((jnp.abs(ao - primres) < 1e-6).all())

                    print(
                        name,
                        ",",
                        pname,
                        ",",
                        backend,
                        ",",
                        "Primal,",
                        timeit.Timer(
                            primalstr,
                            globals={
                                "fn": rfn_enzyme,
                            }
                            | primalins,
                        ).timeit(number)
                        / number,
                    )

            assert primres is not None
            fwdres = None

            for (pname, pipeline, pbackends) in self.fwdfilter(FwdPipelines):
                if backend in pbackends:
                    rfn_enzyme = in_fn if pipeline is None else jax.jit(
                            enzyme_jax_ir(pipeline_options=pipeline, argv=argv)(in_fn),
                            #backend=backend
                    )
                    fwd_enzyme = jax.jit(splatjvp(rfn_enzyme),
                            #backend=backend
                    )

                    primals, tangents = fwd_enzyme(*(ins_backend + dins_backend))

                    self.assertTrue((jnp.abs(primals - primres) < 1e-6).all())

                    if fwdres is None:
                        fwdres = tangents
                    else:
                        if len(tangents.shape) == 0:
                            self.assertTrue((jnp.abs(tangents - fwdres) < 1e-6).all())
                        else:
                            for t, t_p in zip(tangents, fwdres):
                                self.assertTrue((jnp.abs(t - t_p) < 1e-6).all())

                    print(
                        name,
                        ",",
                        pname,
                        ",",
                        backend,
                        ",",
                        "Fwd",
                        ",",
                        timeit.Timer(
                            fwdstr,
                            globals={
                                "fwd": fwd_enzyme,
                            }
                            | fwdins,
                        ).timeit(number)
                        / number,
                    )

            assert fwdres is not None

            revres = None

            for (pname, pipeline, pbackends) in self.revfilter(RevPipelines):
                if backend in pbackends:
                    rfn_enzyme = in_fn if pipeline is None else jax.jit(
                        enzyme_jax_ir(pipeline_options=pipeline, argv=argv)(in_fn),
                        #backend=backend
                    )
                    rev_enzyme = jax.jit(splatvjp(rfn_enzyme),
                        #backend=backend
                    )

                    primals, grads = rev_enzyme(*douts_backend, *ins_backend)
                    self.assertTrue((jnp.abs(primals - primres) < 1e-6).all())

                    if revres is None:
                        revres = grads
                    else:
                        for i, (g, g_p) in enumerate(zip(grads, revres)):
                            self.assertTrue((jnp.abs(g - g_p) < 1e-6).all())

                    print(
                        name,
                        ",",
                        pname,
                        ",",
                        backend,
                        ",",
                        "Rev",
                        ",",
                        timeit.Timer(
                            revstr,
                            globals={
                                "rev": rev_enzyme,
                            }
                            | revins,
                        ).timeit(number)
                        / number,
                    )
            assert revres is not None


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

        self.primfilter = no_newxla
        self.fwdfilter = no_newxla
        self.revfilter = no_newxla

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

        self.primfilter = no_newxla
        self.fwdfilter = no_newxla
        self.revfilter = no_newxla

        def cache(x):
            return x * x[0]

        self.fn = cache
        self.name = "cache"


class Slicing(EnzymeJaxTest):
    def setUp(self):
        dim = 3
        self.ins = [jnp.array(range(dim), dtype=jnp.float32).reshape(1, dim, 1)]
        self.dins = [
            jnp.array([i * i for i in range(dim)], dtype=jnp.float32).reshape(1, dim, 1)
        ]
        self.douts = [jnp.array([i * i for i in range(dim)], dtype=jnp.float32)]

        self.primfilter = no_newxla
        self.fwdfilter = no_newxla
        self.revfilter = no_newxla

        def slicing(x):
            return x[0, 0:1, 0] * jnp.ones((3,))

        self.fn = slicing
        self.name = "slicing"


class ActivityMismatch(EnzymeJaxTest):
    def setUp(self):
        dim = 12
        self.ins = [jnp.array(range(dim), dtype=jnp.float32)]
        self.dins = [jnp.array([i * i for i in range(dim)], dtype=jnp.float32)]
        self.douts = [
            jnp.array([i * i for i in range(2 * dim)], dtype=jnp.float32).reshape(
                (2, dim)
            )
        ]

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
        self.name = "activitymismatch"


class GenDot(EnzymeJaxTest):
    def setUp(self):
        dim = 12
        self.ins = [jnp.array(range(dim), dtype=jnp.float32)]
        self.dins = [jnp.array([i * i for i in range(dim)], dtype=jnp.float32)]
        self.douts = [
            jnp.array([i * i for i in range(2 * dim)], dtype=jnp.float32).reshape(
                (2, dim)
            )
        ]

        def nomlir(x):
            return [
                (name, a)
                for (name, a) in x
                if name != "NewXLAMLIR" and name != "NewXLA" and name != "OldXLA"
            ]

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
        dim = 12
        self.ins = [
            jnp.array(range(dim), dtype=jnp.float32),
            10 * jnp.array(range(dim), dtype=jnp.float32),
        ]
        self.dins = [
            jnp.array([i * i for i in range(dim)], dtype=jnp.float32),
            jnp.array([i * i * i / 3.0 for i in range(dim)], dtype=jnp.float32),
        ]
        self.douts = [jnp.array([i * i for i in range(2 * dim)], dtype=jnp.float32)]

        self.revfilter = nomlir

        def f(x, y):
            return jnp.concat([x, y], axis=None)

        self.fn = f
        self.name = "Concat"


class ValueAndGrad(absltest.TestCase):
    def setUp(self):
        pass

    def test(self):
        def f(x, y):
            return (jnp.sum(x * y[0] + y[1]), y)

        filt = justjax

        for pname, pipeline, backends in AllPipelines:
            prevres = None
            for backend in backends:
                if (pname, pipeline) in filt(AllPipelines):
                    args = (
                        to_backend(3 * jnp.ones((1,), dtype=jnp.float32), backend),
                        (
                            to_backend(5 * jnp.ones((1,), dtype=jnp.float64), backend),
                            to_backend(7 * jnp.ones((1,), dtype=jnp.int32), backend),
                        ),
                    )

                    g = jax.value_and_grad(
                        f if pipeline is None else jax.jit(enzyme_jax_ir(pipeline_options=pipeline, argv=argv)(f),
                            #backend=backend
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
                        self.assertTrue((jnp.abs(res[0][0] - to_backend(prevres[0][0], backend)) < 1e-6).all())
                        self.assertTrue((jnp.abs(res[0][1][0] - to_backend(prevres[0][1][0], backend)) < 1e-6).all())
                        self.assertTrue((jnp.abs(res[0][1][1] - to_backend(prevres[0][1][1], backend)) < 1e-6).all())

                        self.assertTrue((jnp.abs(res[1] - to_backend(prevres[1], backend)) < 1e-6).all())


if __name__ == "__main__":
    absltest.main()
