import jax
import jax.numpy as jnp
from enzyme_ad.jax import (
    enzyme_jax_ir,
    NewXLAPipeline,
    OldXLAPipeline,
    JaXPipeline,
    hlo_opts,
)
from absl.testing import absltest
from timeit import Timer
from statistics import mean, stdev, median
from datetime import datetime
import os
import csv

def dump_to_csv(filename, pipeline, stage, runtime_ms):
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # write the header only if the file doesn't exist
        if not file_exists:
            writer.writerow(['pipeline', 'stage', 'runtime_ms'])
        for runtime in runtime_ms:
            writer.writerow([pipeline, stage, runtime])

def timeit(filename, pipeline, stage, str, globals, count, warmup=100):
    timer = Timer(str, globals=globals)
    _warmup = timer.repeat(repeat=1, number=warmup)
    runtime_ms = [x * 1000 for x in timer.repeat(repeat=count, number=1)]
    dump_to_csv(filename, pipeline, stage, runtime_ms)
    return f"{median(runtime_ms):.6f} (min {min(runtime_ms):.6f}, max {max(runtime_ms):.6f}, mean {mean(runtime_ms):.6f} ± {stdev(runtime_ms):.6f})"

argv = ("-I/usr/include/c++/11", "-I/usr/include/x86_64-linux-gnu/c++/11")

devices = []
CurBackends = [jax.default_backend()]

if jax.default_backend() != "cpu":
    devices = CurBackends

AllBackends = ["cpu"] + devices
AllPipelines = [
    ("JaX", None, AllBackends),
    ("JaXPipe", JaXPipeline(), AllBackends),
    # ("NewXLAMLIR", NewXLAPipeline(mlirad=True)),
    # ("NewXLA", NewXLAPipeline()),
    ("OldXLA", OldXLAPipeline(), ["cpu"]),
]


def no_newxla(x):
    return [
        (name, a, b) for (name, a, b) in x if name != "NewXLAMLIR" and name != "NewXLA"
    ]


def no_newxlamlir(x):
    return [(name, a, b) for (name, a, b) in x if name != "NewXLAMLIR"]


def nomlir(x):
    return [
        (name, a, b)
        for (name, a, b) in x
        if name != "NewXLAMLIR" and name != "NewXLA"  # and name != "OldXLA"
    ]


def justjax(x):
    return [
        (name, a, b) for (name, a, b) in x if a is None or isinstance(a, JaXPipeline)
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


def splatvjp_noprim(in_fn):
    def rev(dout, *args):
        primals, f_vjp = jax.vjp(in_fn, *args)
        grads = f_vjp(dout)
        return primals, grads

    return rev


def to_backend(x, backend):
    dev = jax.local_devices(backend=backend)[0]
    return jax.device_put(x, dev)


def recursive_check(tester, lhs, rhs, tol=1e-6):
    tester.assertEqual(type(lhs), type(rhs))
    if isinstance(lhs, jax.Array):
        legal = (jnp.abs(lhs - rhs) < tol).all()
        if not legal:
            print("lhs", lhs)
            print("rhs", rhs)
            print("abs", jnp.abs(lhs - rhs))
            print("eq", jnp.abs(lhs - rhs) < tol)
            print("max", jnp.max(jnp.abs(lhs - rhs)))
        tester.assertTrue(legal)
        return

    if isinstance(lhs, tuple):
        for i, (g, g_p) in enumerate(zip(lhs, rhs)):
            recursive_check(tester, g, g_p, tol)
        return

    if isinstance(lhs, dict):
        tester.assertEqual(lhs.keys(), rhs.keys())
        for k in lhs.keys():
            recursive_check(tester, lhs[k], rhs[k], tol)
        return

    print("Unknown recursive type", type(lhs), " ", type(rhs))
    tester.assertTrue(False)


class EnzymeJaxTest(absltest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.primfilter = lambda x: x
        self.fwdfilter = lambda x: x
        self.revfilter = lambda x: x
        self.count = 10000
        self.AllBackends = AllBackends
        self.AllPipelines = AllPipelines
        self.revprimal = True
        self.tol = 1e-6

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

        csv_filename = datetime.now().strftime("results_%Y-%m-%d_%H:%M:%S.csv")

        for backend in self.AllBackends:
            ins_backend = [to_backend(x, backend) for x in ins]
            dins_backend = [to_backend(x, backend) for x in dins]
            douts_backend = [to_backend(x, backend) for x in douts]

            primalins = {("in" + str(i)): ins_backend[i] for i in range(len(ins))}
            fwdins = primalins | {
                ("din" + str(i)): dins_backend[i] for i in range(len(dins))
            }
            revins = primalins | {"dout": douts_backend[0]}

            primres = None

            for pname, pipeline, pbackends in self.primfilter(self.AllPipelines):
                if backend in pbackends:
                    rfn_enzyme = jax.jit(
                        (
                            in_fn
                            if pipeline is None
                            else enzyme_jax_ir(pipeline_options=pipeline, argv=argv)(
                                in_fn
                            )
                        ),
                        # backend=backend
                    )
                    ao = rfn_enzyme(*ins_backend)
                    if primres is None:
                        primres = ao
                    else:
                        recursive_check(self, ao, primres, self.tol)

                    print(
                        name,
                        ",",
                        pname,
                        ",",
                        backend,
                        ",",
                        "Primal",
                        ",",
                        timeit(csv_filename, pname, "Primal", primalstr, {'fn': rfn_enzyme} | primalins, self.count),
                        sep="\t",
                    )

            # assert primres is not None
            fwdres = None

            for pname, pipeline, pbackends in self.fwdfilter(self.AllPipelines):
                if backend in pbackends:
                    rfn_enzyme = in_fn
                    fwd_enzyme = jax.jit(
                        (
                            splatjvp(rfn_enzyme)
                            if pipeline is None
                            else enzyme_jax_ir(
                                pipeline_options=pipeline, argv=argv
                            )(splatjvp(rfn_enzyme))
                        ),
                        # backend=backend
                    )

                    primals, tangents = fwd_enzyme(*(ins_backend+dins_backend))

                    recursive_check(self, primals, primres, self.tol)

                    if fwdres is None:
                        fwdres = tangents
                    else:
                        recursive_check(self, tangents, fwdres, self.tol)

                    print(
                        name,
                        ",",
                        pname,
                        ",",
                        backend,
                        ",",
                        "Forward",
                        ",",
                        timeit(csv_filename, pname, "Forward", fwdstr, {'fwd': fwd_enzyme} | fwdins, self.count),
                        sep="\t",
                    )

            # assert fwdres is not None

            revres = None

            revtransform = splatvjp if self.revprimal else splatvjp_noprim

            for pname, pipeline, pbackends in self.revfilter(self.AllPipelines):
                if backend in pbackends:
                    if pipeline is not None:
                        rfn_enzyme = (
                            in_fn
                            if pipeline is None
                            else enzyme_jax_ir(pipeline_options=pipeline, argv=argv)(
                                in_fn
                            )
                        )
                        rev_enzyme = jax.jit(
                            revtransform(rfn_enzyme),
                            # backend=backend
                        )

                        if self.revprimal:
                            primals, grads = rev_enzyme(*douts_backend, *ins_backend)
                        else:
                            grads = rev_enzyme(*douts_backend, *ins_backend)
                            assert grads is not None

                        if self.revprimal and primres is not None:
                            recursive_check(self, primals, primres, self.tol)

                        if revres is None:
                            revres = grads
                        else:
                            recursive_check(self, grads, revres, self.tol)

                        print(
                            name,
                            ",",
                            pname,
                            ",",
                            backend,
                            ",",
                            "PreRev",
                            ",",
                            timeit(csv_filename, pname, "PreRev", revstr, {'rev': rev_enzyme} | revins, self.count),
                            sep="\t",
                        )

                        rfn_enzyme = in_fn
                        rev_enzyme = jax.jit(
                            (
                                revtransform(rfn_enzyme)
                                if pipeline is None
                                else enzyme_jax_ir(
                                    pipeline_options=pipeline, argv=argv
                                )(revtransform(rfn_enzyme))
                            ),
                            # backend=backend
                        )

                        if self.revprimal:
                            primals, grads = rev_enzyme(*douts_backend, *ins_backend)
                        else:
                            grads = rev_enzyme(*douts_backend, *ins_backend)
                            assert grads is not None

                        if self.revprimal and primres is not None:
                            recursive_check(self, primals, primres, self.tol)

                        if revres is None:
                            revres = grads
                        else:
                            recursive_check(self, grads, revres, self.tol)

                        print(
                            name,
                            ",",
                            pname,
                            ",",
                            backend,
                            ",",
                            "PostRev",
                            ",",
                            timeit(csv_filename, pname, "PostRev", revstr, {'rev': rev_enzyme} | revins, self.count),
                            sep="\t",
                        )

                    if pipeline is None or pipeline.mlir_ad():
                        rfn_enzyme = (
                            in_fn
                            if pipeline is None
                            else enzyme_jax_ir(pipeline_options=pipeline, argv=argv)(
                                in_fn
                            )
                        )
                        rev_enzyme = jax.jit(
                            (
                                revtransform(rfn_enzyme)
                                if pipeline is None
                                else enzyme_jax_ir(
                                    pipeline_options=pipeline, argv=argv
                                )(revtransform(rfn_enzyme))
                            ),
                            # backend=backend
                        )

                        if self.revprimal:
                            primals, grads = rev_enzyme(*douts_backend, *ins_backend)
                        else:
                            grads = rev_enzyme(*douts_backend, *ins_backend)
                            assert grads is not None

                        if self.revprimal and primres is not None:
                            recursive_check(self, primals, primres, self.tol)

                        if revres is None:
                            revres = grads
                        else:
                            recursive_check(self, grads, revres, self.tol)

                        print(
                            name,
                            ",",
                            pname,
                            ",",
                            backend,
                            ",",
                            "BothRev",
                            ",",
                            timeit(csv_filename, pname, "BothRev", revstr, {'rev': rev_enzyme} | revins, self.count),
                            sep="\t",
                        )
            assert revres is not None
