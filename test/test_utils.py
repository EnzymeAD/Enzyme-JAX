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
import timeit

argv = ("-I/usr/include/c++/11", "-I/usr/include/x86_64-linux-gnu/c++/11")

devices = []
CurBackends = []
backends_initialized = False


def setup_backends():
    global backends_initialized
    global devices
    global CurBackends
    if backends_initialized:
        return
    backend = jax.default_backend()
    CurBackends.append(backend)
    if jax.default_backend() != "cpu":
        devices.append(backend)
    backends_initialized = True


AllBackends = ["cpu"] + devices
AllPipelines = [
    ("JaX  ", None, AllBackends),
    ("JaXPipe", JaXPipeline(), AllBackends),
    # ("NewXLAMLIR", NewXLAPipeline(mlirad=True)),
    # ("NewXLA", NewXLAPipeline()),
    ("OldXLA", OldXLAPipeline(), ["cpu"]),
]

partialopt = (
    "inline{default-pipeline=canonicalize max-iterations=4},"
    + """canonicalize,cse,
enzyme-hlo-generate-td{
            patterns=compare_op_canon<16>;
transpose_transpose<16>;
broadcast_in_dim_op_canon<16>;
convert_op_canon<16>;
dynamic_broadcast_in_dim_op_not_actually_dynamic<16>;
chained_dynamic_broadcast_in_dim_canonicalization<16>;
dynamic_broadcast_in_dim_all_dims_non_expanding<16>;
noop_reduce_op_canon<16>;
empty_reduce_op_canon<16>;
dynamic_reshape_op_canon<16>;
get_tuple_element_op_canon<16>;
real_op_canon<16>;
imag_op_canon<16>;
get_dimension_size_op_canon<16>;
gather_op_canon<16>;
reshape_op_canon<16>;
merge_consecutive_reshapes<16>;
transpose_is_reshape<16>;
zero_extent_tensor_canon<16>;
reorder_elementwise_and_shape_op<16>;

cse_broadcast_in_dim<16>;
cse_slice<16>;
cse_transpose<16>;
cse_convert<16>;
cse_pad<16>;
cse_dot_general<16>;
cse_reshape<16>;
cse_mul<16>;
cse_div<16>;
cse_add<16>;
cse_subtract<16>;
cse_min<16>;
cse_max<16>;
cse_neg<16>;
cse_concatenate<16>;

concatenate_op_canon<16>(1024);
select_op_canon<16>(1024);
add_simplify<16>;
sub_simplify<16>;
and_simplify<16>;
max_simplify<16>;
min_simplify<16>;
or_simplify<16>;
negate_simplify<16>;
mul_simplify<16>;
div_simplify<16>;
rem_simplify<16>;
pow_simplify<16>;
sqrt_simplify<16>;
cos_simplify<16>;
sin_simplify<16>;
noop_slice<16>;
const_prop_through_barrier<16>;
slice_slice<16>;
shift_right_logical_simplify<16>;
pad_simplify<16>;
negative_pad_to_slice<16>;
tanh_simplify<16>;
exp_simplify<16>;
slice_simplify<16>;
convert_simplify<16>;
dynamic_slice_to_static<16>;
dynamic_update_slice_elim<16>;
concat_to_broadcast<16>;
reduce_to_reshape<16>;
broadcast_to_reshape<16>;
gather_simplify<16>;
iota_simplify<16>(1024);
broadcast_in_dim_simplify<16>(1024);
convert_concat<1>;
dynamic_update_to_concat<1>;
slice_of_dynamic_update<1>;
slice_elementwise<1>;
slice_pad<1>;
dot_reshape_dot<1>;
concat_const_prop<1>;
concat_fuse<1>;
pad_reshape_pad<1>;
pad_pad<1>;
concat_push_binop_add<1>;
concat_push_binop_mul<1>;
scatter_to_dynamic_update_slice<1>;
reduce_concat<1>;
slice_concat<1>;

bin_broadcast_splat_add<1>;
bin_broadcast_splat_subtract<1>;
bin_broadcast_splat_div<1>;
bin_broadcast_splat_mul<1>;
slice_reshape<1>;

dot_reshape_pad<1>;
pad_dot_general<1>(1);
broadcast_reduce<1>;
            },
            transform-interpreter,
            enzyme-hlo-remove-transform,cse"""
)


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


def sync(x):
    return x.block_until_ready()


def syncall(x):
    return map(sync, x)


def fwdsync1(x):
    return (sync(x[0]), sync(x[1]))


def fwdsync2(x):
    return (sync(x[0][0]), sync(x[1][0]))


def fwdsync3(x):
    return (syncall(x[0]), syncall(x[1]))


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
        return grads

    return rev


def revsync0_0(x):
    return (sync(x[0]), sync(x[1][0]))


def revsync0_1(x):
    return (syncall(x[0]), sync(x[1][0]))


def revsync1_0(x):
    return (sync(x[0]), syncall(x[1]))


def revsync1_1(x):
    return (syncall(x[0]), syncall(x[1]))


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
        self.mlirad_fwd = True
        self.mlirad_rev = True

    def setUp(self):
        self.name = None

    def test(self):
        if self.name is None:
            return
        setup_backends()
        self.harness(self.name, self.fn, self.ins, self.dins, self.douts)

    def harness(self, name, in_fn, ins, dins, douts):
        assert len(ins) == len(dins)

        primalstr = "fn(" + (", ".join(["in" + str(i) for i in range(len(ins))])) + ")"
        if isinstance(douts, jax.Array):
            primalstr = "sync(" + primalstr + ")"
        elif len(douts) == 1:
            primalstr = "sync(" + primalstr + "[0])"
        else:
            primalstr = "syncall(" + primalstr + "[0])"

        fwdstr = (
            "fwd("
            + (", ".join(["in" + str(i) for i in range(len(ins))]))
            + ", "
            + (", ".join(["din" + str(i) for i in range(len(dins))]))
            + ")"
        )
        if isinstance(douts, jax.Array):
            fwdstr = "fwdsync1(" + fwdstr + ")"
        elif len(douts) == 1:
            fwdstr = "fwdsync2(" + fwdstr + ")"
        else:
            fwdstr = "fwdsync3(" + fwdstr + ")"

        revstr0 = (
            "rev(dout, " + (", ".join(["in" + str(i) for i in range(len(ins))])) + ")"
        )
        if self.revprimal:
            if len(dins) == 1:
                if isinstance(douts, jax.Array):
                    revstr = "revsync0_0(" + revstr0 + ")"
                else:
                    revstr = "revsync0_1(" + revstr0 + ")"
            else:
                if isinstance(douts, jax.Array):
                    revstr = "revsync1_0(" + revstr0 + ")"
                else:
                    revstr = "revsync1_1(" + revstr0 + ")"
        else:
            if len(dins) == 1:
                revstr = "sync(" + revstr0 + "[0])"
            else:
                revstr = "syncall(" + revstr0 + ")"
        revtransform = splatvjp if self.revprimal else splatvjp_noprim

        for backend in self.AllBackends:
            ins_backend = [to_backend(x, backend) for x in ins]
            dins_backend = [to_backend(x, backend) for x in dins]
            douts_backend = None
            if isinstance(douts, jax.Array):
                douts_backend = to_backend(douts, backend)
            else:
                douts_backend = tuple([to_backend(x, backend) for x in douts])

            primalins = {("in" + str(i)): ins_backend[i] for i in range(len(ins))}
            primalins["sync"] = sync
            primalins["syncall"] = syncall
            primalins["fwdsync1"] = fwdsync1
            primalins["fwdsync2"] = fwdsync2
            primalins["fwdsync3"] = fwdsync3
            primalins["revsync0_0"] = revsync0_0
            primalins["revsync0_1"] = revsync0_1
            primalins["revsync1_0"] = revsync1_0
            primalins["revsync1_1"] = revsync1_1
            fwdins = primalins | {
                ("din" + str(i)): dins_backend[i] for i in range(len(dins))
            }
            revins = primalins | {"dout": douts_backend}
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
                        timeit.Timer(
                            primalstr,
                            globals={
                                "fn": rfn_enzyme,
                            }
                            | primalins,
                        ).timeit(self.count)
                        / self.count,
                        sep="\t",
                    )

            # assert primres is not None
            fwdres = None

            for pname, pipeline, pbackends in self.fwdfilter(self.AllPipelines):
                if backend in pbackends:
                    if self.mlirad_fwd or pipeline is None:
                        rfn_enzyme = (
                            in_fn
                            if pipeline is None
                            else jax.jit(
                                enzyme_jax_ir(pipeline_options=pipeline, argv=argv)(
                                    in_fn
                                ),
                                # backend=backend
                            )
                        )
                        fwd_enzyme = jax.jit(
                            splatjvp(rfn_enzyme),
                            # backend=backend
                        )

                        primals, tangents = fwd_enzyme(*(ins_backend + dins_backend))

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
                            timeit.Timer(
                                fwdstr,
                                globals={
                                    "fwd": fwd_enzyme,
                                }
                                | fwdins,
                            ).timeit(self.count)
                            / self.count,
                            sep="\t",
                        )

            # assert fwdres is not None

            revres = None

            for pname, pipeline, pbackends in self.revfilter(self.AllPipelines):
                if backend in pbackends:

                    adout = douts_backend
                    if pipeline is not None:
                        if self.mlirad_rev or pipeline is None:
                            rfn_enzyme = (
                                in_fn
                                if pipeline is None
                                else enzyme_jax_ir(
                                    pipeline_options=pipeline, argv=argv
                                )(in_fn)
                            )
                            rev_enzyme = jax.jit(
                                revtransform(rfn_enzyme),
                                # backend=backend
                            )

                            if self.revprimal:
                                primals, grads = rev_enzyme(adout, *ins_backend)
                            else:
                                grads = rev_enzyme(adout, *ins_backend)
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
                                timeit.Timer(
                                    revstr,
                                    globals={
                                        "rev": rev_enzyme,
                                    }
                                    | revins,
                                ).timeit(self.count)
                                / self.count,
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
                            primals, grads = rev_enzyme(adout, *ins_backend)
                        else:
                            grads = rev_enzyme(adout, *ins_backend)
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
                            timeit.Timer(
                                revstr,
                                globals={
                                    "rev": rev_enzyme,
                                }
                                | revins,
                            ).timeit(self.count)
                            / self.count,
                            sep="\t",
                        )

                    if pipeline is None or (pipeline.mlir_ad() and self.mlirad_rev):
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
                            primals, grads = rev_enzyme(adout, *ins_backend)
                        else:
                            grads = rev_enzyme(adout, *ins_backend)
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
                            timeit.Timer(
                                revstr,
                                globals={
                                    "rev": rev_enzyme,
                                }
                                | revins,
                            ).timeit(self.count)
                            / self.count,
                            sep="\t",
                        )
