def fix_paths():
    import os

    for nm in [
        "NV_LIBCUBLAS_VERSION",
        "NVIDIA_VISIBLE_DEVICES",
        "NV_NVML_DEV_VERSION",
        "NV_LIBNCCL_DEV_PACKAGE",
        "NV_LIBNCCL_DEV_PACKAGE_VERSION",
        "NVIDIA_REQUIRE_CUDA",
        "NV_LIBCUBLAS_DEV_PACKAGE",
        "NV_NVTX_VERSION",
        "NV_NVPROF_VERSION",
        "NV_LIBNCCL_PACKAGE",
        "NV_LIBCUBLAS_PACKAGE_NAME",
        "NV_LIBNPP_DEV_VERSION",
        "NV_LIBCUSPARSE_DEV_VERSION",
        "NV_CUDA_LIB_VERSION",
        "NVARCH",
        "NV_CUDA_NSIGHT_COMPUTE_VERSION",
        "NV_LIBNCCL_PACKAGE_NAME",
        "NV_LIBNCCL_PACKAGE_VERSION",
    ]:
        os.environ.pop(nm, None)

    runfiles = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    # https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
    # os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    CUDA_DIR = os.path.join(
        runfiles, "pypi_nvidia_cuda_nvcc_cu12", "site-packages", "nvidia", "cuda_nvcc"
    )
    os.environ["CUDA_DIR"] = CUDA_DIR
    os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=" + CUDA_DIR

    LD_LIB = os.environ.get("LD_LIBRARY_PATH", "")
    LD_LIB = (
        os.path.join(
            runfiles,
            "pypi_nvidia_cusolver_cu12",
            "site-packages",
            "nvidia",
            "cusolver",
            "lib",
        )
        + ":"
        + LD_LIB
    )
    LD_LIB = (
        os.path.join(
            runfiles,
            "pypi_nvidia_cudnn_cu12",
            "site-packages",
            "nvidia",
            "cudnn",
            "lib",
        )
        + ":"
        + LD_LIB
    )
    LD_LIB = (
        os.path.join(
            runfiles,
            "pypi_nvidia_cublas_cu12",
            "site-packages",
            "nvidia",
            "cublas",
            "lib",
        )
        + ":"
        + LD_LIB
    )
    LD_LIB = (
        os.path.join(
            runfiles,
            "pypi_nvidia_cuda_cupti_cu12",
            "site-packages",
            "nvidia",
            "cuda_cupti",
            "lib",
        )
        + ":"
        + LD_LIB
    )
    LD_LIB = (
        os.path.join(
            runfiles,
            "pypi_nvidia_cuda_runtime_cu12",
            "site-packages",
            "nvidia",
            "cuda_runtime",
            "lib",
        )
        + ":"
        + LD_LIB
    )

    os.environ["LD_LIBRARY_PATH"] = LD_LIB

    PATH = os.environ.get("PATH", "")
    PATH = (
        os.path.join(
            runfiles,
            "pypi_nvidia_cuda_nvcc_cu12",
            "site-packages",
            "nvidia",
            "cuda_nvcc",
            "bin",
        )
        + ":"
        + PATH
    )
    os.environ["PATH"] = PATH

    CUDNN_PATH = os.path.join(
        runfiles, "pypi_nvidia_cudnn_cu12", "site-packages", "nvidia", "cudnn"
    )
    os.environ["CUDNN_PATH"] = CUDNN_PATH

    # Somewhere, someone hardcodes the path to the nvidia libs
    src_path = os.path.join(
        runfiles, "pypi_nvidia_cuda_runtime_cu12", "site-packages", "nvidia"
    )
    if os.path.exists(src_path):
        for dst_path in [
            os.path.join(runfiles, "pypi_jax_cuda12_plugin", "nvidia"),
            os.path.join(runfiles, "pypi_jax_cuda12_pjrt", "nvidia"),
        ]:
            if not os.path.exists(dst_path):
                os.symlink(src_path, dst_path)

    # Hardcoding also exists in tensorflow....and causes a segfault in jax otherwise???
    for src_path in [
        os.path.join(runfiles, "pypi_tensorflow", "site-packages", "nvidia"),
        os.path.join(runfiles, "pypi_tensorflow_cpu", "site-packages", "nvidia"),
    ]:
        dst_path = os.path.join(runfiles, "pypi_jax_cuda12_plugin", "nvidia")
        if os.path.exists(src_path):
            if not os.path.exists(dst_path):
                os.symlink(src_path, dst_path)

    # And finally because a path to cublas can't be found otherwise
    # or worse it will use an incorrect version thereof. The reason is because
    # when dlopen opens a file it uses the LD_LIBRARY_PATH from the start
    # of program execution. However, jax looks at pypath variables and will
    # think that its version will exist. However, it just calls dlopen without
    # a full path, and will end up in cublas/cudnn internal errors with mismatched
    # versions. If we force loading the right version, dlopen will not reopen
    # an incorrect library.
    cublas_path = os.path.join(
        runfiles,
        "pypi_nvidia_cublas_cu12",
        "site-packages",
        "nvidia",
        "cublas",
        "lib",
        "libcublas.so.12",
    )

    if os.path.exists(cublas_path):

        import ctypes

        ctypes.cdll.LoadLibrary(cublas_path)

    cudnngraph_path = os.path.join(
        runfiles,
        "pypi_nvidia_cudnn_cu12",
        "site-packages",
        "nvidia",
        "cudnn",
        "lib",
        "libcudnn_graph.so.9",
    )

    if os.path.exists(cudnngraph_path):
        import ctypes

        ctypes.cdll.LoadLibrary(cudnngraph_path)

    cudnn_path = os.path.join(
        runfiles,
        "pypi_nvidia_cudnn_cu12",
        "site-packages",
        "nvidia",
        "cudnn",
        "lib",
        "libcudnn.so.9",
    )

    if os.path.exists(cudnn_path):
        import ctypes

        ctypes.cdll.LoadLibrary(cudnn_path)

    # jitlink must come before cusolver
    jitlink_path = os.path.join(
        runfiles,
        "pypi_nvidia_nvjitlink_cu12",
        "site-packages",
        "nvidia",
        "nvjitlink",
        "lib",
        "libnvJitLink.so.12",
    )

    if os.path.exists(jitlink_path):
        import ctypes

        ctypes.cdll.LoadLibrary(jitlink_path)

    # cusparse comes before cusolver but after jitlink
    cusparse_path = os.path.join(
        runfiles,
        "pypi_nvidia_cusparse_cu12",
        "site-packages",
        "nvidia",
        "cusparse",
        "lib",
        "libcusparse.so.12",
    )

    if os.path.exists(cusparse_path):
        import ctypes

        ctypes.cdll.LoadLibrary(cusparse_path)

    cusolver_path = os.path.join(
        runfiles,
        "pypi_nvidia_cusolver_cu12",
        "site-packages",
        "nvidia",
        "cusolver",
        "lib",
        "libcusolver.so.11",
    )

    if os.path.exists(cusolver_path):
        import ctypes

        ctypes.cdll.LoadLibrary(cusolver_path)


from absl.testing import absltest
from timeit import Timer
from statistics import mean, stdev, median
from datetime import datetime
import os
import csv

argv = ("-I/usr/include/c++/11", "-I/usr/include/x86_64-linux-gnu/c++/11")

devices = []
CurBackends = []
AllBackends = []
backends_initialized = False


def setup_backends():
    global backends_initialized
    global devices
    global CurBackends
    global AllBackends
    if backends_initialized:
        return
    import jax

    AllBackends.append("cpu")
    backend = jax.default_backend()
    CurBackends.append(backend)
    if backend != "cpu":
        devices.append(backend)
        AllBackends.append(backend)

    backends_initialized = True


# def AllPipelines():
#     setup_backends()
#     from enzyme_ad.jax import (
#         NewXLAPipeline,
#         OldXLAPipeline,
#         JaXPipeline,
#         hlo_opts,
#     )

#     return [
#         ("JaX  ", None, AllBackends),
#         ("JaXPipe", JaXPipeline(), AllBackends),
#         # ("NewXLAMLIR", NewXLAPipeline(mlirad=True)),
#         # ("NewXLA", NewXLAPipeline()),
#         ("OldXLA", OldXLAPipeline(), ["cpu"]),
#     ]


reduced_opt = (
    "inline{default-pipeline=canonicalize max-iterations=4},"
    + """canonicalize,cse,
enzyme-hlo-generate-td{
            patterns=noop_slice<16>;
slice_slice<16>;
slice_broadcast<16>;
slice_transpose<16>;
slice_pad<1>;
slice_concat<1>;
pad_simplify<16>;
negative_pad_to_slice<16>;
binop_pad_to_concat_add<16>;
binop_pad_to_concat_mul<16>;
unary_pad_push_tanh<1>;
unary_pad_push_exp<1>;
transpose_pad<1>;
concat_push_binop_add<1>;
concat_push_binop_mul<1>;
reshape_empty_broadcast<1>;
broadcast_reshape<1>;
broadcast_to_reshape<16>;
broadcast_pad<1>;
concat_pad<1>;
slice_simplify<16>;
transpose_simplify<16>;
transpose_transpose<1>;
transpose_dot_reorder<1>;
dot_transpose<1>;
binop_pad_pad_add<1>;
binop_pad_pad_subtract<1>;
binop_pad_pad_mul<1>;
binop_pad_pad_div<1>;
binop_pad_pad_min<1>;
binop_pad_pad_max<1>;
add_pad_pad_to_concat<1>;
slice_dot_general<1>;
slice_reshape_slice<1>;
slice_reshape_concat<1>;
slice_reshape_transpose<1>;
slice_reshape_pad<1>;
            },
            transform-interpreter,
            enzyme-hlo-remove-transform,cse"""
)

def pipelines():
    setup_backends()
    from enzyme_ad.jax import (
        JaXPipeline,
        hlo_opts,
    )

    eqsat = (
        "EqSat",
        JaXPipeline(
            "inline{default-pipeline=canonicalize max-iterations=4},"
            + "equality-saturation-pass"
        ),
        CurBackends,
    )

    eqsat_only = os.getenv("EQSAT_ONLY") == "true"
    if eqsat_only:
        return [eqsat]

    return [
        ("JaX  ", None, CurBackends),
        # ("JaXPipe", JaXPipeline(), CurBackends),
        # ("ReducedOpt", JaXPipeline(reduced_opt), CurBackends),
        ("DefOpt", JaXPipeline(hlo_opts()), CurBackends),
        eqsat,
        # (
        #     "HLOOpt",
        #     JaXPipeline(
        #         "inline{default-pipeline=canonicalize inlining-threshold=4294967295 max-iterations=4},"
        #         + "canonicalize,cse,enzyme-hlo-opt,cse"
        #     ),
        #     CurBackends,
        # ),
        # (
        #     "IPartOpt",
        #     JaXPipeline(
        #         "inline{default-pipeline=canonicalize inlining-threshold=4294967295 max-iterations=4},"
        #         + partialopt
        #     ),
        #     CurBackends,
        # ),
        # (
        #     "IDefOpt",
        #     JaXPipeline(
        #         "inline{default-pipeline=canonicalize inlining-threshold=4294967295 max-iterations=4},"
        #         + hlo_opts()
        #     ),
        #     CurBackends,
        # ),
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
    from enzyme_ad.jax import JaXPipeline

    return [
        (name, a, b) for (name, a, b) in x if a is None or isinstance(a, JaXPipeline)
    ]


def splatjvp(in_fn):
    import jax

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


def splatvjp(in_fn):
    import jax

    def rev(dout, *args):
        primals, f_vjp = jax.vjp(in_fn, *args)
        grads = f_vjp(dout)
        return primals, grads

    return rev


def splatvjp_noprim(in_fn):
    import jax

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
    import jax

    dev = jax.local_devices(backend=backend)[0]
    return jax.device_put(x, dev)


def recursive_check(tester, lhs, rhs, tol=1e-6):
    import jax.numpy as jnp
    import jax

    tester.assertEqual(type(lhs), type(rhs))
    if isinstance(lhs, jax.Array):
        legal_rel = ((jnp.abs(lhs - rhs) / jnp.maximum(lhs, rhs)) < tol).all()
        legal_abs = (jnp.abs(lhs - rhs) < tol).all()
        legal = legal_rel or legal_abs
        if not legal:
            print("lhs", lhs)
            print("rhs", rhs)
            print("abs", jnp.abs(lhs - rhs))
            print("eq", (jnp.abs(lhs - rhs) / jnp.maximum(lhs, rhs)) < tol)
            print("max", jnp.max(jnp.abs(lhs - rhs)))
            print("max rel", jnp.max(jnp.abs(lhs - rhs) / jnp.maximum(lhs, rhs)))
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
        setup_backends()
        self.primfilter = lambda x: x
        self.fwdfilter = lambda x: x
        self.revfilter = lambda x: x
        self.count = 100
        self.AllBackends = AllBackends
        self.AllPipelines = pipelines()
        self.revprimal = True
        self.tol = 1e-6
        self.mlirad_fwd = True
        self.mlirad_rev = True
        EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
        if EXPERIMENT_NAME:
            # self.csv_filename = f"results_{EXPERIMENT_NAME}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.csv"
            self.csv_filename = f"results_{EXPERIMENT_NAME}.csv"
        else:
            self.csv_filename = f"results_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.csv"

    def dump_to_csv(self, filename, pipeline, stage, runtime_ms):
        file_exists = os.path.isfile(filename)

        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            # write the header only if the file doesn't exist
            if not file_exists:
                writer.writerow(['pipeline', 'stage', 'runtime_ms'])
            for runtime in runtime_ms:
                writer.writerow([pipeline, stage, runtime])

    def timeit(self, filename, pipeline, stage, str, globals, count, warmup=100):
        timer = Timer(str, globals=globals)
        _warmup = timer.repeat(repeat=1, number=warmup)
        runtime_ms = [x * 1000 for x in timer.repeat(repeat=count, number=1)]
        self.dump_to_csv(filename, pipeline, stage, runtime_ms)
        return f"{median(runtime_ms):.6f} (min {min(runtime_ms):.6f}, max {max(runtime_ms):.6f}, mean {mean(runtime_ms):.6f} ± {stdev(runtime_ms):.6f})"

    def setUp(self):
        self.name = None

    def test(self):
        if self.name is None:
            return
        setup_backends()
        self.harness(self.name, self.fn, self.ins, self.dins, self.douts)

    def harness(self, name, in_fn, ins, dins, douts):
        import timeit
        import jax
        from enzyme_ad.jax import enzyme_jax_ir

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
                            else enzyme_jax_ir(
                                pipeline_options=pipeline, argv=argv, inner_jit=False
                            )(in_fn)
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
                        self.timeit(self.csv_filename, pname, "Primal", primalstr, {'fn': rfn_enzyme} | primalins, self.count),
                        sep="\t",
                    )

            # # assert primres is not None
            # fwdres = None

            # for pname, pipeline, pbackends in self.fwdfilter(self.AllPipelines):
            #     if backend in pbackends:
            #         if self.mlirad_fwd or pipeline is None:
            #             rfn_enzyme = (
            #                 in_fn
            #                 if pipeline is None
            #                 else jax.jit(
            #                     enzyme_jax_ir(
            #                         pipeline_options=pipeline,
            #                         argv=argv,
            #                         inner_jit=False,
            #                     )(in_fn),
            #                     # backend=backend
            #                 )
            #             )
            #             fwd_enzyme = jax.jit(
            #                 (
            #                     splatjvp(rfn_enzyme)
            #                 if pipeline is None
            #                 else enzyme_jax_ir(
            #                         pipeline_options=pipeline, argv=argv
            #                 )(splatjvp(rfn_enzyme))
            #                 ),
            #                 # backend=backend
            #             )

            #             primals, tangents = fwd_enzyme(*(ins_backend+dins_backend))

            #             recursive_check(self, primals, primres, self.tol)

            #             if fwdres is None:
            #                 fwdres = tangents
            #             else:
            #                 recursive_check(self, tangents, fwdres, self.tol)

            #             print(
            #                 name,
            #                 ",",
            #                 pname,
            #                 ",",
            #                 backend,
            #                 ",",
            #                 "Forward",
            #                 ",",
            #                 self.timeit(self.csv_filename, pname, "Forward", fwdstr, {'fwd': fwd_enzyme} | fwdins, self.count),
            #                 sep="\t",
            #             )

            # # assert fwdres is not None

            # revres = None

            # for pname, pipeline, pbackends in self.revfilter(self.AllPipelines):
            #     if backend in pbackends:

            #         adout = douts_backend
            #         if pipeline is not None:
            #             if self.mlirad_rev or pipeline is None:
            #                 rfn_enzyme = (
            #                     in_fn
            #                     if pipeline is None
            #                     else enzyme_jax_ir(
            #                         pipeline_options=pipeline,
            #                         argv=argv,
            #                         inner_jit=False,
            #                     )(in_fn)
            #                 )
            #                 rev_enzyme = jax.jit(
            #                     revtransform(rfn_enzyme),
            #                     # backend=backend
            #                 )

            #                 if self.revprimal:
            #                     primals, grads = rev_enzyme(adout, *ins_backend)
            #                 else:
            #                     grads = rev_enzyme(adout, *ins_backend)
            #                     assert grads is not None

            #                 if self.revprimal and primres is not None:
            #                     recursive_check(self, primals, primres, self.tol)

            #                 if revres is None:
            #                     revres = grads
            #                 else:
            #                     recursive_check(self, grads, revres, self.tol)

            #                 print(
            #                     name,
            #                     ",",
            #                     pname,
            #                     ",",
            #                     backend,
            #                     ",",
            #                     "PreRev",
            #                     ",",
            #                     self.timeit(self.csv_filename, pname, "PreRev", revstr, {'rev': rev_enzyme} | revins, self.count),
            #                     sep="\t",
            #                 )

            #             rfn_enzyme = in_fn
            #             rev_enzyme = jax.jit(
            #                 (
            #                     revtransform(rfn_enzyme)
            #                     if pipeline is None
            #                     else enzyme_jax_ir(
            #                         pipeline_options=pipeline,
            #                         argv=argv,
            #                         inner_jit=False,
            #                     )(revtransform(rfn_enzyme))
            #                 ),
            #                 # backend=backend
            #             )

            #             if self.revprimal:
            #                 primals, grads = rev_enzyme(adout, *ins_backend)
            #             else:
            #                 grads = rev_enzyme(adout, *ins_backend)
            #                 assert grads is not None

            #             if self.revprimal and primres is not None:
            #                 recursive_check(self, primals, primres, self.tol)

            #             if revres is None:
            #                 revres = grads
            #             else:
            #                 recursive_check(self, grads, revres, self.tol)

            #             print(
            #                 name,
            #                 ",",
            #                 pname,
            #                 ",",
            #                 backend,
            #                 ",",
            #                 "PostRev",
            #                 ",",
            #                 self.timeit(self.csv_filename, pname, "PostRev", revstr, {'rev': rev_enzyme} | revins, self.count),
            #                 sep="\t",
            #             )

            #         if pipeline is None or (pipeline.mlir_ad() and self.mlirad_rev):
            #             rfn_enzyme = (
            #                 in_fn
            #                 if pipeline is None
            #                 else enzyme_jax_ir(
            #                     pipeline_options=pipeline, argv=argv, inner_jit=False
            #                 )(in_fn)
            #             )
            #             rev_enzyme = jax.jit(
            #                 (
            #                     revtransform(rfn_enzyme)
            #                     if pipeline is None
            #                     else enzyme_jax_ir(
            #                         pipeline_options=pipeline,
            #                         argv=argv,
            #                         inner_jit=False,
            #                     )(revtransform(rfn_enzyme))
            #                 ),
            #                 # backend=backend
            #             )

            #             if self.revprimal:
            #                 primals, grads = rev_enzyme(adout, *ins_backend)
            #             else:
            #                 grads = rev_enzyme(adout, *ins_backend)
            #                 assert grads is not None

            #             if self.revprimal and primres is not None:
            #                 recursive_check(self, primals, primres, self.tol)

            #             if revres is None:
            #                 revres = grads
            #             else:
            #                 recursive_check(self, grads, revres, self.tol)

            #             print(
            #                 name,
            #                 ",",
            #                 pname,
            #                 ",",
            #                 backend,
            #                 ",",
            #                 "BothRev",
            #                 ",",
            #                 self.timeit(self.csv_filename, pname, "BothRev", revstr, {'rev': rev_enzyme} | revins, self.count),
            #                 sep="\t",
            #             )
