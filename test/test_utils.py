import os
import tempfile
import jax.numpy as jnp
import jax
from enzyme_ad.jax import (
    JaXPipeline,
    full_optimization_pass_pipeline,
)

from xprof_utils import profile_function


def fix_paths():
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
            "pypi_nvidia_cufft_cu12",
            "site-packages",
            "nvidia",
            "cufft",
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

    cupti_path = os.path.join(
        runfiles,
        "pypi_nvidia_cuda_cupti_cu12",
        "site-packages",
        "nvidia",
        "cuda_cupti",
        "lib",
        "libcupti.so.12",
    )

    if os.path.exists(cupti_path):
        import ctypes

        ctypes.cdll.LoadLibrary(cupti_path)

    cufft_path = os.path.join(
        runfiles,
        "pypi_nvidia_cufft_cu12",
        "site-packages",
        "nvidia",
        "cufft",
        "lib",
        "libcufft.so.11",
    )

    if os.path.exists(cufft_path):
        import ctypes

        ctypes.cdll.LoadLibrary(cufft_path)


from absl.testing import absltest  # noqa: E402

# import logging
# logging.getLogger("jax").setLevel(logging.INFO)
# import absl.logging
# absl.logging.set_verbosity(logging.INFO)

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
mul_simplify<16>;
div_simplify<16>;
rem_simplify<16>;
pow_simplify<16>;
noop_slice<16>;
const_prop_through_barrier<16>;
slice_slice<16>;
shift_right_logical_simplify<16>;
pad_simplify<16>(1024);
negative_pad_to_slice<16>;
slice_simplify<16>;
convert_simplify<16>;
dynamic_slice_to_static<16>;
dynamic_update_slice_elim<16>;
concat_to_broadcast<16>;
reduce_to_reshape<16>;
broadcast_to_reshape<16>;
iota_simplify<16>(1024);
broadcast_in_dim_simplify<16>(1024);
convert_concat<1>;
dynamic_update_to_concat<1>;
slice_of_dynamic_update<1>;
slice_elementwise<1>;
slice_pad<1>;
dot_reshape_dot<1>;
concat_const_prop<1>(1024);
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


def get_pipeline(name: str):
    if name == "JaxPipe":
        return ("JaXPipe", JaXPipeline(), CurBackends)
    elif name == "Jax":
        return ("Jax", None, CurBackends)
    elif name == "PartOpt":
        return ("PartOpt", JaXPipeline(partialopt), CurBackends)
    elif name == "HLOOpt":
        return (
            "HLOOpt",
            JaXPipeline(
                "inline{default-pipeline=canonicalize inlining-threshold=4294967295 max-iterations=4},"
                + "canonicalize,cse,enzyme-hlo-opt,cse"
            ),
            CurBackends,
        )
    elif name == "DefOpt":
        return (
            "DefOpt",
            JaXPipeline(full_optimization_pass_pipeline(inline=False)),
            CurBackends,
        )
    elif name == "IPartOpt":
        return (
            "IPartOpt",
            JaXPipeline(
                "inline{default-pipeline=canonicalize inlining-threshold=4294967295 max-iterations=4},"
                + partialopt
            ),
            CurBackends,
        )
    elif name == "IDefOpt":
        return (
            "IDefOpt",
            JaXPipeline(full_optimization_pass_pipeline()),
            CurBackends,
        )


def pipelines():
    setup_backends()

    return [
        get_pipeline("JaxPipe"),
        get_pipeline("Jax"),
        get_pipeline("HLOOpt"),
        get_pipeline("PartOpt"),
        get_pipeline("IPartOpt"),
        get_pipeline("DefOpt"),
        get_pipeline("IDefOpt"),
    ]


def no_newxla(x):
    return x


def no_newxlamlir(x):
    return x


def nomlir(x):
    return [(name, a, b) for (name, a, b) in x if name == "XLA"]


def justjax(x):
    from enzyme_ad.jax import JaXPipeline

    return [
        (name, a, b) for (name, a, b) in x if a is None or isinstance(a, JaXPipeline)
    ]


def splatjvp(in_fn):
    def fwd(*args):
        assert len(args) % 2 == 0
        return jax.jvp(
            in_fn, tuple(args[: len(args) // 2]), tuple(args[len(args) // 2 :])
        )

    return fwd


def sync(x):
    return jax.block_until_ready(x)


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


def to_backend(x, backend):
    dev = jax.local_devices(backend=backend)[0]
    return jax.device_put(x, dev)


def recursive_check(tester, lhs, rhs, pname=None):
    def leaves_allclose(leaf1, leaf2):
        legal = jnp.allclose(leaf1, leaf2, atol=tester.atol, rtol=tester.rtol)
        if not legal:
            max_err = jnp.max(jnp.abs(leaf1 - leaf2))
            if tester.skip_test_assert:
                print(
                    f"Skipping test assert for {tester.name} {pname} but test"
                    + f" failed with max error {max_err}."
                )
                return legal

            if pname is not None:
                print("lhs (", pname, ")", leaf1)
            else:
                print("lhs", leaf1)
            print("rhs", leaf2)
            print("abs", jnp.abs(leaf1 - leaf2))
            print("eq", jnp.abs(leaf1 - leaf2) < tester.atol)
            print("max", max_err)
        tester.assertTrue(legal)
        return legal

    comparison_tree = jax.tree.map(leaves_allclose, lhs, rhs)
    legal = jax.tree.all(comparison_tree)
    if not legal and tester.skip_test_assert:
        print(f"Skipping test assert for {tester.name} {pname}")
        return
    tester.assertTrue(legal)


def _dump_mlir_to_file(fn, args, key: str, dump_mlir_dir: str):
    loweredfn = fn.trace(*args).lower()
    source = loweredfn.as_text()

    # compiled_fn = loweredfn.compile()
    # print(compiled_fn.cost_analysis())

    # bazel will zip up the outputs in this directory
    env_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", None)
    if env_dir is not None:
        dump_mlir_dir = env_dir

    tmpfile = tempfile.NamedTemporaryFile(
        suffix=key.replace(" ", "") + ".mlir", dir=dump_mlir_dir, delete=False
    )
    with open(tmpfile.name, "w") as f:
        f.write(str(source))

    # print(f"Dumped mlir to {tmpfile.name}")
    return tmpfile.name


class EnzymeJaxTest(absltest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        setup_backends()
        self.primfilter = lambda x: x
        self.fwdfilter = lambda x: x
        self.revfilter = lambda x: x
        self.count = 1000
        self.repeat = 10
        self.AllBackends = AllBackends
        self.AllPipelines = pipelines()
        self.revprimal = True
        self.atol = 1e-6
        self.rtol = 0.0
        self.mlirad_fwd = True
        self.mlirad_rev = True
        self.results = []
        self.skip_test_assert = False

    def pretty_print_table(self, name, pname, backend, key, time):
        print_str = "{:<20}\t{:<20}\t{:<15}\t{:<10}\t{:<15.8f}".format(
            name, pname, backend, key, time
        )
        print(print_str, flush=True)

        result_str = "{}\t{}\t{}\t{}\t{}".format(name, pname, backend, key, time)
        self.results.append(result_str)

    def write_results_csv(self, filename=""):
        import os
        import csv

        if filename == "":
            filename = f"results_{self.__class__.__name__}.csv"

        outdir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", ".")
        outfile = os.path.join(outdir, filename)
        with open(outfile, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Benchmark Name", "Pass Pipeline", "Backend", "Key", "Time"]
            )
            for line in self.results:
                writer.writerow(line.split("\t"))
        print(f"Results written to {outfile}")

    def setUp(self):
        self.name = None

    def test(self):
        if self.name is None:
            return
        setup_backends()
        self.harness(self.name, self.fn, self.ins, self.dins, self.douts)
        self.write_results_csv()

    def harness(self, name, in_fn, ins, dins, douts):
        import timeit
        from enzyme_ad.jax import enzyme_jax_ir

        dump_mlir_dir = tempfile.gettempdir()

        if self.mlirad_fwd:
            assert len(ins) == len(dins)

        primalstr = (
            "sync(fn(" + (", ".join(["in" + str(i) for i in range(len(ins))])) + "))"
        )

        fwdstr = (
            "sync(fwd("
            + (", ".join(["in" + str(i) for i in range(len(ins))]))
            + ", "
            + (", ".join(["din" + str(i) for i in range(len(dins))]))
            + "))"
        )

        revstr = (
            "sync(rev(dout, "
            + (", ".join(["in" + str(i) for i in range(len(ins))]))
            + "))"
        )

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
                        )
                    )

                    _dump_mlir_to_file(
                        rfn_enzyme,
                        ins_backend,
                        pname + "_" + backend + "_primal",
                        dump_mlir_dir,
                    )

                    ao = rfn_enzyme(*ins_backend)

                    if primres is None:
                        primres = ao
                    else:
                        recursive_check(self, ao, primres, "Primal " + pname)

                    if backend == "cpu":  # xprof doesn't get correct info for cpu
                        runtime = (
                            min(
                                timeit.Timer(
                                    primalstr,
                                    globals={
                                        "fn": rfn_enzyme,
                                    }
                                    | primalins,
                                ).repeat(repeat=self.repeat, number=self.count)
                            )
                            / self.count
                        )
                    else:
                        runtime = profile_function(
                            rfn_enzyme, ins_backend, nrepeat=self.repeat
                        )["avg_time_s"]

                    self.pretty_print_table(name, pname, backend, "Primal", runtime)

            # assert primres is not None
            fwdres = None

            for pname, pipeline, pbackends in self.fwdfilter(self.AllPipelines):
                if backend in pbackends:
                    if self.mlirad_fwd or pipeline is None:
                        rfn_enzyme = (
                            in_fn
                            if pipeline is None
                            else jax.jit(
                                enzyme_jax_ir(
                                    pipeline_options=pipeline,
                                    argv=argv,
                                    inner_jit=False,
                                )(in_fn)
                            )
                        )
                        fwd_enzyme = jax.jit(splatjvp(rfn_enzyme))

                        _dump_mlir_to_file(
                            fwd_enzyme,
                            ins_backend + dins_backend,
                            pname + "_" + backend + "_forward",
                            dump_mlir_dir,
                        )

                        primals, tangents = fwd_enzyme(*(ins_backend + dins_backend))

                        if primres is None:
                            primres = primals
                        else:
                            recursive_check(self, primals, primres, "Primal " + pname)

                        if fwdres is None:
                            fwdres = tangents
                        else:
                            recursive_check(self, tangents, fwdres, "Forward " + pname)

                        if backend == "cpu":  # xprof doesn't get correct info for cpu
                            runtime = (
                                min(
                                    timeit.Timer(
                                        fwdstr, globals={"fwd": fwd_enzyme} | fwdins
                                    ).repeat(repeat=self.repeat, number=self.count)
                                )
                                / self.count
                            )
                        else:
                            runtime = profile_function(
                                fwd_enzyme,
                                ins_backend + dins_backend,
                                nrepeat=self.repeat,
                            )["avg_time_s"]

                        self.pretty_print_table(
                            name, pname, backend, "Forward", runtime
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
                                    pipeline_options=pipeline,
                                    argv=argv,
                                    inner_jit=False,
                                )(in_fn)
                            )
                            rev_enzyme = jax.jit(revtransform(rfn_enzyme))

                            _dump_mlir_to_file(
                                rev_enzyme,
                                [adout] + ins_backend,
                                pname + "_" + backend + "_prerev",
                                dump_mlir_dir,
                            )

                            if self.revprimal:
                                primals, grads = rev_enzyme(adout, *ins_backend)
                            else:
                                grads = rev_enzyme(adout, *ins_backend)
                                assert grads is not None

                            if self.revprimal and primres is not None:
                                recursive_check(
                                    self, primals, primres, "Primal " + pname
                                )

                            if revres is None:
                                revres = grads
                            else:
                                recursive_check(self, grads, revres, "Reverse " + pname)

                            if (
                                backend == "cpu"
                            ):  # xprof doesn't get correct info for cpu
                                runtime = (
                                    min(
                                        timeit.Timer(
                                            revstr, globals={"rev": rev_enzyme} | revins
                                        ).repeat(repeat=self.repeat, number=self.count)
                                    )
                                    / self.count
                                )
                            else:
                                runtime = profile_function(
                                    rev_enzyme,
                                    [adout] + ins_backend,
                                    nrepeat=self.repeat,
                                )["avg_time_s"]

                            self.pretty_print_table(
                                name, pname, backend, "PreRev", runtime
                            )

                        rfn_enzyme = in_fn
                        rev_enzyme = jax.jit(
                            (
                                revtransform(rfn_enzyme)
                                if pipeline is None
                                else enzyme_jax_ir(
                                    pipeline_options=pipeline,
                                    argv=argv,
                                    inner_jit=False,
                                )(revtransform(rfn_enzyme))
                            )
                        )

                        _dump_mlir_to_file(
                            rev_enzyme,
                            [adout] + ins_backend,
                            pname + "_" + backend + "_postrev",
                            dump_mlir_dir,
                        )

                        if self.revprimal:
                            primals, grads = rev_enzyme(adout, *ins_backend)
                        else:
                            grads = rev_enzyme(adout, *ins_backend)
                            assert grads is not None

                        if self.revprimal and primres is not None:
                            recursive_check(self, primals, primres)

                        if revres is None:
                            revres = grads
                        else:
                            recursive_check(self, grads, revres)

                        if backend == "cpu":  # xprof doesn't get correct info for cpu
                            runtime = (
                                min(
                                    timeit.Timer(
                                        revstr, globals={"rev": rev_enzyme} | revins
                                    ).repeat(repeat=self.repeat, number=self.count)
                                )
                                / self.count
                            )
                        else:
                            runtime = profile_function(
                                rev_enzyme, [adout] + ins_backend, nrepeat=self.repeat
                            )["avg_time_s"]

                        self.pretty_print_table(
                            name, pname, backend, "PostRev", runtime
                        )

                    if pipeline is None or (pipeline.mlir_ad() and self.mlirad_rev):
                        rfn_enzyme = (
                            in_fn
                            if pipeline is None
                            else enzyme_jax_ir(
                                pipeline_options=pipeline, argv=argv, inner_jit=False
                            )(in_fn)
                        )
                        rev_enzyme = jax.jit(
                            (
                                revtransform(rfn_enzyme)
                                if pipeline is None
                                else enzyme_jax_ir(
                                    pipeline_options=pipeline,
                                    argv=argv,
                                    inner_jit=False,
                                )(revtransform(rfn_enzyme))
                            )
                        )

                        _dump_mlir_to_file(
                            rev_enzyme,
                            [adout] + ins_backend,
                            pname + "_" + backend + "_bothrev",
                            dump_mlir_dir,
                        )

                        if self.revprimal:
                            primals, grads = rev_enzyme(adout, *ins_backend)
                        else:
                            grads = rev_enzyme(adout, *ins_backend)
                            assert grads is not None

                        if self.revprimal and primres is not None:
                            recursive_check(self, primals, primres)

                        if revres is None:
                            revres = grads
                        else:
                            recursive_check(self, grads, revres)

                        if backend == "cpu":  # xprof doesn't get correct info for cpu
                            runtime = (
                                min(
                                    timeit.Timer(
                                        revstr, globals={"rev": rev_enzyme} | revins
                                    ).repeat(repeat=self.repeat, number=self.count)
                                )
                                / self.count
                            )
                        else:
                            runtime = profile_function(
                                rev_enzyme, [adout] + ins_backend, nrepeat=self.repeat
                            )["avg_time_s"]

                        self.pretty_print_table(
                            name, pname, backend, "BothRev", runtime
                        )
