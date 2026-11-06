"""Helpers shared by the distributed pipeline tests: exporting a JAX program
to Shardy-annotated StableHLO, running it through enzymexlamlir-opt, and
executing both the distributed export and an unsharded reference on XLA CPU
devices for a numerical comparison.

The pipeline runs as a subprocess of the enzymexlamlir-opt binary (the same
tool the lit tests drive) rather than in-process via
enzyme_call.run_pass_pipeline. The in-process path lacks two things the CLI
driver's main() sets up: registration of the Distributed/Shardy passes and
pipelines (enzymexlamlir-opt.cpp's initializePasses()), and a PassManager
with implicit op nesting (MlirOptMain.cpp), which lets a CLI pipeline add a
func::FuncOp-scoped pass directly to a ModuleOp-anchored pass manager.
"""

import os
import re
import subprocess
import tempfile

import numpy as np

import jax
from jax._src import compiler
from jax._src.interpreters import mlir
from jax._src.lib import xla_client
from jax._src.lib.mlir import ir
from jax._src.sharding_impls import GSPMDSharding

from enzyme_ad.jax import hlo_call

# The numerics tests compile the distributed export for this many XLA CPU
# devices. A test module must set jax_num_cpu_devices to it before jax
# initializes its backends.
NUM_DEVICES = 16

# A 16-device physical mesh for --insert-physical-mesh: two axes of 4, the
# first major (stride 4).
PHYSICAL_MESH_16 = (
    'distributed.PhysicalMesh @mesh16 device_target "cpu" axes '
    "[!distributed.physical_comm_axis<4, 4>, "
    "!distributed.physical_comm_axis<4, 1>]"
)

# The same mesh with CPU-like costs (1 TFLOP/s, 100 GB/s memory, 10 GB/s
# links, 1 us latencies), under which communication pays at test sizes far
# more often than under the default roofline.
PHYSICAL_MESH_16_CHEAP_COLLECTIVES = PHYSICAL_MESH_16 + (
    " {axis_bandwidths = [1.0e10, 1.0e10], axis_latencies = [1.0e-6, 1.0e-6], "
    "launch_latency = 1.0e-6, device_flops = 1.0e12, device_mem_bandwidth = 1.0e11}"
)

# The StableHLO ops the export's per-device program communicates with. Local
# slices are dynamic_slices indexed by partition_id; they are counted
# separately (see exported_collective_kinds).
COLLECTIVE_OPS = (
    "all_reduce",
    "all_gather",
    "all_to_all",
    "collective_permute",
    "reduce_scatter",
)


def find_enzymexlamlir_opt():
    """Locates the enzymexlamlir-opt binary, whether run as a bazel test (a
    data dependency staged into runfiles) or from the repo root against a
    manually built bazel-bin/."""
    override = os.environ.get("ENZYMEXLAMLIR_OPT")
    if override:
        return override
    candidates = [
        os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "__main__",
            "enzymexlamlir-opt",
        ),
        os.path.join(os.getcwd(), "bazel-bin", "enzymexlamlir-opt"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Could not locate the enzymexlamlir-opt binary; set ENZYMEXLAMLIR_OPT "
        f"explicitly. Tried: {candidates}"
    )


def export_shardy_module(fn, *avals):
    """Traces `fn` on `avals` and returns its Shardy-annotated StableHLO text."""
    lowered = jax.jit(fn).lower(*avals)
    module = lowered.compiler_ir(dialect="stablehlo")
    return module.operation.get_asm(enable_debug_info=False)


def search_pipeline_prefix(physical_mesh_config_path, beam_size):
    """Propagation, conversion to the distributed dialect and the
    sharding/scheduling search, up to the winner's lowering."""
    return [
        # The search runs its lowering pipeline once per scored candidate and
        # relies on op verification/remarks to reject bad candidates. MLIR's
        # default of attaching a full op dump to every diagnostic makes that
        # cost scale with IR size per candidate, which dominates runtime once
        # the beam explores more than a handful of them.
        "--mlir-print-op-on-diagnostic=false",
        "--sdy-propagation-pipeline",
        "--shardy-to-distributed-pipeline",
        "--cse",
        "--canonicalize",
        f"--insert-physical-mesh=configuration-file={physical_mesh_config_path}",
        f"--distributed-search-strategies=beam-size={beam_size}",
        "--cse",
        "--canonicalize",
        "--stabilize-axis-order",
    ]


def build_export_pipeline_argv(physical_mesh_config_path, beam_size):
    """The search, then the winner's finalized lowering and its export as a
    global-shaped StableHLO entry that XLA compiles with SPMD partitioning."""
    return search_pipeline_prefix(physical_mesh_config_path, beam_size) + [
        "--distributed-search-lowering-pipeline=finalize-schedule=true",
        "--distributed-to-stablehlo",
    ]


def build_reference_pipeline_argv(_physical_mesh_config_path=None):
    """The unsharded reference: the same Shardy export, resolved by Shardy's
    own propagation and the standard XLA export pipeline instead of the
    distributed pipeline. Its mhlo.sharding attributes are partitioning hints
    only (see strip_mhlo_sharding_attrs), so executed on one partition it
    computes the program's values from the same model and annotations rather
    than a separately maintained copy of the math. Takes (and ignores) a
    physical-mesh config path, for a uniform call site in
    run_through_cli_pipeline."""
    return [
        "--mlir-print-op-on-diagnostic=false",
        "--sdy-propagation-pipeline",
        # keep-hlo-sharding-constraints=true exports sharding constraints as
        # stablehlo.custom_call @Sharding (a plain identity op at
        # num_partitions=1) rather than the default mhlo.copy, which
        # hlo_call's stablehlo-only legalization can't handle.
        "--xla-sdy-stablehlo-export-pipeline=keep-hlo-sharding-constraints=true",
        "--canonicalize",
        "--cse",
    ]


_MHLO_SHARDING_ATTR_RE = re.compile(r'\s*\{mhlo\.sharding = "[^"]*"\}')


def strip_mhlo_sharding_attrs(text):
    """Removes the {mhlo.sharding = "..."} argument/result attributes the
    reference pipeline leaves behind. The PJRT client reads them and falls
    back to real GSPMD partitioning when it sees one, which for a mesh larger
    than one device is a pointless multi-way compilation. hlo_call executes
    at num_partitions=1, so the reference must not carry them."""
    return _MHLO_SHARDING_ATTR_RE.sub("", text)


def run_through_cli_pipeline(
    text, argv_builder, physical_mesh_config, dump_dir=None, dump_name=None
):
    """Runs `text` through enzymexlamlir-opt with the argv
    `argv_builder(mesh_config_path)` builds, after writing
    `physical_mesh_config` to `mesh_config_path`, and returns the resulting
    module text. With `dump_dir` and `dump_name`, also writes the result
    there."""
    opt_binary = find_enzymexlamlir_opt()
    with tempfile.TemporaryDirectory() as scratch_dir:
        mesh_config_path = os.path.join(scratch_dir, "physical_mesh.mlir")
        with open(mesh_config_path, "w") as f:
            f.write(physical_mesh_config)

        input_path = os.path.join(scratch_dir, "input.mlir")
        with open(input_path, "w") as f:
            f.write(text)

        output_path = os.path.join(scratch_dir, "output.mlir")
        argv = (
            [opt_binary]
            + argv_builder(mesh_config_path)
            + [input_path, "-o", output_path]
        )
        result = subprocess.run(argv, capture_output=True, text=True)
        if result.returncode != 0:
            raise AssertionError(
                f"enzymexlamlir-opt failed (exit {result.returncode}):\n{result.stderr}"
            )

        with open(output_path) as f:
            out_text = f.read()

    if dump_dir and dump_name:
        os.makedirs(dump_dir, exist_ok=True)
        with open(os.path.join(dump_dir, dump_name), "w") as f:
            f.write(out_text)
    return out_text


def run_reference(reference_text, flat_inputs):
    """Executes the reference pipeline's output on one CPU device.

    The reference runs on CPU like the export: on a GPU, XLA's default f32
    dot uses TF32, which alone misses the tolerance on these workloads."""
    reference_text = strip_mhlo_sharding_attrs(reference_text)
    reference_fn = jax.jit(
        lambda *args: hlo_call(*args, source=reference_text, passes="")
    )
    with jax.default_device(jax.devices("cpu")[0]):
        return [np.asarray(x) for x in reference_fn(*flat_inputs)]


def run_exported(export_text, flat_inputs, num_devices=NUM_DEVICES):
    """Compiles a distributed export with SPMD partitioning over
    `num_devices` CPU devices, places each global input with the sharding the
    executable expects for it, and returns the global results."""
    devices = jax.devices("cpu")[:num_devices]
    assert len(devices) == num_devices, (
        f"need {num_devices} CPU devices, have {len(devices)}; set "
        "jax_num_cpu_devices before jax initializes"
    )
    device_list = xla_client.DeviceList(tuple(devices))
    options = compiler.get_compile_options(
        num_replicas=1,
        num_partitions=num_devices,
        device_assignment=np.array(devices).reshape(1, num_devices),
    )
    options.executable_build_options.use_spmd_partitioning = True
    # The export carries mhlo.sharding attributes, which GSPMD reads.
    options.executable_build_options.use_shardy_partitioner = False
    executable = devices[0].client.compile_and_load(export_text, device_list, options)

    def sharding(proto):
        return GSPMDSharding(tuple(devices), xla_client.HloSharding.from_proto(proto))

    inputs = [
        jax.device_put(np.asarray(value), sharding(proto))
        for value, proto in zip(
            flat_inputs, executable.get_parameter_shardings(), strict=True
        )
    ]
    shards = executable.execute_sharded(inputs).disassemble_into_single_device_arrays()
    return [
        np.asarray(
            jax.make_array_from_single_device_arrays(shape, sharding(proto), per_device)
        )
        for shape, per_device, proto in zip(
            _main_result_shapes(export_text),
            shards,
            executable.get_output_shardings(),
            strict=True,
        )
    ]


def _main_result_shapes(export_text):
    """The global result shapes of the export's @main."""
    with mlir.make_ir_context():
        module = ir.Module.parse(export_text)
        main = next(
            op
            for op in module.body.operations
            if "sym_name" in op.attributes
            and ir.StringAttr(op.attributes["sym_name"]).value == "main"
        )
        signature = ir.FunctionType(ir.TypeAttr(main.attributes["function_type"]).value)
        return [tuple(ir.RankedTensorType(t).shape) for t in signature.results]


def exported_collective_kinds(export_text):
    """Counts the communication ops in a distributed export by kind, for
    logging which patterns a search winner used."""
    counts = {
        kind: len(re.findall(rf"\bstablehlo\.{kind}\b", export_text))
        for kind in COLLECTIVE_OPS
    }
    # Approximate: every dynamic_slice counts once the program reads
    # partition_id, including any a kernel uses on its own.
    if "stablehlo.partition_id" in export_text:
        counts["local_slice"] = len(
            re.findall(r"\bstablehlo\.dynamic_slice\b", export_text)
        )
    return {kind: count for kind, count in counts.items() if count}


def assert_all_close(actual, expected, rtol, atol, context=""):
    """Compares lists of results elementwise and, on failure, reports each
    mismatching result's max absolute and relative error against the
    tolerance, plus `context` (e.g. the exported collective kinds)."""
    assert len(actual) == len(
        expected
    ), f"{len(actual)} results, expected {len(expected)}. {context}"
    failures = []
    for index, (got, want) in enumerate(zip(actual, expected)):
        if got.shape != want.shape:
            failures.append(f"result {index}: shape {got.shape}, expected {want.shape}")
            continue
        if np.allclose(got, want, rtol=rtol, atol=atol):
            continue
        error = np.abs(got.astype(np.float64) - want.astype(np.float64))
        relative = error / np.maximum(np.abs(want.astype(np.float64)), 1e-30)
        failures.append(
            f"result {index}: max abs error {error.max():.3g}, max rel error "
            f"{relative.max():.3g} (rtol={rtol}, atol={atol})"
        )
    if failures:
        raise AssertionError("; ".join(failures) + f". {context}")
