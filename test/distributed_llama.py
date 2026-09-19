"""Exports a llama-style transformer forward pass (a standalone copy of
llama.py's forward(), not an import -- sharding annotations are easiest to
read/maintain sitting directly next to the tensors they describe) as
Shardy-annotated StableHLO, then runs it through the whole distributed
pipeline -- propagation, distributed conversion, physical mesh insertion, the
sharding/scheduling search (a small beam, just enough to exercise it), and
kernel-to-executable lowering -- and checks that it compiles without
crashing. This is a smoke test: it doesn't check the result is correct, just
that this more realistic model makes it through the pipeline in one piece.

Set DISTRIBUTED_LLAMA_DUMP_DIR to also write out the intermediate/final
modules for manual inspection, in particular the per-kernel modules under
kernel_modules/ -- the standalone modules the external compiler will
eventually be dispatched on (see LowerKernelsToExecutable.cpp).

The pipeline itself runs as a subprocess of the enzymexlamlir-opt binary
(the same tool the lit tests and tmp.py drive), rather than in-process via
enzyme_call.run_pass_pipeline: that in-process path turned out to need two
things the CLI driver's main() sets up for itself and enzyme_call.cc's
bridge doesn't -- registering the Distributed/Shardy passes and pipelines
(enzymexlamlir-opt.cpp's initializePasses()) and constructing its
PassManager with implicit op nesting (MlirOptMain.cpp) so a pipeline built
for the CLI can add a func::FuncOp-scoped pass directly to a
ModuleOp-anchored pass manager. Reusing the already-registered, already-
nesting-correct binary sidesteps both instead of duplicating that setup in
the Python bridge.
"""

import os
import subprocess
import tempfile

from absl.testing import absltest

import jax
import jax.numpy as jnp
from jax.sharding import AbstractMesh, NamedSharding, PartitionSpec

jax.config.update("jax_use_shardy_partitioner", True)

# ---------------------------------------------------------------------------
# Model config. Mirrors llama.py's default Llama config, except:
#  - wcls/vocab_size dropped: forward() never actually uses wcls in its
#    returned computation (the "logits = wcls @ x" line is dead/commented out
#    upstream too).
#  - pos (KV cache length) bumped from 1 to 64, and a batch dimension added,
#    so the "cp" and "data" axes below have something non-trivial to shard.
# ---------------------------------------------------------------------------
DIM = 288
HIDDEN_DIM = 768
N_LAYERS = 6
N_HEADS = 6
N_KV_HEADS = 6
HEAD_SIZE = DIM // N_HEADS
KV_DIM = DIM // N_HEADS * N_KV_HEADS
KV_MUL = N_HEADS // N_KV_HEADS

POS = 64  # KV cache length so far (excludes the new token being appended)
BATCH = 8

# ---------------------------------------------------------------------------
# Mesh / sharding plan selection. "tp" and "fsdp" are two different strategies
# for sharding the *same* weight tensors and are never combined in one plan
# (see shard_weight()). Pipeline parallelism is deliberately not modeled here:
# it requires different devices to run genuinely different program stages,
# which doesn't fit Shardy's single-program-multi-data model.
#
# Axis sizes must evenly divide the dimension they shard: "tp"/"fsdp" against
# DIM/HIDDEN_DIM/N_LAYERS, "cp" against POS, "data" against BATCH.
# ---------------------------------------------------------------------------
PLANS = {
    "tp4": {"tp": 4},
    "tp8": {"tp": 8},
    "tp4_cp2": {"tp": 4, "cp": 2},
    "dp2_tp4": {"data": 2, "tp": 4},
    "fsdp3": {"fsdp": 3},
    "fsdp6": {"fsdp": 6},
}
SELECTED_PLAN = "tp4"

_plan = PLANS[SELECTED_PLAN]
AXIS_NAMES = tuple(_plan.keys())
AXIS_SIZES = tuple(_plan.values())
MESH = AbstractMesh(AXIS_SIZES, AXIS_NAMES)

# A minimal physical-device-topology config for --insert-physical-mesh. This
# is unrelated to the logical sdy.mesh above -- InsertPhysicalMesh only maps
# logical axes onto a physical comm-axis topology, and its axis counts don't
# need to match the logical mesh's (see config_mesh_4x4x8.mlir, which pairs a
# 128-device physical mesh with hand-written tests using a 32-device logical
# one).
PHYSICAL_MESH_CONFIG = (
    'distributed.PhysicalMesh @mesh448 device_target "cpu" axes '
    "[!distributed.physical_comm_axis<4, 32>,"
    "!distributed.physical_comm_axis<4, 8>, "
    "!distributed.physical_comm_axis<8, 1>]"
)

# Small beam: this is a smoke test for "does it crash", not a check on the
# quality of the chosen sharding/scheduling decisions.
SEARCH_BEAM_SIZE = 2


def find_enzymexlamlir_opt():
    """Locates the enzymexlamlir-opt binary, whether run as a bazel test (where
    it's a data dependency staged into runfiles) or invoked directly from the
    repo root the way tmp.py does (against a manually-built bazel-bin/)."""
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


def build_pipeline_argv(physical_mesh_config_path, kernel_modules_dir=""):
    args = [
        "--sdy-propagation-pipeline",
        "--shardy-to-distributed-pipeline",
        "--cse",
        "--canonicalize",
        f"--insert-physical-mesh=configuration-file={physical_mesh_config_path}",
        f"--distributed-search-strategies=beam-size={SEARCH_BEAM_SIZE}",
        "--cse",
        "--canonicalize",
        "--stabilize-axis-order",
    ]
    if kernel_modules_dir:
        args.append(
            f"--distributed-lower-kernels-to-executable=dump-kernel-modules-to={kernel_modules_dir}"
        )
    else:
        args.append("--distributed-lower-kernels-to-executable")
    return args


def maybe_constrain(tensor, dim_to_axis):
    """Attach a sharding constraint for whichever (dim -> axis name) entries name an
    axis present in the active mesh; leaves everything else replicated. No-ops
    entirely if none of the requested axes are part of the current plan, so the same
    call sites work unmodified across every PLANS entry."""
    spec = [None] * tensor.ndim
    used = False
    for dim, axis in dim_to_axis.items():
        if axis in MESH.axis_names:
            spec[dim] = axis
            used = True
    if not used:
        return tensor
    return jax.lax.with_sharding_constraint(tensor, NamedSharding(MESH, PartitionSpec(*spec)))


def shard_weight(w, tp_out_dim):
    """Megatron-style TP shards the matmul's output dim (tp_out_dim) so propagation
    naturally produces the right collective: no communication needed going into a
    column-parallel op, an all-reduce after a row-parallel one. FSDP instead shards
    axis 0 (the per-layer n_layers axis, orthogonal to the matmul itself) so every
    device still runs the identical program/layer loop but must all-gather that
    layer's full weight shard on demand -- a distinct, gather-before-compute pattern
    from TP's compute-then-reduce, unlike pipeline parallelism this stays SPMD."""
    return maybe_constrain(w, {0: "fsdp", tp_out_dim: "tp"})


def rmsnorm(x, weight):
    ss = 1 / jnp.sqrt(x.dot(x) / x.shape[0] + 1e-5)
    return weight * x * ss


def softmax(x, axis=-1):
    max_val = jnp.max(x, axis=axis, keepdims=True)
    x = jnp.exp(x - max_val)
    return x / jnp.sum(x, axis=axis, keepdims=True)


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def silu(x):
    return x * sigmoid(x)


def forward(x, weights, key_cache, value_cache):
    """Single-example decode step. Weight tensors are shape (N_LAYERS, out_dim,
    in_dim) so that weights[i, :, :] @ activation matches jnp's (m, n) @ (n,) -> (m,)
    convention: axis 1 is always the matmul's output/kept dim, axis 2 the
    contracted/input dim -- that's what shard_weight()'s tp_out_dim picks between."""
    pos = key_cache.shape[1]

    wq = shard_weight(weights["wq"], tp_out_dim=1)
    wk = shard_weight(weights["wk"], tp_out_dim=1)
    wv = shard_weight(weights["wv"], tp_out_dim=1)
    wo = shard_weight(weights["wo"], tp_out_dim=2)
    w1 = shard_weight(weights["w1"], tp_out_dim=1)
    w2 = shard_weight(weights["w2"], tp_out_dim=2)
    w3 = shard_weight(weights["w3"], tp_out_dim=1)
    rms_att_weight = weights["rms_att_weight"]
    rms_ffn_weight = weights["rms_ffn_weight"]
    rms_final_weight = weights["rms_final_weight"]

    key_cache = maybe_constrain(key_cache, {1: "cp"})
    value_cache = maybe_constrain(value_cache, {1: "cp"})

    toconv = []
    for i in range(0, DIM, 2):
        freq = 1 / jnp.power(10000, (i % HEAD_SIZE) / HEAD_SIZE)
        val = pos * freq
        fcr = jnp.cos(val)
        fci = jnp.sin(val)
        rotM = jnp.array([[fcr, -fci], [fci, fcr]])
        toconv.append(rotM)
    toconv2 = toconv[: KV_DIM // 2] + [jnp.eye(2)] * (DIM // 2 - KV_DIM // 2)
    toconv = jnp.array(toconv)
    toconv2 = jnp.array(toconv2)

    for i in range(N_LAYERS):
        xb = rmsnorm(x, rms_att_weight[i, :])

        q = wq[i, :, :] @ xb
        k = wk[i, :, :] @ xb
        v = wv[i, :, :] @ xb

        q_tmp = jnp.reshape(q, (DIM // 2, 2))
        k_tmp = jnp.reshape(k, (DIM // 2, 2))

        k = jnp.reshape(jnp.einsum("ijk,ik -> ij", toconv2, k_tmp), (DIM,))
        q = jnp.reshape(jnp.einsum("ijk,ik -> ij", toconv, q_tmp), (DIM,))

        # jnp.append lowers to a separate outlined func.call wrapping a plain
        # stablehlo.concatenate, which the distributed pipeline has no sharding
        # rule for; jnp.concatenate produces the same op inlined.
        key_cache_l = key_cache[i, :, :]
        key_cache_l = jnp.concatenate([key_cache_l, jnp.reshape(k, (1, DIM))], axis=0)
        value_cache_l = value_cache[i, :, :]
        value_cache_l = jnp.concatenate(
            [value_cache_l, jnp.reshape(v, (1, DIM))], axis=0
        )

        # Multi-head attention over a real "head" dimension via einsum,
        # instead of Python-unrolled static per-head slices: a static slice
        # gets a Shardy "permutation" sharding-rule factor, which entangles
        # the TP-sharded head-split axis with whatever else that factor
        # touches. A reshape splitting DIM into (N_HEADS, HEAD_SIZE) is an
        # ordinary pass-through factor instead, so the TP axis stays cleanly
        # shardable across heads.
        q_heads = jnp.reshape(q, (N_HEADS, HEAD_SIZE))
        key_cache_heads = jnp.reshape(key_cache_l, (pos + 1, N_KV_HEADS, HEAD_SIZE))
        value_cache_heads = jnp.reshape(
            value_cache_l, (pos + 1, N_KV_HEADS, HEAD_SIZE)
        )
        if KV_MUL > 1:
            key_cache_heads = jnp.repeat(key_cache_heads, KV_MUL, axis=1)
            value_cache_heads = jnp.repeat(value_cache_heads, KV_MUL, axis=1)

        att = jnp.einsum("phd,hd->hp", key_cache_heads, q_heads)
        att = att / jnp.sqrt(HEAD_SIZE)
        att = softmax(att, axis=-1)

        xb = jnp.einsum("phd,hp->hd", value_cache_heads, att)
        xb = jnp.reshape(xb, (DIM,))

        xb2 = wo[i, :, :] @ xb
        x = x + xb2

        xb = rmsnorm(x, rms_ffn_weight[i, :])

        hb = w1[i, :, :] @ xb
        hb2 = w3[i, :, :] @ xb
        hb = silu(hb)
        hb = hb * hb2
        xb = w2[i, :, :] @ hb

        x = x + xb

    x = rmsnorm(x, rms_final_weight)
    return x


def forward_batched(x, weights, key_cache, value_cache):
    """DP annotation lives here, not inside forward(): the batch axis doesn't exist
    from forward()'s (per-example) point of view, so it has to be constrained on the
    batched arguments before vmapping over them."""
    x = maybe_constrain(x, {0: "data"})
    key_cache = maybe_constrain(key_cache, {0: "data"})
    value_cache = maybe_constrain(value_cache, {0: "data"})
    return jax.vmap(forward, in_axes=(0, None, 0, 0))(x, weights, key_cache, value_cache)


def export_shardy_module():
    """Traces/lowers forward_batched() and returns its Shardy-annotated StableHLO
    text. Set DISTRIBUTED_LLAMA_DUMP_DIR to also write it to disk for inspection."""
    weight_shapes = {
        "wq": (N_LAYERS, DIM, DIM),
        "wk": (N_LAYERS, KV_DIM, DIM),
        "wv": (N_LAYERS, KV_DIM, DIM),
        "wo": (N_LAYERS, DIM, DIM),
        "w1": (N_LAYERS, HIDDEN_DIM, DIM),
        "w2": (N_LAYERS, DIM, HIDDEN_DIM),
        "w3": (N_LAYERS, HIDDEN_DIM, DIM),
        "rms_att_weight": (N_LAYERS, DIM),
        "rms_ffn_weight": (N_LAYERS, DIM),
        "rms_final_weight": (DIM,),
    }
    weights_aval = {
        name: jax.ShapeDtypeStruct(shape, jnp.float32)
        for name, shape in weight_shapes.items()
    }
    x_aval = jax.ShapeDtypeStruct((BATCH, DIM), jnp.float32)
    key_cache_aval = jax.ShapeDtypeStruct((BATCH, N_LAYERS, POS, KV_DIM), jnp.float32)
    value_cache_aval = jax.ShapeDtypeStruct((BATCH, N_LAYERS, POS, KV_DIM), jnp.float32)

    lowered = jax.jit(forward_batched).lower(
        x_aval, weights_aval, key_cache_aval, value_cache_aval
    )
    module = lowered.compiler_ir(dialect="stablehlo")
    text = module.operation.get_asm(enable_debug_info=False)

    dump_dir = os.environ.get("DISTRIBUTED_LLAMA_DUMP_DIR")
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        with open(os.path.join(dump_dir, "llama_shardy_export.mlir"), "w") as f:
            f.write(text)

    return text


class DistributedLlamaTest(absltest.TestCase):
    def test_compiles_without_crashing(self):
        text = export_shardy_module()
        self.assertIn("sdy.mesh", text)
        self.assertIn("sdy.sharding", text)

        dump_dir = os.environ.get("DISTRIBUTED_LLAMA_DUMP_DIR")
        kernel_modules_dir = ""
        if dump_dir:
            kernel_modules_dir = os.path.join(dump_dir, "kernel_modules")
            os.makedirs(kernel_modules_dir, exist_ok=True)

        opt_binary = find_enzymexlamlir_opt()
        with tempfile.TemporaryDirectory() as scratch_dir:
            mesh_config_path = os.path.join(scratch_dir, "physical_mesh.mlir")
            with open(mesh_config_path, "w") as f:
                f.write(PHYSICAL_MESH_CONFIG)

            input_path = os.path.join(scratch_dir, "llama_shardy_export.mlir")
            with open(input_path, "w") as f:
                f.write(text)

            output_path = os.path.join(scratch_dir, "llama_distributed_out.mlir")
            argv = (
                [opt_binary]
                + build_pipeline_argv(mesh_config_path, kernel_modules_dir)
                + [input_path, "-o", output_path]
            )
            result = subprocess.run(argv, capture_output=True, text=True)
            self.assertEqual(
                result.returncode,
                0,
                f"enzymexlamlir-opt failed (exit {result.returncode}):\n"
                f"{result.stderr}",
            )

            if dump_dir:
                with open(output_path) as f:
                    out_text = f.read()
                with open(
                    os.path.join(dump_dir, "llama_distributed_out.mlir"), "w"
                ) as f:
                    f.write(out_text)


if __name__ == "__main__":
    from test_utils import fix_paths

    fix_paths()
    absltest.main()
