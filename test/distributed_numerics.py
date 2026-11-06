"""End-to-end numerics of the distributed compiler on small JAX workloads.

Each workload is exported to Shardy-annotated StableHLO, searched, finalized
and exported by distributed-to-stablehlo, then run on 16 XLA CPU devices from
global inputs and compared with the unsharded reference. Only numerical agreement is checked.

Sharding constraints alone do not force communication: the search binds
logical axes and the entry/result layouts freely. Each workload therefore
builds in a structural reason for one (a contraction over a sharded
dimension, a value used both as itself and transposed, ...), with enough
compute that sharding pays. Nothing asserts which collectives a winner uses,
since the search may legitimately choose another decomposition; each case
logs the kinds it exported.

Two cost settings on the same 16-device physical mesh reach different
winners: the default roofline favours gathers, slices and replication, while
cheap collectives favour permutes, reduce-scatters and all-reduces. Every
workload runs under both.
"""

import os
from typing import Callable, NamedTuple

import numpy as np
from absl.testing import absltest, parameterized

import jax
import jax.numpy as jnp
from jax.sharding import AbstractMesh, AxisType, NamedSharding, PartitionSpec

from distributed_test_utils import (
    NUM_DEVICES,
    PHYSICAL_MESH_16,
    PHYSICAL_MESH_16_CHEAP_COLLECTIVES,
    assert_all_close,
    build_export_pipeline_argv,
    build_reference_pipeline_argv,
    export_shardy_module,
    exported_collective_kinds,
    run_exported,
    run_reference,
    run_through_cli_pipeline,
)

jax.config.update("jax_use_shardy_partitioner", True)
# Must precede any use of a jax backend.
jax.config.update("jax_num_cpu_devices", NUM_DEVICES)

MESH = AbstractMesh((4, 4), ("a", "b"), axis_types=(AxisType.Auto,) * 2)
SEARCH_BEAM_SIZE = 10
TOLERANCE = 1e-4
N = 1024


def constrain(x, *spec):
    return jax.lax.with_sharding_constraint(
        x, NamedSharding(MESH, PartitionSpec(*spec))
    )


def tp_mlp(x, w1, w2):
    """One token through large weights: sharding the hidden dimension beats
    replicating the weights, and the second matmul then contracts over it
    (an all-reduce)."""
    h = jnp.tanh(x @ constrain(w1, None, "a") / np.sqrt(N))
    return (h @ constrain(w2, "a", None) / np.sqrt(4 * N),)


def softmax_both(x):
    """Softmax along each dimension in turn: each wants the other dimension
    local (a gather and a slice, or partial reductions)."""
    y = jax.nn.softmax(constrain(x, "a", None), axis=1)
    return (jax.nn.softmax(constrain(y, None, "a"), axis=0),)


def outer_product(x):
    """`y @ y.T` reads `y` both as rows and as columns of the result (a local
    slice of a replicated copy, or a permute)."""
    y = constrain(jnp.tanh(constrain(x, ("a", "b"), None)), ("a", "b"), None)
    return (y @ y.T / np.sqrt(y.shape[1]),)


def transpose_add(x, w):
    """A 2-D tiled matmul result added to its own transpose: tile (i, j)
    needs tile (j, i) (a permute)."""
    y = jnp.tanh(x @ w / np.sqrt(N))
    return (y + y.T,)


def take_after_matmul(x, w, idx):
    """A gather reads the whole matmul result, so the contraction's partial
    sums must meet (a reduce-scatter and the collectives around it)."""
    y = constrain(jnp.tanh(x @ w / np.sqrt(w.shape[0])), ("a", "b"))
    return (jnp.take(y, idx),)


def strided(x):
    """The iota's dimension is unshardable, so only the minor digit of `x`'s
    `4 * M` dimension can be sharded: a strided entry boundary, which the
    export relays out in its global entry."""
    y = jnp.reshape(constrain(x, "a"), (4, -1))
    y = y + jax.lax.broadcasted_iota(jnp.float32, y.shape, 0)
    return (jnp.tanh(y),)


class Workload(NamedTuple):
    fn: Callable
    # One (shape, dtype) per argument; an integer argument is an index into
    # an array of length `index_bound`.
    arguments: list
    index_bound: int = 0


WORKLOADS = {
    "tp_mlp": Workload(
        tp_mlp,
        [((1, N), jnp.float32), ((N, 4 * N), jnp.float32), ((4 * N, N), jnp.float32)],
    ),
    "softmax_both": Workload(softmax_both, [((N, N), jnp.float32)]),
    "outer_product": Workload(outer_product, [((N, 256), jnp.float32)]),
    "transpose_add": Workload(
        transpose_add, [((N, N), jnp.float32), ((N, N), jnp.float32)]
    ),
    "take_after_matmul": Workload(
        take_after_matmul,
        [((16 * N, 256), jnp.float32), ((256,), jnp.float32), ((64,), jnp.int32)],
        index_bound=16 * N,
    ),
    "strided": Workload(strided, [((64 * N,), jnp.float32)]),
}

MESHES = {
    "default_costs": PHYSICAL_MESH_16,
    "cheap_collectives": PHYSICAL_MESH_16_CHEAP_COLLECTIVES,
}


def concrete_inputs(workload, seed=0):
    rng = np.random.default_rng(seed)
    inputs = []
    for shape, dtype in workload.arguments:
        if jnp.issubdtype(dtype, jnp.integer):
            inputs.append(rng.integers(0, workload.index_bound, shape).astype(dtype))
        else:
            inputs.append(rng.standard_normal(shape).astype(dtype))
    return inputs


class DistributedNumericsTest(parameterized.TestCase):
    @parameterized.named_parameters(
        (f"{workload}_{mesh}", workload, mesh)
        for workload in WORKLOADS
        for mesh in MESHES
    )
    def test_matches_unsharded_reference(self, workload_name, mesh_name):
        workload = WORKLOADS[workload_name]
        avals = [
            jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in workload.arguments
        ]
        text = export_shardy_module(workload.fn, *avals)
        dump_dir = os.environ.get("DISTRIBUTED_NUMERICS_DUMP_DIR")

        export = run_through_cli_pipeline(
            text,
            lambda path: build_export_pipeline_argv(path, SEARCH_BEAM_SIZE),
            MESHES[mesh_name],
            dump_dir=dump_dir,
            dump_name=f"{workload_name}_{mesh_name}.export.mlir",
        )
        kinds = exported_collective_kinds(export)
        print(f"{workload_name} ({mesh_name}): exported {kinds or 'no communication'}")

        inputs = concrete_inputs(workload)
        expected = run_reference(
            run_through_cli_pipeline(
                text, build_reference_pipeline_argv, MESHES[mesh_name]
            ),
            inputs,
        )
        actual = run_exported(export, inputs)
        assert_all_close(
            actual, expected, TOLERANCE, TOLERANCE, context=f"Exported {kinds}."
        )


if __name__ == "__main__":
    from test_utils import fix_paths

    fix_paths()
    absltest.main()
