"""Utility functions for extracting profiling data from xprof/XLA traces."""

import json
import os
import tempfile
import glob
from typing import Callable, Any

import jax


def profile_function(
    fn: Callable,
    args: tuple = (),
    kwargs: dict | None = None,
    nrepeat: int = 1,
    warmup: int = 1,
    trace_dir: str | None = None,
) -> dict[str, Any]:
    """
    Profile a JAX function and return timing data.

    Args:
        fn: The function to profile (should be a compiled JAX function)
        args: Positional arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        nrepeat: Number of times to run the function during profiling
        warmup: Number of warmup runs before profiling (to ensure compilation is done)
        trace_dir: Directory to save traces. If None, uses a temporary directory.

    Returns:
        A dictionary containing:
            - 'total_time_us': Total runtime in microseconds
            - 'total_time_ms': Total runtime in milliseconds
            - 'avg_time_us': Average runtime per iteration in microseconds
            - 'avg_time_ms': Average runtime per iteration in milliseconds
            - 'operations': List of dicts with operation details
            - 'nrepeat': Number of repetitions
            - 'trace_dir': Directory where traces were saved
            - 'xplane_file': Path to the xplane.pb file
    """
    if kwargs is None:
        kwargs = {}

    # Warmup runs (ensure compilation is complete)
    for _ in range(warmup):
        jax.block_until_ready(fn(*args, **kwargs))

    # Create trace directory
    if trace_dir is None:
        trace_dir = tempfile.mkdtemp(prefix="jax_profile_")

    # Profile the function
    with jax.profiler.trace(trace_dir):
        for _ in range(nrepeat):
            jax.block_until_ready(fn(*args, **kwargs))

    # Find the xplane.pb file
    xplane_pattern = os.path.join(trace_dir, "plugins/profile/*/*.xplane.pb")
    xplane_files = glob.glob(xplane_pattern)

    if not xplane_files:
        raise RuntimeError(
            f"No xplane.pb file found in {trace_dir}. "
            "Make sure the function executed on a device."
        )

    xplane_file = xplane_files[0]  # Take the most recent one

    # Extract timing data
    timing_data = compute_total_runtime_from_overview_page([xplane_file])

    # Add additional metadata
    timing_data["nrepeat"] = nrepeat
    timing_data["avg_time_us"] = timing_data["total_time_us"] / nrepeat
    timing_data["avg_time_ms"] = timing_data["total_time_ms"] / nrepeat
    timing_data["avg_time_s"] = timing_data["total_time_s"] / nrepeat
    timing_data["trace_dir"] = trace_dir
    timing_data["xplane_file"] = xplane_file

    return timing_data


def compute_total_runtime_from_overview_page(
    xplane_files: list[str],
) -> dict[str, float]:
    """
    Compute the total runtime of a function from the xprof overview page data.

    Args:
        xplane_files: List of paths to xplane.pb files

    Returns:
        A dictionary containing:
            - 'total_time_us': Total runtime in microseconds
            - 'total_time_ms': Total runtime in milliseconds
            - 'operations': List of dicts with operation details (name, time_us, time_percent)
    """
    from xprof.convert.raw_to_tool_data import xspace_to_tool_data

    # Get framework_op_stats which has the time in microseconds
    raw_data = xspace_to_tool_data(
        xplane_files,
        "framework_op_stats",
        {},
    )[0].decode("utf-8")

    data = json.loads(raw_data)

    # The data is a list of tables, we want the first one which has the op stats
    op_stats_table = data[0] if isinstance(data, list) else data

    # Extract column indices
    cols = op_stats_table["cols"]
    col_indices = {col["id"]: i for i, col in enumerate(cols)}

    # Get indices for the columns we need
    operation_idx = col_indices.get("operation")
    type_idx = col_indices.get("type")
    total_time_idx = col_indices.get("total_time")  # Total time in microseconds
    host_or_device_idx = col_indices.get("host_or_device")

    operations = []
    total_time_us = 0.0

    for row in op_stats_table["rows"]:
        cells = row["c"]

        host_or_device = (
            cells[host_or_device_idx]["v"] if host_or_device_idx is not None else None
        )
        op_type = cells[type_idx]["v"] if type_idx is not None else None

        # Skip IDLE time and host operations (we only want device compute time)
        if op_type == "IDLE":
            continue
        if host_or_device == "Host":
            continue

        op_name = cells[operation_idx]["v"] if operation_idx is not None else "Unknown"
        time_us = cells[total_time_idx]["v"] if total_time_idx is not None else 0.0

        operations.append(
            {
                "name": op_name,
                "type": op_type,
                "time_us": time_us,
            }
        )
        total_time_us += time_us

    # Calculate percentages
    for op in operations:
        op["time_percent"] = (
            (op["time_us"] / total_time_us * 100) if total_time_us > 0 else 0.0
        )

    # Sort by time descending
    operations.sort(key=lambda x: x["time_us"], reverse=True)

    return {
        "total_time_us": total_time_us,
        "total_time_ms": total_time_us / 1000.0,
        "total_time_s": total_time_us / 1_000_000.0,
        "operations": operations,
    }


def print_runtime_summary(result: dict) -> None:
    """Pretty print the runtime summary."""
    print(
        f"Total Runtime: {result['total_time_ms']:.3f} ms ({result['total_time_us']:.3f} μs)"
    )
    print("\nOperation Breakdown:")
    print("-" * 80)
    print(f"{'Operation':<50} {'Time (μs)':>12} {'%':>8}")
    print("-" * 80)
    for op in result["operations"]:
        print(f"{op['name']:<50} {op['time_us']:>12.3f} {op['time_percent']:>7.2f}%")
    print("-" * 80)
