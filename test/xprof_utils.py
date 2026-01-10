"""Utility functions for extracting profiling data from xprof/XLA traces."""

import glob
import json
import os
import tempfile
import timeit
import warnings
from typing import Any, Callable

try:
    from xprof.convert.raw_to_tool_data import xspace_to_tool_data

    XPROF_AVAILABLE = True
except ImportError:
    XPROF_AVAILABLE = False


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
        fn: The function to profile
        args: Positional arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        nrepeat: Number of times to run the function during profiling
        warmup: Number of warmup runs before profiling (to ensure compilation is done)
        trace_dir: Directory to save traces. If None, uses a temporary directory.

    Returns:
        A dictionary containing:
            - 'min_time_us': Minimum runtime per iteration in microseconds
            - 'min_time_ms': Minimum runtime per iteration in milliseconds
            - 'min_time_s': Minimum runtime per iteration in seconds
            - 'avg_time_s': Average runtime per iteration in seconds (same as min_time_s)
            - 'nrepeat': Number of repetitions
            - 'trace_dir': Directory where traces were saved
            - 'xplane_file': Path to the xplane.pb file
    """
    if kwargs is None:
        kwargs = {}

    import jax

    compiled_fn = jax.jit(fn).trace(*args, **kwargs).lower().compile()

    # Warmup runs (ensure compilation is complete)
    for _ in range(warmup):
        jax.block_until_ready(compiled_fn(*args, **kwargs))

    profile_compiled_function(compiled_fn, args, kwargs, nrepeat, trace_dir)


def profile_compiled_function(
    compiled_fn: Callable,
    args: tuple = (),
    kwargs: dict | None = None,
    nrepeat: int = 1,
    trace_dir: str | None = None,
) -> dict[str, Any]:
    """
    Profile a JAX function and return timing data.

    Args:
        fn: The function to profile. We assume that the function was AoT compiled.
        args: Positional arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        nrepeat: Number of times to run the function during profiling
        trace_dir: Directory to save traces. If None, uses a temporary directory.

    Returns:
        A dictionary containing:
            - 'min_time_us': Minimum runtime per iteration in microseconds
            - 'min_time_ms': Minimum runtime per iteration in milliseconds
            - 'min_time_s': Minimum runtime per iteration in seconds
            - 'avg_time_s': Average runtime per iteration in seconds (same as min_time_s)
            - 'nrepeat': Number of repetitions
            - 'trace_dir': Directory where traces were saved
            - 'xplane_file': Path to the xplane.pb file
    """
    import jax

    if kwargs is None:
        kwargs = {}

    if not XPROF_AVAILABLE:
        warnings.warn("xprof not found, falling back to timeit for profiling.")
        # Fallback to timeit
        times = []
        for _ in range(nrepeat):
            start = timeit.default_timer()
            jax.block_until_ready(compiled_fn(*args, **kwargs))
            end = timeit.default_timer()
            times.append(end - start)

        min_time_s = min(times)
        return {
            "min_time_us": min_time_s * 1e6,
            "min_time_ms": min_time_s * 1e3,
            "min_time_s": min_time_s,
            "avg_time_s": sum(times) / len(times),
            "nrepeat": nrepeat,
            "trace_dir": None,
            "xplane_file": None,
        }

    # Create trace directory
    if trace_dir is None:
        trace_dir = tempfile.mkdtemp(prefix="jax_profile_")

    # Profile the function with step annotations for proper wall-clock timing
    with jax.profiler.trace(trace_dir):
        for i in range(nrepeat):
            # Use StepTraceAnnotation to mark each iteration as a step
            # This allows xprof to compute proper wall-clock step times
            with jax.profiler.StepTraceAnnotation("step", step_num=i):
                jax.block_until_ready(compiled_fn(*args, **kwargs))

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
    min_time_ms = extract_min_step_time([xplane_file], nrepeat)

    return {
        "min_time_us": min_time_ms * 1000.0,
        "min_time_ms": min_time_ms,
        "min_time_s": min_time_ms / 1000.0,
        "avg_time_s": min_time_ms / 1000.0,  # For compatibility with test harness
        "nrepeat": nrepeat,
        "trace_dir": trace_dir,
        "xplane_file": xplane_file,
    }


def extract_min_step_time(xplane_files: list[str], nrepeat: int) -> float:
    """
    Extract the minimum step time from xprof data.

    Parses the overview_page step table rows to get full-precision per-step
    timing data and returns the minimum step time (best case for benchmarking).

    Args:
        xplane_files: List of paths to xplane.pb files

    Returns:
        Minimum step time in milliseconds
    """
    if not XPROF_AVAILABLE:
        raise RuntimeError("xprof is not available.")

    res = extract_min_step_time_from_overview_data(xplane_files, nrepeat)
    if res is None:
        return extract_min_step_time_from_hlo_op_profile(xplane_files, nrepeat)
    return res


def extract_min_step_time_from_hlo_op_profile(
    xplane_files: list[str], nrepeat: int
) -> None | float:
    data = json.loads(
        xspace_to_tool_data(xplane_files, "op_profile", {})[0].decode("utf-8")
    )
    picosec = data["byProgram"]["metrics"]["normalizedTimePs"]
    return (picosec / 1e9) / nrepeat


def extract_min_step_time_from_overview_data(
    xplane_files: list[str], nrepeat: int
) -> None | float:
    overview_data = json.loads(
        xspace_to_tool_data(xplane_files, "overview_page", {})[0].decode("utf-8")
    )

    # overview_page returns a list of tables; the second table (index 1)
    # contains step timing information with per-step data in rows
    if not isinstance(overview_data, list) or len(overview_data) <= 1:
        return None

    step_table = overview_data[1]
    cols = step_table.get("cols", [])
    rows = step_table.get("rows", [])

    # Find the column index for stepTimeMs (full precision step time)
    col_indices = {col["id"]: i for i, col in enumerate(cols)}
    step_time_idx = col_indices.get("stepTimeMs")

    if step_time_idx is None or not rows:
        return None

    # Extract step times from each row with full precision
    step_times_ms = []
    for row in rows:
        cells = row.get("c", [])
        if step_time_idx < len(cells):
            step_time = cells[step_time_idx].get("v", 0.0)
            if step_time > 0:
                step_times_ms.append(step_time)

    if not step_times_ms:
        return None

    # Return minimum step time for benchmarking (best case)
    return min(step_times_ms)


def print_runtime_summary(result: dict) -> None:
    """Pretty print the runtime summary."""
    print(
        f"Min Runtime: {result['min_time_ms']:.6f} ms ({result['min_time_us']:.3f} μs)"
    )
