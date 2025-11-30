
import timeit
import jax
import jax.numpy as jnp
from enzyme_ad.jax import enzyme_jax_ir, JaXPipeline
from test_utils import setup_backends

def benchmark_symmetry():
    setup_backends()
    
    # Create a large symmetric tensor
    N = 4096 * 4
    # Create a symmetric matrix: A = X + X.T
    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (N, N))
    A = X + X.T
    
    # Ensure it's on the device (GPU if available)
    A = jax.device_put(A)
    
    def symmetric_op(x):
        return x.T + x

    # 1. Baseline: Standard JAX JIT
    baseline_fn = jax.jit(symmetric_op)
    
    # Warmup
    _ = baseline_fn(A).block_until_ready()
    
    # Benchmark Baseline
    start_time = timeit.default_timer()
    for _ in range(100):
        _ = baseline_fn(A).block_until_ready()
    end_time = timeit.default_timer()
    baseline_avg = (end_time - start_time) / 10.0
    print(f"Baseline (Standard JAX) Average Time: {baseline_avg:.6f} s")

    # 2. Optimized: Enzyme-JAX with partial-symmetry-simplify
    # We need a pipeline that includes our pass.
    # Based on Passes.td, the pass is 'partial-symmetry-simplify'.
    # We'll construct a pipeline string.
    
    # The pass 'partial-symmetry-simplify' is defined in Passes.td.
    # We can try to construct a pipeline that runs it.
    # Usually analysis runs before simplification.
    
    pipeline_str = "partial-symmetry-simplify"
    
    optimized_fn = jax.jit(
        enzyme_jax_ir(pipeline_options=JaXPipeline(pipeline_str))(symmetric_op)
    )

    # Warmup
    _ = optimized_fn(A).block_until_ready()
    
    # Benchmark Optimized
    start_time = timeit.default_timer()
    for _ in range(100):
        _ = optimized_fn(A).block_until_ready()
    end_time = timeit.default_timer()
    optimized_avg = (end_time - start_time) / 10.0
    print(f"Optimized (Enzyme-JAX) Average Time: {optimized_avg:.6f} s")
    
    speedup = baseline_avg / optimized_avg
    print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    benchmark_symmetry()
