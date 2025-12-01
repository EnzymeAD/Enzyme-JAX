import timeit
import io
import jax
import jax.numpy as jnp
from enzyme_ad.jax import enzyme_jax_ir, JaXPipeline
from test_utils import setup_backends


def count_transposes(ir_string):
    return ir_string.count("stablehlo.transpose")


def benchmark_symmetry():
    setup_backends()
    
    N = 2048
    key = jax.random.PRNGKey(0)
    X = jax.device_put(jax.random.normal(key, (N, N)))
    
    def single_symmetric_op(x):
        a = x.T + x
        return a + a.T

    def chained_symmetric_op(x):
        a = x.T + x
        for _ in range(9):
            a = a.T + a
        return a

    def interleaved_symmetric_op(x):
        a = x.T + x
        for _ in range(9):
            a = a.T * 0.99 + a * 0.01
        return a

    pipeline = JaXPipeline(
        "inline{default-pipeline=canonicalize max-iterations=4},"
        "partial-symmetry-annotate,enzyme-hlo-generate-td{patterns=transpose_partial_symmetry_simplify},transform-interpreter,enzyme-hlo-remove-transform"
    )
    
    NUM_ITER = 100
    tests = [
        ("Single op", single_symmetric_op),
        ("Chained (10x)", chained_symmetric_op),
        ("Interleaved (10x)", interleaved_symmetric_op),
    ]
    
    print(f"{'Test':<20} {'Transposes':<15} {'Baseline':<12} {'Optimized':<12} {'Speedup':<8}")
    print("-" * 70)
    
    for name, fn in tests:
        # Count transposes
        ir_buf = io.StringIO()
        _ = jax.jit(enzyme_jax_ir(pipeline_options=JaXPipeline(""), 
                                   jit_options={"print_mlir": ir_buf})(fn))(X).block_until_ready()
        base_t = count_transposes(ir_buf.getvalue())
        
        ir_buf = io.StringIO()
        _ = jax.jit(enzyme_jax_ir(pipeline_options=pipeline,
                                   jit_options={"print_mlir": ir_buf})(fn))(X).block_until_ready()
        opt_t = count_transposes(ir_buf.getvalue())
        
        # Benchmark
        baseline_fn = jax.jit(fn)
        _ = baseline_fn(X).block_until_ready()
        start = timeit.default_timer()
        for _ in range(NUM_ITER):
            _ = baseline_fn(X).block_until_ready()
        baseline_ms = (timeit.default_timer() - start) / NUM_ITER * 1000
        
        optimized_fn = jax.jit(enzyme_jax_ir(pipeline_options=pipeline)(fn))
        _ = optimized_fn(X).block_until_ready()
        start = timeit.default_timer()
        for _ in range(NUM_ITER):
            _ = optimized_fn(X).block_until_ready()
        opt_ms = (timeit.default_timer() - start) / NUM_ITER * 1000
        
        speedup = baseline_ms / opt_ms
        print(f"{name:<20} {base_t:>2} -> {opt_t:<2} (-{base_t-opt_t})  {baseline_ms:>8.2f} ms  {opt_ms:>8.2f} ms  {speedup:.2f}x")


if __name__ == "__main__":
    benchmark_symmetry()
