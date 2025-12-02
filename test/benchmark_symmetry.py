import timeit
import io
import os
from datetime import datetime
import jax
import jax.numpy as jnp
from enzyme_ad.jax import enzyme_jax_ir, JaXPipeline
from test_utils import setup_backends


def count_transposes(ir_string):
    return ir_string.count("stablehlo.transpose")


def benchmark_symmetry():
    setup_backends()
    
    # Create tmp directory for MLIR files in current directory
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    tmp_dir = os.path.join(".", "tmp", f"enzyme_mlir")
    os.makedirs(tmp_dir, exist_ok=True)
    
    N = 2048
    key = jax.random.PRNGKey(0)
    X = jax.device_put(jax.random.normal(key, (N, N)))
    
    def single_symmetric_op(x):
        a = x.T + x
        return a + a.T

    def chained_symmetric_op(x):
        a = x.T + x
        for _ in range(10):
            a = a.T + a
        return a

    def interleaved_symmetric_op(x):
        a = x.T + x
        for _ in range(10):
            a = a.T * 0.99 + a * 0.01
        return a

    pipeline_debug = JaXPipeline(
        "inline{default-pipeline=canonicalize max-iterations=4},"
        "partial-symmetry-annotate,enzyme-hlo-generate-td{patterns=transpose_partial_symmetry_simplify},transform-interpreter,enzyme-hlo-remove-transform"
    , keep_enzyme_attributes=True)

    pipeline= JaXPipeline(
        "inline{default-pipeline=canonicalize max-iterations=4},"
        "partial-symmetry-annotate,enzyme-hlo-generate-td{patterns=transpose_partial_symmetry_simplify},transform-interpreter,enzyme-hlo-remove-transform"
    )
    
    NUM_ITER = 100
    tests = [
        ("Single op", single_symmetric_op),
        ("Chained (10x)", chained_symmetric_op),
        ("Interleaved (10x)", interleaved_symmetric_op),
    ]
    
    # Collect MLIR file paths to print at the end
    mlir_files = []
    
    print(f"{'Test':<20} {'Transposes':<15} {'Baseline':<12} {'Optimized':<12} {'Speedup':<8}")
    print("-" * 70)
    
    for name, fn in tests:
        # Count transposes
        ir_buf = io.StringIO()
        _ = jax.jit(enzyme_jax_ir(pipeline_options=JaXPipeline(""), 
                                   jit_options={"print_mlir": ir_buf})(fn))(X).block_until_ready()
        base_mlir = ir_buf.getvalue()
        base_t = count_transposes(base_mlir)
        
        # Save MLIR before enzyme opt
        test_name_safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        before_path = os.path.join(tmp_dir, f"before_{test_name_safe}.mlir")
        with open(before_path, 'w') as f:
            f.write(base_mlir)
        mlir_files.append(("before", name, os.path.abspath(before_path)))
        
        # Capture MLIR with pipeline_debug (may fail to parse due to unregistered dialects,
        # but MLIR will still be written to buffer before the exception)
        ir_buf = io.StringIO()
        try:
            _ = enzyme_jax_ir(pipeline_options=pipeline_debug,
                              jit_options={"print_mlir": ir_buf})(fn)(X).block_until_ready()
        except Exception:
            # MLIR was already written to buffer before parsing failed
            # The exception is expected when keep_enzyme_attributes=True due to unregistered dialects
            pass
        
        opt_mlir = ir_buf.getvalue()
        opt_t = count_transposes(opt_mlir)
        
        # Save MLIR after enzyme opt
        after_path = os.path.join(tmp_dir, f"after_{test_name_safe}.mlir")
        with open(after_path, 'w') as f:
            f.write(opt_mlir)
        mlir_files.append(("after", name, os.path.abspath(after_path)))
        
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
    
    # Print MLIR file paths after the table
    print("\n" + "=" * 70)
    print("Saved MLIR files:")
    print("=" * 70)
    for stage, test_name, file_path in mlir_files:
        print(f"  {stage.upper():<6} {test_name:<20}: {file_path}")


if __name__ == "__main__":
    benchmark_symmetry()
