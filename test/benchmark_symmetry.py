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

    def dot_cse(x):
        a = x.T + x
        return jnp.dot(a, a) + jnp.dot(a, a.T) + jnp.dot(a.T, a)
        
    def reduce_partial_symmetry(x):
        a = x.transpose((3, 1, 2, 0)) + x
        result = jnp.zeros(a.shape[1:])
        for _ in range(20):
            reduced = jnp.sum(a, axis=0)
            result = result + reduced
            a = a + 1 
        return result
        
    passes = "inline{default-pipeline=canonicalize max-iterations=4}, canonicalize, cse, partial-symmetry-annotate, enzyme-hlo-generate-td{patterns=transpose_partial_symmetry_simplify;reduce_partial_symmetry_rotate_axes}, transform-interpreter, enzyme-hlo-remove-transform, canonicalize, cse"
    passes_control = "inline{default-pipeline=canonicalize max-iterations=4}, canonicalize, cse"

    pipeline_debug = JaXPipeline(passes, keep_enzyme_attributes=True)
    pipeline = JaXPipeline(passes)
    pipeline_control = JaXPipeline(passes_control)
    
    NUM_ITER = 100
    tests = [
        ("Single op", single_symmetric_op, (2048, 2048)),
        ("Chained (10x)", chained_symmetric_op, (2048, 2048)),
        ("Interleaved (10x)", interleaved_symmetric_op, (2048, 2048)),
        ("Dot CSE", dot_cse, (1024, 1024)),
        ("Reduce partial symmetry", reduce_partial_symmetry, (32, 32, 32, 32))
    ]
    
    # Collect MLIR file paths to print at the end
    mlir_files = []
    
    print(f"{'Test':<20} {'Shape':<15} {'Transposes':<15} {'Baseline':<12} {'Optimized':<12} {'Speedup':<8}")
    print("-" * 85)
    
    for name, fn, shape in tests:
        # Construct input X based on shape
        key = jax.random.PRNGKey(0)
        X = jax.device_put(jax.random.normal(key, shape))
        # Count transposes
        ir_buf = io.StringIO()
        _ = jax.jit(enzyme_jax_ir(pipeline_options=pipeline_control, 
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
        shape_str = f"{shape[0]}x{shape[1]}" if len(shape) == 2 else str(shape)
        print(f"{name:<20} {shape_str:<15} {base_t:>2} -> {opt_t:<2} (-{base_t-opt_t})  {baseline_ms:>8.2f} ms  {opt_ms:>8.2f} ms  {speedup:.2f}x")
    
    # Print MLIR file paths after the table
    print("\n" + "=" * 70)
    print("Saved MLIR files:")
    print("=" * 70)
    for stage, test_name, file_path in mlir_files:
        print(f"  {stage.upper():<6} {test_name:<20}: {file_path}")


if __name__ == "__main__":
    benchmark_symmetry()
