import timeit
import io
import os
from datetime import datetime
import jax
import jax.numpy as jnp
import jax.scipy.linalg
from enzyme_ad.jax import enzyme_jax_ir, JaXPipeline
from test_utils import setup_backends


def count_transposes(ir_string):
    return ir_string.count("stablehlo.transpose")


def benchmark_symmetry():
    setup_backends()
    
    # Register BLAS symbols for Enzyme-JAX JIT
    import ctypes
    import os
    import sys
    
    try:
        from enzyme_ad.jax import enzyme_call
        enzyme_call_path = os.path.dirname(enzyme_call.__file__)
        
        # Try to load Enzyme-JAX shared library
        enzyme_lib = None
        for lib_name in ['enzyme_ad_jax', 'enzyme_call', 'enzyme']:
            for ext in ['.so', '.dylib', '.dll']:
                lib_path = os.path.join(enzyme_call_path, f'lib{lib_name}{ext}')
                if os.path.exists(lib_path):
                    try:
                        enzyme_lib = ctypes.CDLL(lib_path)
                        break
                    except:
                        pass
            if enzyme_lib:
                break
        
        if not enzyme_lib:
            # Try loading from the enzyme_call module's directory
            try:
                # enzyme_call might be a .so file itself
                if hasattr(enzyme_call, '__file__'):
                    enzyme_file = enzyme_call.__file__
                    if enzyme_file.endswith('.so') or enzyme_file.endswith('.dylib'):
                        try:
                            enzyme_lib = ctypes.CDLL(enzyme_file)
                        except:
                            pass
            except:
                pass
        
        if not enzyme_lib:
            for lib_name in ['enzyme_ad_jax', 'enzyme_call', 'enzyme']:
                try:
                    enzyme_lib = ctypes.CDLL(f'lib{lib_name}.so')
                    break
                except:
                    pass
        
        if enzyme_lib:
            try:
                map_symbol = enzyme_lib.EnzymeJaXMapSymbol
                map_symbol.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
                map_symbol.restype = None
                
                # Load BLAS library - try multiple approaches
                blas_lib = None
                blas_names = ['libblas.so.3', 'libopenblas.so.0', 'libblas.so', 'libcblas.so', 
                             'libblas.so.2', 'libopenblas.so']
                
                for blas_name in blas_names:
                    try:
                        blas_lib = ctypes.CDLL(blas_name, ctypes.RTLD_GLOBAL)
                        break
                    except:
                        pass
                
                # Also try using numpy's BLAS if available
                if not blas_lib:
                    try:
                        import numpy as np
                        # numpy might have BLAS linked
                        blas_lib = ctypes.CDLL(np.__file__.replace('__init__.py', '') + 
                                              '../.libs/libopenblas.so.0', ctypes.RTLD_GLOBAL)
                    except:
                        pass
                
                if blas_lib:
                    # Register ssyrk_ (single precision) - try both with and without underscore
                    registered_ssyrk = False
                    for sym_name in ['ssyrk_', 'ssyrk']:
                        try:
                            ssyrk_sym = getattr(blas_lib, sym_name)
                            # Get the actual function pointer address
                            if hasattr(ssyrk_sym, '_handle'):
                                func_ptr = ssyrk_sym._handle
                            else:
                                func_ptr = ctypes.cast(ssyrk_sym, ctypes.c_void_p).value
                            if func_ptr:
                                map_symbol(b'enzymexla_blas_ssyrk_', func_ptr)
                                registered_ssyrk = True
                                print(f"Registered enzymexla_blas_ssyrk_ from {sym_name}", file=sys.stderr)
                                break
                        except (AttributeError, TypeError) as e:
                            continue
                    
                    if not registered_ssyrk:
                        print("Warning: Could not register ssyrk_", file=sys.stderr)
                    
                    # Register dsyrk_ (double precision)
                    registered_dsyrk = False
                    for sym_name in ['dsyrk_', 'dsyrk']:
                        try:
                            dsyrk_sym = getattr(blas_lib, sym_name)
                            if hasattr(dsyrk_sym, '_handle'):
                                func_ptr = dsyrk_sym._handle
                            else:
                                func_ptr = ctypes.cast(dsyrk_sym, ctypes.c_void_p).value
                            if func_ptr:
                                map_symbol(b'enzymexla_blas_dsyrk_', func_ptr)
                                registered_dsyrk = True
                                print(f"Registered enzymexla_blas_dsyrk_ from {sym_name}", file=sys.stderr)
                                break
                        except (AttributeError, TypeError):
                            continue
                    
                    if not registered_dsyrk:
                        print("Warning: Could not register dsyrk_", file=sys.stderr)
                        
                else:
                    print("Warning: Could not load BLAS library", file=sys.stderr)
            except AttributeError as e:
                print(f"Warning: Could not find EnzymeJaXMapSymbol: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Error registering BLAS symbols: {e}", file=sys.stderr)
        else:
            print("Warning: Could not load Enzyme-JAX library", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Could not register BLAS symbols: {e}", file=sys.stderr)
    
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
    
    def solve_linear_system(x):
        a = x.T + x 
        y = jnp.ones(a.shape[0])  # Right-hand side vector
        return jnp.linalg.solve(a, y)
    
    def solve_linear_system_cholesky(x):
        a = x.T + x 
        y = jnp.ones(a.shape[0])  # Right-hand side vector
        L = jnp.linalg.cholesky(a)  # x = L @ L.T
        z = jax.scipy.linalg.solve_triangular(L, y, lower=True)
        return jax.scipy.linalg.solve_triangular(L.T, z, lower=False)
        
    def matrix_multiply(x):
        a = x.T + x 
        return jnp.matmul(a, a)
        
    passes = "inline{default-pipeline=canonicalize max-iterations=4}, canonicalize, cse, partial-symmetry-annotate, enzyme-hlo-generate-td{patterns=transpose_partial_symmetry_simplify;reduce_partial_symmetry_rotate_axes;dot_general_to_syrk;transpose_syrk_to_syrk;fuse_mul_into_syrk;fuse_add_into_syrk}, transform-interpreter,lower-jit{backend=cpu},  lower-enzymexla-blas{backend=cpu}, lower-jit{backend=cpu}, enzyme-hlo-remove-transform, canonicalize, cse"
    passes_control = "inline{default-pipeline=canonicalize max-iterations=4}, canonicalize, cse"

    pipeline_debug = JaXPipeline(passes, keep_enzyme_attributes=True)
    pipeline = JaXPipeline(passes)
    pipeline_control = JaXPipeline(passes_control)
    
    NUM_ITER = 100
    tests = [
        # ("Single op", single_symmetric_op, (2048, 2048)),
        # ("Chained (10x)", chained_symmetric_op, (2048, 2048)),
        # ("Interleaved (10x)", interleaved_symmetric_op, (2048, 2048)),
        # ("Dot CSE", dot_cse, (1024, 1024)),
        # ("Reduce partial symmetry", reduce_partial_symmetry, (32, 32, 32, 32)),
        ("Matrix multiply", matrix_multiply, (2048, 2048))
    ]
    
    # Collect MLIR file paths to print at the end
    mlir_files = []
    
    print(f"{'Test':<20} {'Shape':<15} {'Transposes':<15} {'Baseline':<12} {'Optimized':<12} {'Speedup':<8}")
    print("-" * 85)
    
    for name, fn, shape in tests:
        # Construct input X based on shape
        key = jax.random.PRNGKey(0)
        X = jax.device_put(jax.random.uniform(key, shape))
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
