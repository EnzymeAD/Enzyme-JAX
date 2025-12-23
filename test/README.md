# Enzyme-JAX Python Tests

This directory contains Python tests for Enzyme-JAX. Tests can be run using either Bazel (traditional) or `uv` (recommended for development).

## Quick Start with `uv`

### Prerequisites

1. **Install `uv`**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Build the enzyme-ad wheel** (required):
   ```bash
   cd /path/to/Enzyme-JAX
   bazel build :wheel
   ```

### Run Tests

```bash
cd test/

# Run basic tests (CPU)
uv run python test.py

# Run benchmark tests (CPU)
uv run python bench_vs_xla.py

# Run LLaMA tests  
uv run python llama.py

# Run FFI tests
uv run python testffi.py
```

### Run with GPU (CUDA)

```bash
# CUDA tests (includes xprof for profiling)
uv run --extra cuda python bench_vs_xla.py
uv run --extra cuda python llama.py
```

### Run with TPU

```bash
# TPU tests (includes xprof for profiling)
uv run --extra tpu python bench_vs_xla.py
```

### Run Tests with Other Extras

```bash
# Run with JAX-MD (for jaxmd.py)
uv run --extra jaxmd python jaxmd.py

# Run with NeuralGCM (for neuralgcm_test.py)
uv run --extra neuralgcm python neuralgcm_test.py

# Run with Keras (for keras_test.py)
uv run --extra keras python keras_test.py

# Combine extras
uv run --extra cuda --extra jaxmd python jaxmd.py
```

## Test Files

| Test File | Description | Extras Needed |
|-----------|-------------|---------------|
| `test.py` | Basic enzyme tests | None |
| `bench_vs_xla.py` | Benchmark tests against XLA | `--extra cuda` or `--extra tpu` for GPU/TPU |
| `testffi.py` | FFI tests | None |
| `llama.py` | LLaMA model tests | `--extra cuda` for GPU |
| `jaxmd.py` | JAX-MD molecular dynamics tests | `--extra jaxmd` |
| `neuralgcm_test.py` | NeuralGCM weather model tests | `--extra neuralgcm` |
| `keras_test.py` | Keras model tests | `--extra keras` |

## Available Extras

| Extra | Description |
|-------|-------------|
| `cuda` | CUDA GPU support + xprof profiling |
| `tpu` | TPU support + xprof profiling |
| `jaxmd` | JAX-MD molecular dynamics |
| `neuralgcm` | NeuralGCM weather model |
| `keras` | Keras deep learning |
| `all` | All extras (CPU) |
| `all-cuda` | All extras with CUDA |
| `all-tpu` | All extras with TPU |

## How It Works

The `pyproject.toml` configures:
- **Python 3.11**: Pinned to match the Bazel-built enzyme-ad wheel
- **enzyme-ad**: Found via `find-links` pointing to `../bazel-bin`
- **Base deps**: jax, jaxlib, numpy, absl-py
- **Optional extras**: cuda (with xprof), tpu (with xprof), jaxmd, neuralgcm, keras

When you run `uv run`, it:
1. Creates a `.venv` with Python 3.11
2. Installs all dependencies including enzyme-ad from bazel-bin
3. Runs your command in that environment

## Advanced Usage

### Using a Different Wheel

If you have an enzyme-ad wheel elsewhere:

```bash
uv run --find-links /path/to/wheels python test.py
```

### Fresh Environment

```bash
rm -rf .venv
uv run python test.py
```

## Alternative: run_tests.py

The `run_tests.py` script provides additional features like:
- Separate venvs per test group
- Automatic enzyme wheel discovery
- Platform detection

```bash
python run_tests.py test
python run_tests.py bench --cuda
python run_tests.py --list
```

## Bazel Testing (Legacy)

Tests can still be run with Bazel:

```bash
bazel test //test:test
bazel test //test:bench_vs_xla
bazel test //test:llama
```
