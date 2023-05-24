# Enzyme-JAX

## Building from source

Requirements: `bazel-5.3.0`, `clang++`, `python`, `python-virtualenv`,
`python3-dev`.

```sh
# Get submodules.
git submodule update --init --recursive

# Apply Bazel-related patches. This will enable JAX to build with the LLVM
# version provided as submodule instead of downloading the one specified by JAX.
# However dirty this looks, it is actually the suggested mechanism for JAX.
./patches/apply.sh

# Build and install JAX into a new virtual environment.
# Refer to https://jax.readthedocs.io/en/latest/developer.html for more details.
virtualenv .venv
pip install numpy wheel
cd jax
python build/build.py
pip install dist/*.whl --force-reinstall
pip install -e .
cd ..

# Build our extension.
bazel-5.3.0 build :enzyme_call
```

After changing LLVM, it is necessary to rebuild and reinstall JAX.

## Running the test

```sh
export PYTHONPATH=$PWD/bazel-bin:$PWD/jax:$PYTHONPATH
python primitives.py
```
