# Enzyme-JAX

Custom bindings for Enzyme within JaX. Currently this is set up to allow you
to automatically import, and automatically differentiate (both jvp and vjp)
external C++ code into JaX. As Enzyme is language-agnostic, this can be extended
for arbitrary programming languages (Julia, Swift, Fortran, Rust, and even Python)!

As JaX only supports a single custom rule (either forward or reverse) on a function,
Enzyme exports two versions of its FFI: one for forward mode, and one for reverse mode.

You can use 

```python
from enzyme_jax import cpp_fwd, cpp_rev

# Forward-mode C++ AD example

@jax.jit
def fwd_something(inp):
    y = cpp_fwd(inp, out_shapes=[jax.core.ShapedArray([2, 3], jnp.float32)], source="""
        template<std::size_t N, std::size_t M>
        void myfn(enzyme::tensor<float, N, M>& out0, const enzyme::tensor<float, N, M>& in0) {
        out0 = 56.0f + in0(0, 0);
        }
        """, fn="myfn")
    return y

ones = jnp.ones((2, 3), jnp.float32)
primals, tangents = jax.jvp(fwd_somthing, (ones,), (ones,) )

# Reverse-mode C++ AD example

@jax.jit
def rev_something(inp):
    y = cpp_rev(inp, out_shapes=[jax.core.ShapedArray([2, 3], jnp.float32)], source="""
        template<std::size_t N, std::size_t M>
        void myfn(enzyme::tensor<float, N, M>& out0, const enzyme::tensor<float, N, M>& in0) {
        out0 = 56.0f + in0(0, 0);
        }
        """, fn="myfn")
    return y

primals, f_vjp = jax.vjp(rev_something(), ones)
(grads,) = f_vjp((x, y, z))
```

# Installation

The easiest way to install is using pip.

! Note that the current pypi binary will only work on Linux. This is intended to be fixed once I get macOS CI hours. If anyone is interested with supporting please reach out to @wsmoses.

```bash
# The project is available on PyPi and installable like
# a usual python package (https://pypi.org/project/enzyme-jax/)
pip install enzyme-jax
```

## Building from source

Requirements: `bazel-5.3.0`, `clang++`, `python`, `python-virtualenv`,
`python3-dev`.

Build our extension with:
```sh
# Will create a whl in bazel-bin/enzyme_jax-VERSION-SYSTEM.whl
bazel build :enzyme_jax
```

Finally, install the built library with:
```sh
pip install bazel-bin/enzyme_jax-VERSION-SYSTEM.whl
```

## Running the test

```sh
pip install bazel-bin/enzyme_jax-VERSION-SYSTEM.whl
cd test && python test.py
```
