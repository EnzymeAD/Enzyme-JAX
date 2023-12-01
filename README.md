# Enzyme-JAX

Custom bindings for Enzyme within JaX. Currently this is set up to allow you
to automatically import, and automatically differentiate (both jvp and vjp)
external C++ code into JaX. As Enzyme is language-agnostic, this can be extended
for arbitrary programming languages (Julia, Swift, Fortran, Rust, and even Python)!

You can use 

```python
from enzyme_jax import cpp_call

# Forward-mode C++ AD example

@jax.jit
def something(inp):
    y = cpp_call(inp, out_shapes=[jax.core.ShapedArray([2, 3], jnp.float32)], source="""
        template<std::size_t N, std::size_t M>
        void myfn(enzyme::tensor<float, N, M>& out0, const enzyme::tensor<float, N, M>& in0) {
        out0 = 56.0f + in0(0, 0);
        }
        """, fn="myfn")
    return y

ones = jnp.ones((2, 3), jnp.float32)
primals, tangents = jax.jvp(something, (ones,), (ones,) )

# Reverse-mode C++ AD example

primals, f_vjp = jax.vjp(something, ones)
(grads,) = f_vjp((x,))
```

# Installation

The easiest way to install is using pip.

```bash
# The project is available on PyPi and installable like
# a usual python package (https://pypi.org/project/enzyme-jax/)
pip install enzyme-jax
```

## Building from source

Requirements: `bazel-5.4.0`, `clang++`, `python`, `python-virtualenv`,
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
Note that you cannot run code from the root of the git directory. For instance, in the code below, you have to first run `cd test` before running `test.py`.

## Running the test

```sh
cd test && python test.py
```
