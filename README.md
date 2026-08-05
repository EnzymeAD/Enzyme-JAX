# Enzyme-JAX

Enzyme-JAX is a C++ project whose original aim was to integrate the Enzyme automatic differentiation tool [1] with JAX, enabling automatic differentiation of external C++ code within JAX. It has since expanded to incorporate Polygeist's [2] high performance raising, parallelization, cross compilation workflow, as well as numerous tensor, linear algerba, and communication optimizations. The project uses LLVM's MLIR framework for intermediate representation and transformation of code. As Enzyme is language-agnostic, this can be extended for arbitrary programming
languages (Julia, Swift, Fortran, Rust, and even Python)!

# Usage Examples

## Usage with C++

You can use `cpp_call` to differentiate external C++ code:

```python
from enzyme_ad.jax import cpp_call

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

## Usage with Jax

You can also use Enzyme to optimize and differentiate vanilla JAX code using the `@enzyme_jax_ir` decorator. This allows applying Enzyme's optimizations and AD to standard JAX functions.

```python
from enzyme_ad.jax import enzyme_jax_ir
import jax
import jax.numpy as jnp

# Apply Enzyme optimizations and AD support
@jax.jit
@enzyme_jax_ir
def add_one(x, y):
    return x + 1 + y

x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([10.0, 20.0, 30.0])

# Run the function
result = add_one(x, y)
print("Result:", result)

# Forward-mode AD (JVP)
primals, tangents = jax.jvp(
    add_one,
    (x, y),
    (jnp.array([0.1, 0.2, 0.3]), jnp.array([50.0, 70.0, 110.0])),
)
print("Primals:", primals)
print("Tangents:", tangents)

# Reverse-mode AD (VJP)
primals, f_vjp = jax.vjp(add_one, x, y)
grads = f_vjp(jnp.array([500.0, 700.0, 110.0]))
print("Gradients:", grads)
```

# Installation

The easiest way to install is using pip.

```bash
# The project is available on PyPi and installable like
# a usual python package (https://pypi.org/project/enzyme-ad/)
pip install enzyme-ad
```

## Building from source

Requirements: `bazel-6.5`, `clang++`, `python`, `python-virtualenv`,
`python3-dev`.

Build our extension with:
```sh
# Will create a whl in bazel-bin/enzyme_ad-VERSION-SYSTEM.whl
bazel build :wheel
```

Finally, install the built library with:
```sh
pip install bazel-bin/enzyme_ad-VERSION-SYSTEM.whl
```
Note that you cannot run code from the root of the git directory. For instance, in the code below, you have to first run `cd test` before running `test.py`.

## Running the test

To run tests, you can simply execute the following bazel commands (this does not require building or installing the wheel).
```sh
bazel test //test/...
```

Alternatively, if you have installed the wheel, you can manually invoke the tests as follows
```sh
cd test && python test.py
```
## LSP Support

Enzyme-Jax exposes a bunch of different tensor rewrites as MLIR passes in `src/enzyme_ad/jax/Passes`. If you want to enable LSP support when working with this code, we recommend that you generate a `compile_commands.json` by running

```bash
bazel run :refresh_compile_commands
```

# References
[1] Moses, William, and Valentin Churavy. "Instead of rewriting foreign code for machine learning, automatically synthesize fast gradients." Advances in neural information processing systems 33 (2020): 12472-12485.

[2] Moses, William S., et al. "Polygeist: Raising C to polyhedral MLIR." 2021 30th International Conference on Parallel Architectures and Compilation Techniques (PACT). IEEE, 2021.
