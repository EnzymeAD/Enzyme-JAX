import jax
import jax.numpy as jnp
from enzyme_jax import enzyme_jax_ir


@enzyme_jax_ir()
def add_one(x: jax.Array, y) -> jax.Array:
  return x + 1 + y


# TODO: this currently throws NYI as it is not yet connected to JIT and runtime.
# But it should print LLVM IR in the process.
add_one(jnp.array([1., 2., 3.]), jnp.array([1., 2., 3.]))

