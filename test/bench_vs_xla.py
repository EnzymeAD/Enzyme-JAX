import jax
import jax.numpy as jnp
from enzyme_jax import enzyme_jax_ir


@enzyme_jax_ir()
def add_one(x: jax.Array, y) -> jax.Array:
  return x + 1 + y

@jax.jit
def add_one_plain(x: jax.Array, y) -> jax.Array:
  return x + 1 + y

in0, in1 = jnp.array([1., 2., 3.]), jnp.array([10., 20., 30.])
# TODO: this currently throws NYI as it is not yet connected to JIT and runtime.
# But it should print LLVM IR in the process.

add_one(in0, in1)
add_one_plain(in0, in1)
import timeit

print(timeit.Timer('add_one(in0, in1)', globals={'add_one':add_one, 'in0':in0, 'in1':in1}).timeit())
print(timeit.Timer('add_one_plain(in0, in1)', globals={'add_one_plain':add_one_plain, 'in0':in0, 'in1':in1}).timeit())

din0, din1 = (jnp.array([.1, .2, .3]), jnp.array([50., 70., 110.]))

@jax.jit
def fwd(in0, in1, din0, din1):
  return jax.jvp(add_one, (in0, in1),  (din0, din1))

@jax.jit
def fwd_plain(in0, in1, din0, din1):
  return jax.jvp(add_one_plain, (in0, in1),  (din0, din1))

primals, tangents = fwd(in0, in1, din0, din1)
primals, tangents = fwd_plain(in0, in1, din0, din1)

print(timeit.Timer('fwd(in0, in1, din0, din1)', globals={'fwd':fwd, 'in0':in0, 'in1':in1, 'din0':din0, 'din1':din1}).timeit())
print(timeit.Timer('fwd_plain(in0, in1, din0, din1)', globals={'fwd_plain':fwd_plain, 'in0':in0, 'in1':in1, 'din0':din0, 'din1':din1}).timeit())


@jax.jit
def rev(in0, in1, dout):
  primals, f_vjp = jax.vjp(add_one, in0, in1)
  grads = f_vjp(dout)
  return primals, grads

@jax.jit
def rev_plain(in0, in1, dout):
  primals, f_vjp = jax.vjp(add_one_plain, in0, in1)
  grads = f_vjp(dout)
  return primals, grads

dout = jnp.array([500., 700., 110.])

rev(in0, in1, dout)
rev_plain(in0, in1, dout)

print(rev_plain.lower(in0, in1, dout).compiler_ir(dialect="mhlo"))

rev_plain(in0, in1, dout)

print(timeit.Timer('rev(in0, in1, dout)', globals={'rev':rev, 'in0':in0, 'in1':in1, 'dout':dout}).timeit())
print(timeit.Timer('rev_plain(in0, in1, dout)', globals={'rev_plain':rev_plain, 'in0':in0, 'in1':in1, 'dout':dout}).timeit())

