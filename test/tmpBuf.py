import jax
import jax.numpy as jnp
from enzyme_jax import enzyme_jax_ir

x = jnp.array(range(50), dtype=jnp.float32) 
dx = jnp.array([i*i for i in range(50)], dtype=jnp.float32) 

@enzyme_jax_ir()
def esum(x):
    return jnp.sum(x)

eres = esum(x)
print(eres)
assert jnp.abs(eres-50*49/2)<1e-6

@jax.jit
def sumfwd(in0, din0):
  return jax.jvp(esum, (in0,), (din0,))

primals, tangents = sumfwd(x, dx)
print(primals, tangents)
assert jnp.abs(primals-50*49/2)<1e-6
assert jnp.abs(tangents-50*49*99/6)<1e-6

@jax.jit
def sumrev_p(in0):
  primals, f_vjp = jax.vjp(jnp.sum, in0)
  grads = f_vjp(1.0)
  return primals, grads

primals, grads = sumrev_p(x)
print(primals, grads)

@jax.jit
def sumrev(in0):
  primals, f_vjp = jax.vjp(esum, in0)
  grads = f_vjp(1.0)
  return primals, grads

primals, grads = sumrev(x)
print(primals, grads)
assert jnp.abs(primals-50*49/2)<1e-6
assert (jnp.abs(grads-1) <1e-6).all()

