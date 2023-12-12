import jax
import jax.numpy as jnp
from enzyme_ad.jax import enzyme_jax_ir


@enzyme_jax_ir()
def add_one(x: jax.Array, y) -> jax.Array:
    return x + 1 + y


@jax.jit
def add_one_plain(x: jax.Array, y) -> jax.Array:
    return x + 1 + y


@enzyme_jax_ir()
def add_two(x: jax.Array, z, y) -> jax.Array:
    return x + y


@jax.jit
def add_two_plain(x: jax.Array, z, y) -> jax.Array:
    return x + y


in0, in1, in2 = (
    jnp.array([1.0, 2.0, 3.0]),
    jnp.array([10.0, 20.0, 30.0]),
    jnp.array([100.0, 200.0, 300.0]),
)
# TODO: this currently throws NYI as it is not yet connected to JIT and runtime.
# But it should print LLVM IR in the process.

ao = add_one(in0, in1)
aop = add_one_plain(in0, in1)
assert (jnp.abs(ao - aop) < 1e-6).all()
print("Primal success")

at = add_two(in0, in1, in2)
atp = add_two_plain(in0, in1, in2)

assert (jnp.abs(at - atp) < 1e-6).all()
print("Primal Deadarg success")

import timeit

print(
    timeit.Timer(
        "add_one(in0, in1)", globals={"add_one": add_one, "in0": in0, "in1": in1}
    ).timeit()
)
print(
    timeit.Timer(
        "add_one_plain(in0, in1)",
        globals={"add_one_plain": add_one_plain, "in0": in0, "in1": in1},
    ).timeit()
)

din0, din1, din2 = (
    jnp.array([0.1, 0.2, 0.3]),
    jnp.array([50.0, 70.0, 110.0]),
    jnp.array([1300.0, 1700.0, 1900.0]),
)


@jax.jit
def fwd(in0, in1, din0, din1):
    return jax.jvp(add_one, (in0, in1), (din0, din1))


@jax.jit
def fwd_plain(in0, in1, din0, din1):
    return jax.jvp(add_one_plain, (in0, in1), (din0, din1))


primals, tangents = fwd(in0, in1, din0, din1)
primals_p, tangents_p = fwd_plain(in0, in1, din0, din1)

assert (jnp.abs(primals - primals_p) < 1e-6).all()
for t, t_p in zip(tangents, tangents_p):
    assert (jnp.abs(t - t_p) < 1e-6).all()

print("Tangent success")


@jax.jit
def fwd2(in0, in1, in2, din0, din1, din2):
    return jax.jvp(add_two, (in0, in1, in2), (din0, din1, din2))


@jax.jit
def fwd2_plain(in0, in1, in2, din0, din1, din2):
    return jax.jvp(add_two_plain, (in0, in1, in2), (din0, din1, din2))


primals, tangents = fwd2(in0, in1, in2, din0, din1, din2)
primals_p, tangents_p = fwd2_plain(in0, in1, in2, din0, din1, din2)

print(primals, primals_p)
assert (jnp.abs(primals - primals_p) < 1e-6).all()
for i, (t, t_p) in enumerate(zip(tangents, tangents_p)):
    print(i, t, t_p)
    assert (jnp.abs(t - t_p) < 1e-6).all()

print("Tangent deadarg success")


print(
    timeit.Timer(
        "fwd(in0, in1, din0, din1)",
        globals={"fwd": fwd, "in0": in0, "in1": in1, "din0": din0, "din1": din1},
    ).timeit()
)
print(
    timeit.Timer(
        "fwd_plain(in0, in1, din0, din1)",
        globals={
            "fwd_plain": fwd_plain,
            "in0": in0,
            "in1": in1,
            "din0": din0,
            "din1": din1,
        },
    ).timeit()
)


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


dout = jnp.array([500.0, 700.0, 110.0])

primals, grads = rev(in0, in1, dout)
# TODO enzyme will in place 0 the gradient inputs, which may not be expected
print(dout)
dout = jnp.array([500.0, 700.0, 110.0])
primals_p, grads_p = rev_plain(in0, in1, dout)

assert (jnp.abs(primals - primals_p) < 1e-6).all()
for g, g_p in zip(grads, grads_p):
    print(i, g, g_p)
    assert (jnp.abs(g - g_p) < 1e-6).all()

print("Gradient success")


@jax.jit
def rev2(in0, in1, in2, dout):
    primals, f_vjp = jax.vjp(add_two, in0, in1, in2)
    grads = f_vjp(dout)
    return primals, grads


@jax.jit
def rev2_plain(in0, in1, in2, dout):
    primals, f_vjp = jax.vjp(add_two_plain, in0, in1, in2)
    grads = f_vjp(dout)
    return primals, grads


dout = jnp.array([500.0, 700.0, 110.0])
primals, grads = rev2(in0, in1, in2, dout)
# TODO enzyme will in place 0 the gradient inputs, which may not be expected
print(dout)
dout = jnp.array([500.0, 700.0, 110.0])
primals_p, grads_p = rev2_plain(in0, in1, in2, dout)

assert (jnp.abs(primals - primals_p) < 1e-6).all()
for g, g_p in zip(grads, grads_p):
    print(i, g, g_p)
    assert (jnp.abs(g - g_p) < 1e-6).all()

print("Gradient deadarg success")

print(
    timeit.Timer(
        "rev(in0, in1, dout)",
        globals={"rev": rev, "in0": in0, "in1": in1, "dout": dout},
    ).timeit()
)
print(
    timeit.Timer(
        "rev_plain(in0, in1, dout)",
        globals={"rev_plain": rev_plain, "in0": in0, "in1": in1, "dout": dout},
    ).timeit()
)

x = jnp.array(range(50), dtype=jnp.float32)
dx = jnp.array([i * i for i in range(50)], dtype=jnp.float32)


@enzyme_jax_ir()
def esum(x):
    return jnp.sum(x)


eres = esum(x)
print(eres)
assert jnp.abs(eres - 50 * 49 / 2) < 1e-6


@jax.jit
def sumfwd(in0, din0):
    return jax.jvp(esum, (in0,), (din0,))


primals, tangents = sumfwd(x, dx)
print(primals, tangents)
assert jnp.abs(primals - 50 * 49 / 2) < 1e-6
assert jnp.abs(tangents - 50 * 49 * 99 / 6) < 1e-6


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
assert jnp.abs(primals - 50 * 49 / 2) < 1e-6
assert (jnp.abs(grads[0] - 1) < 1e-6).all()


@enzyme_jax_ir()
def ecache(x):
    return x * x[0]


@jax.jit
def cacherev(in0, din0):
    primals, f_vjp = jax.vjp(ecache, in0)
    grads = f_vjp(din0)
    return grads


dim = 288

x = jnp.array(range(dim), dtype=jnp.float32)
dx = jnp.array(range(dim), dtype=jnp.float32)

grads = cacherev(x, dx)
assert jnp.abs(grads[0][0] - 287 * 288 * (2 * 287 + 1) / 6) < 1e-6
assert (jnp.abs(grads[0][1:]) < 1e-6).all()
