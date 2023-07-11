import jax
import jax.numpy as jnp
from enzyme_jax import cpp_call

# @jax.jit
def do_something(ones, twos):
    shape = jax.core.ShapedArray(tuple(3 * s for s in ones.shape), ones.dtype)
    shape2 = jax.core.ShapedArray(tuple(2 * s for s in ones.shape), ones.dtype)
    a, b = cpp_call(ones, twos, out_shapes=[shape, shape2], source="""
    template<std::size_t N3, std::size_t M3, std::size_t N, std::size_t M, std::size_t N2, std::size_t M2, std::size_t N4, std::size_t M4>
    void myfn(enzyme::tensor<float, N, M>& out0, enzyme::tensor<float, N2, M2>& out1, const enzyme::tensor<float, N3, M3>& in0, const enzyme::tensor<float, N4, M4>& in1) {
        for (int j=0; j<N; j++) {
        for (int k=0; k<M; k++) {
            out0[j][k] = in0[0][0] + 42;
        }
        }
        for (int j=0; j<2; j++) {
        for (int k=0; k<2; k++) {
            out1[j][k] = in0[j][k] + 2 * 42;
        }
        }
    }
    """, fn="myfn")
    return a, b

ones = jnp.ones((2, 3), jnp.float32)
twos = jnp.ones((5, 7), jnp.float32)
x, y = jax.jit(do_something)(ones, twos)

# print(x)
# print(y)
# print(z)

# primals, tangents = jax.jvp(do_something, (ones,), (ones,) )
# print(primals)
# print(tangents)


@jax.jit
def f(a, b):
    return jax.vjp(do_something, a, b)



@jax.jit
def g(a, b, x, y):
    primals, f_vjp = jax.vjp(do_something, a, b)
    return primals, f_vjp((x, y))

print(f.lower(ones, twos).compiler_ir(dialect="mhlo"))
print(g.lower(ones, twos, x, y).compiler_ir(dialect="stablehlo"))
print(g.lower(ones, twos, x, y).compiler_ir(dialect="mhlo"))

primals, f_vjp = jax.vjp(jax.jit(do_something), ones, twos)
grads = f_vjp((x, y))
print(primals)
print(grads)


print(jax.jit(f_vjp).lower((x, y)).compiler_ir(dialect="mhlo"))