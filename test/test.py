import jax
import jax.numpy as jnp
from enzyme_jax import cpp_fwd

def do_something(fn):
    @jax.jit
    def do_something_internal(ones):
        shape = jax.core.ShapedArray(ones.shape, ones.dtype)
        a, b = fn(ones, out_shapes=[shape, shape], source="""
        template<std::size_t N, std::size_t M>
        void myfn(enzyme::tensor<float, N, M>& out0, enzyme::tensor<float, N, M>& out1, const enzyme::tensor<float, N, M>& in0) {
            for (int j=0; j<N; j++) {
            for (int k=0; k<M; k++) {
                out0[j][k] = in0[j][k] + 42;
            }
            }
            for (int j=0; j<2; j++) {
            for (int k=0; k<3; k++) {
                out0[j][k] = in0[j][k] + 2 * 42;
            }
            }
        }
        """, fn="myfn")
        c = fn(a, out_shapes=[jax.core.ShapedArray([4, 4], jnp.float32)], source="""
        template<typename T1, typename T2>
        void f(T1& out0, const T2& in1) {
        out0 = 56.0f;
        }
        """)
        return a, b, c
    return do_something_internal

ones = jnp.ones((2, 3), jnp.float32)
x, y, z = do_something(cpp_fwd)(ones)

# print(x)
# print(y)
# print(z)

# primals, tangents = jax.jvp(do_something(cpp_fwd), (ones,), (ones,) )
# print(primals)
# print(tangents)


primals, f_vjp = jax.vjp(do_something(cpp_fwd), ones)
(grads,) = f_vjp((x, y, z))
print(primals)
print(grads)
