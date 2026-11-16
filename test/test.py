from absl.testing import absltest
import jax
import jax.numpy as jnp
from enzyme_ad.jax import cpp_call, enzyme_jax_ir, hlo_call
import test_utils

jax.config.update("jax_platforms", "cpu")

argv = test_utils.argv


class EnzymePipeline(absltest.TestCase):
    def test_pipeline(self):
        def fn(x):
            return x

        x = jnp.ones(3)
        if hasattr(fn, "trace"):
            module = jax.jit(fn).trace(x).lower().compiler_ir(dialect="stablehlo")
            # Only applies if jax-mlir and enzyme-mlir are built on the same version
            # optimize_module(module)
            print(str(module))


class EnzymeJax(absltest.TestCase):
    def test_custom_cpp_kernel(self):
        @jax.jit
        def do_something(ones):
            shape = jax.core.ShapedArray(ones.shape, ones.dtype)
            a, b = cpp_call(
                ones,
                out_shapes=[shape, shape],
                source="""
        template<std::size_t N, std::size_t M>
        void myfn(enzyme::tensor<float, N, M>& out0,
                  enzyme::tensor<float, N, M>& out1,
                  const enzyme::tensor<float, N, M>& in0) {
          for (int j=0; j<N; j++) {
            for (int k=0; k<M; k++) {
                out0[j][k] = in0[j][k] + 42;
            }
          }
          for (int j=0; j<2; j++) {
            for (int k=0; k<3; k++) {
                out1[j][k] = in0[j][k] + 2 * 42;
            }
          }
        }
        """,
                fn="myfn",
                argv=argv,
            )
            c = cpp_call(
                a,
                out_shapes=[jax.core.ShapedArray([4, 4], jnp.float32)],
                source="""
        template<typename T1, typename T2>
        void f(T1& out0, const T2& in1) {
          out0 = 56.0f;
        }
        """,
                argv=argv,
            )
            return a, b, c

        ones = jnp.ones((2, 3), jnp.float32)
        x, y, z = do_something(ones)

        self.assertTrue((x == 43).all())
        self.assertTrue((y == 85).all())
        self.assertTrue((z[0] == 56).all())

        # JVP
        primals, tangents = jax.jvp(do_something, (ones,), (ones,))
        self.assertTrue((primals[0] == 43).all())
        self.assertTrue((primals[1] == 85).all())
        self.assertTrue((primals[2][0] == 56).all())
        self.assertTrue((tangents[0] == 1).all())
        self.assertTrue((tangents[1] == 1).all())
        self.assertTrue((tangents[2][0] == 0).all())

        # VJP
        primals, f_vjp = jax.vjp(do_something, ones)
        (grads,) = f_vjp((x, y, z))
        self.assertTrue((primals[0] == 43).all())
        self.assertTrue((primals[1] == 85).all())
        self.assertTrue((primals[2][0] == 56).all())

        self.assertTrue(
            (
                grads[1]
                == jnp.array(
                    [
                        [128.0, 128.0, 128.0],
                    ]
                )
            ).all()
        )

    def test_enzyme_mlir_jit(self):
        @jax.jit
        @enzyme_jax_ir(argv=argv)
        def add_one(x: jax.Array, y) -> jax.Array:
            return x + 1 + y

        add_one(jnp.array([1.0, 2.0, 3.0]), jnp.array([10.0, 20.0, 30.0]))

        primals, tangents = jax.jvp(
            add_one,
            (jnp.array([1.0, 2.0, 3.0]), jnp.array([10.0, 20.0, 30.0])),
            (jnp.array([0.1, 0.2, 0.3]), jnp.array([50.0, 70.0, 110.0])),
        )
        self.assertTrue(
            (
                primals
                == jnp.array(
                    [
                        [12.0, 23.0, 34.0],
                    ]
                )
            ).all()
        )
        self.assertTrue(
            (
                tangents
                == jnp.array(
                    [
                        [50.1, 70.2, 110.3],
                    ]
                )
            ).all()
        )

        primals, f_vjp = jax.vjp(
            add_one, jnp.array([1.0, 2.0, 3.0]), jnp.array([10.0, 20.0, 30.0])
        )
        grads = f_vjp(jnp.array([500.0, 700.0, 110.0]))
        self.assertTrue(
            (
                primals
                == jnp.array(
                    [
                        [12.0, 23.0, 34.0],
                    ]
                )
            ).all()
        )
        self.assertTrue(
            (
                grads[0]
                == jnp.array(
                    [
                        [500.0, 700.0, 110.0],
                    ]
                )
            ).all()
        )
        self.assertTrue(
            (
                grads[1]
                == jnp.array(
                    [
                        [500.0, 700.0, 110.0],
                    ]
                )
            ).all()
        )

    def test_large_bool_constant(self):
        # The pipeline folds the mask to an i1 constant of more elements than
        # MLIR prints in full, so the module it hands back to JAX carries the
        # constant in a form JAX's MLIR has to be able to read.
        @jax.jit
        @enzyme_jax_ir(argv=argv)
        def mask_neg(x: jax.Array) -> jax.Array:
            mask = jnp.arange(128) % 3 == 0
            return jnp.where(mask, x, -x)

        x = jnp.arange(128, dtype=jnp.float32)
        expected = jnp.where(jnp.arange(128) % 3 == 0, x, -x)
        self.assertTrue((mask_neg(x) == expected).all())

    def test_hlo_call_multi_out_and_higher_order_ad(self):
        # 1 input, 2 outputs + private helper function
        mlir_dup = """
        module {
          func.func private @helper(%arg0: tensor<3xf32>) -> tensor<3xf32> {
            %0 = stablehlo.multiply %arg0, %arg0 : tensor<3xf32>
            return %0 : tensor<3xf32>
          }
          func.func @main(%arg0: tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>) {
            %0 = func.call @helper(%arg0) : (tensor<3xf32>) -> tensor<3xf32>
            %1 = stablehlo.multiply %0, %arg0 : tensor<3xf32>
            return %0, %1 : tensor<3xf32>, tensor<3xf32>
          }
        }
        """

        @jax.jit
        def fn_dup(x):
            y1, y2 = hlo_call(x, source=mlir_dup, argv=argv)
            return jnp.sum(y1 + y2)

        x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        self.assertTrue(jnp.allclose(fn_dup(x), jnp.sum(x**2 + x**3)))
        g1 = jax.jit(jax.grad(fn_dup))(x)
        self.assertTrue(jnp.allclose(g1, 2.0 * x + 3.0 * x**2))

        # Second-order derivative via jvp of grad
        _, hvp = jax.jit(
            lambda z: jax.jvp(jax.grad(fn_dup), (z,), (jnp.ones_like(z),))
        )(x)
        self.assertTrue(jnp.allclose(hvp, 2.0 + 6.0 * x))

        # Non-differentiable integer argument alongside float argument
        mlir_mixed = """
        module {
          func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<i32>) -> tensor<3xf32> {
            %0 = stablehlo.convert %arg1 : (tensor<i32>) -> tensor<f32>
            %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<3xf32>
            %2 = stablehlo.multiply %arg0, %1 : tensor<3xf32>
            return %2 : tensor<3xf32>
          }
        }
        """

        @jax.jit
        def fn_mixed(x, n):
            (y,) = hlo_call(x, n, source=mlir_mixed, argv=argv)
            return jnp.sum(y**2)

        n = jnp.array(4, dtype=jnp.int32)
        g_mixed = jax.jit(jax.grad(fn_mixed, argnums=0))(x, n)
        self.assertTrue(jnp.allclose(g_mixed, 32.0 * x))


if __name__ == "__main__":
    absltest.main()
