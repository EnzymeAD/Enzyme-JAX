from absl.testing import absltest
import jax
import jax.numpy as jnp
from enzyme_ad.jax import checkpoint, cpp_call, enzyme_jax_ir
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

    def test_custom_cpp_kernel_checkpoint(self):
        # A C++ loop under [[enzyme::checkpoint(...)]] (Enzyme's Clang plugin)
        # is checkpointed when Enzyme differentiates the kernel, and its
        # gradient must be that of the same loop without the annotation.
        def kernel(spelling):
            source = """
        template<std::size_t N>
        void steps(enzyme::tensor<float, N>& out,
                   const enzyme::tensor<float, N>& in) {
          float u[N];
          for (int k = 0; k < N; k++)
            u[k] = in[k];
          SPELLING
          for (int t = 0; t < 13; t++)
            for (int k = 0; k < N; k++)
              u[k] = u[k] - 0.1f * u[k] * u[k] * u[k] + 0.05f * in[k];
          for (int k = 0; k < N; k++)
            out[k] = u[k] * u[k];
        }
        """.replace("SPELLING", spelling)

            @jax.jit
            def f(x):
                (y,) = cpp_call(
                    x,
                    out_shapes=[jax.core.ShapedArray(x.shape, x.dtype)],
                    source=source,
                    fn="steps",
                    argv=argv,
                )
                return y

            return f

        x = jnp.array([0.3, 0.5, 0.7], dtype=jnp.float32)
        dy = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        want_y, want_vjp = jax.vjp(kernel(""), x)
        (want,) = want_vjp(dy)
        for spelling in [
            '[[enzyme::checkpoint("binomial", 3)]]',
            '[[enzyme::checkpoint("revolve", 2)]]',
            '[[enzyme::checkpoint("periodic", 4)]]',
            "[[enzyme::checkpoint]]",
            '_Pragma("enzyme checkpoint(\\"binomial\\", 2)")',
        ]:
            y, f_vjp = jax.vjp(kernel(spelling), x)
            (grad,) = f_vjp(dy)
            self.assertTrue((y == want_y).all(), spelling)
            self.assertTrue(jnp.allclose(grad, want, rtol=1e-6), spelling)

    def test_enzyme_mlir_checkpoint(self):
        # A loop run in enzyme_ad.jax.checkpoint(schedule, budget) is
        # checkpointed when Enzyme-MLIR differentiates it, for every schedule,
        # and its gradient must be that of JAX's own AD of the plain loop.
        def run(x, schedule, budget):
            def step(i, u):
                return 0.9 * jnp.sin(u) + 0.1 * x * u

            with checkpoint(schedule, budget):
                u = jax.lax.fori_loop(0, 13, step, x)
            return jnp.sum(u * u)

        x = jnp.array([0.3, 0.5, 0.7])
        want_y, want_vjp = jax.vjp(lambda x: run(x, "none", 0), x)
        (want,) = want_vjp(jnp.float32(1.0))
        for schedule, budget in [
            ("binomial", 3),
            ("revolve", 0),
            ("periodic", 4),
            ("store_all", 0),
        ]:

            @jax.jit
            @enzyme_jax_ir(argv=argv)
            def f(x):
                return run(x, schedule, budget)

            y, f_vjp = jax.vjp(f, x)
            (grad,) = f_vjp(jnp.float32(1.0))
            self.assertTrue(jnp.allclose(y, want_y, rtol=1e-6), schedule)
            self.assertTrue(jnp.allclose(grad, want, rtol=1e-5), schedule)

        with self.assertRaises(ValueError):
            checkpoint("fastest", 4)
        with self.assertRaises(ValueError):
            checkpoint("binomial", -1)

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


if __name__ == "__main__":
    absltest.main()
