from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from enzyme_ad.jax import enzyme_jax_ir, julia_call, optimization_passes
import test_utils

jax.config.update("jax_platforms", "cpu")

argv = test_utils.argv


class EnzymeJuliaTest(absltest.TestCase):
    def test_basic_array_and_ad(self):
        @jax.jit
        def fn(x):
            (y,) = julia_call(
                x,
                source="""
                function f(x)
                    return x .^ 3 .+ 2.0f0 .* x .+ 1.0f0
                end
                """,
                fn="f",
                argv=argv,
            )
            return y

        x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        y = fn(x)
        expected = x**3 + 2.0 * x + 1.0
        np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

        # Forward-mode AD (JVP)
        dx = jnp.array([0.5, 1.0, -0.5], dtype=jnp.float32)
        primals, tangents = jax.jvp(fn, (x,), (dx,))
        expected_tan = (3.0 * x**2 + 2.0) * dx
        np.testing.assert_allclose(primals, expected, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(tangents, expected_tan, rtol=1e-5, atol=1e-5)

        # Reverse-mode AD (VJP and grad)
        primals_vjp, f_vjp = jax.vjp(fn, x)
        (grads,) = f_vjp(dx)
        np.testing.assert_allclose(primals_vjp, expected, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(grads, expected_tan, rtol=1e-5, atol=1e-5)

        grad_sum = jax.jit(jax.grad(lambda z: jnp.sum(fn(z))))(x)
        np.testing.assert_allclose(
            grad_sum, 3.0 * x**2 + 2.0, rtol=1e-5, atol=1e-5
        )

    def test_multi_input_multi_output(self):
        @jax.jit
        def do_something(a, b):
            out1, out2 = julia_call(
                a,
                b,
                source="""
                function myfn(a, b)
                    return a .+ 2 .* b, a .* b
                end
                """,
                fn="myfn",
                argv=argv,
            )
            return out1, out2

        a = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32)
        o1, o2 = do_something(a, b)
        np.testing.assert_allclose(o1, a + 2 * b, rtol=1e-5)
        np.testing.assert_allclose(o2, a * b, rtol=1e-5)

        # JVP
        da = jnp.ones_like(a)
        db = jnp.full_like(b, 0.5)
        (p1, p2), (t1, t2) = jax.jvp(do_something, (a, b), (da, db))
        np.testing.assert_allclose(p1, a + 2 * b, rtol=1e-5)
        np.testing.assert_allclose(p2, a * b, rtol=1e-5)
        np.testing.assert_allclose(t1, da + 2 * db, rtol=1e-5)
        np.testing.assert_allclose(t2, da * b + a * db, rtol=1e-5)

        # VJP
        (p1, p2), vjp_fn = jax.vjp(do_something, a, b)
        ct1 = jnp.ones_like(a)
        ct2 = jnp.full_like(a, 2.0)
        ga, gb = vjp_fn((ct1, ct2))
        np.testing.assert_allclose(ga, ct1 + ct2 * b, rtol=1e-5)
        np.testing.assert_allclose(gb, 2 * ct1 + ct2 * a, rtol=1e-5)

    def test_identity_and_duplicate_outputs(self):
        @jax.jit
        def fn_id(x):
            (y,) = julia_call(x, source="f(x) = x", argv=argv)
            return y

        x = jnp.array([3.0, 4.0, 5.0], dtype=jnp.float32)
        np.testing.assert_allclose(fn_id(x), x)
        np.testing.assert_allclose(
            jax.grad(lambda z: jnp.sum(fn_id(z) * 2.0))(x),
            jnp.full_like(x, 2.0),
        )

        @jax.jit
        def fn_dup(x, scale):
            y1, y2 = julia_call(
                x,
                scale,
                source="""
                function f(x, scale)
                    y = scale .* x
                    return y, y
                end
                """,
                argv=argv,
            )
            return y1 + y2

        scale = jnp.array(3.0, dtype=jnp.float32)
        np.testing.assert_allclose(fn_dup(x, scale), 6.0 * x, rtol=1e-5)
        np.testing.assert_allclose(
            jax.grad(lambda z: jnp.sum(fn_dup(z, scale)))(x),
            jnp.full_like(x, 6.0),
            rtol=1e-5,
        )

    def test_scalars_and_reductions(self):
        @jax.jit
        def scalar_fn(x, alpha):
            (res,) = julia_call(
                x,
                alpha,
                source="""
                function f(x, alpha)
                    return sum(x .^ 2) * alpha
                end
                """,
                argv=argv,
            )
            return res

        x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        alpha = jnp.array(2.5, dtype=jnp.float32)
        val = scalar_fn(x, alpha)
        np.testing.assert_allclose(val, jnp.sum(x**2) * alpha, rtol=1e-5)

        gx, galpha = jax.grad(scalar_fn, argnums=(0, 1))(x, alpha)
        np.testing.assert_allclose(gx, 2.0 * x * alpha, rtol=1e-5)
        np.testing.assert_allclose(galpha, jnp.sum(x**2), rtol=1e-5)

    def test_matmul_and_reverse_dims(self):
        @jax.jit
        def matmul_default(a, b):
            (c,) = julia_call(
                a,
                b,
                source="f(a, b) = a * b",
                reverse_dims=False,
                argv=argv,
            )
            return c

        a = jnp.arange(6, dtype=jnp.float32).reshape(2, 3) + 1.0
        b = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) + 0.5
        c = matmul_default(a, b)
        np.testing.assert_allclose(c, a @ b, rtol=1e-5)

        ga, gb = jax.grad(lambda x, y: jnp.sum(matmul_default(x, y)), argnums=(0, 1))(
            a, b
        )
        expected_ga, expected_gb = jax.grad(
            lambda x, y: jnp.sum(x @ y), argnums=(0, 1)
        )(a, b)
        np.testing.assert_allclose(ga, expected_ga, rtol=1e-5)
        np.testing.assert_allclose(gb, expected_gb, rtol=1e-5)

        # With reverse_dims=True, a (2, 3) JAX array has Julia size (3, 2) and
        # b (3, 4) JAX array has Julia size (4, 3), so b * a in Julia has Julia
        # size (4, 2), which maps back to JAX shape (2, 4).
        @jax.jit
        def matmul_reversed(a, b):
            (c,) = julia_call(
                a,
                b,
                source="f(a, b) = b * a",
                reverse_dims=True,
                argv=argv,
            )
            return c

        c_rev = matmul_reversed(a, b)
        np.testing.assert_allclose(c_rev, a @ b, rtol=1e-5)

    def test_loops_and_helper_functions(self):
        source = """
        helper(v, k) = v .* Float32(k)

        function accumulate(x)
            acc = zero(x)
            for k in 1:4
                acc = acc .+ helper(x, k)
            end
            return acc
        end
        """

        @jax.jit
        def fn(x):
            (y,) = julia_call(x, source=source, fn="accumulate", argv=argv)
            return y

        x = jnp.array([1.5, -2.0, 0.5], dtype=jnp.float32)
        np.testing.assert_allclose(fn(x), 10.0 * x, rtol=1e-5)
        np.testing.assert_allclose(
            jax.grad(lambda z: jnp.sum(fn(z) ** 2))(x),
            200.0 * x,
            rtol=1e-4,
        )

    def test_higher_order_ad_and_passes(self):
        passes = optimization_passes()

        def cubic(x):
            (y,) = julia_call(
                x,
                source="f(x) = sum(x .^ 3)",
                passes=passes,
                argv=argv,
            )
            return y

        x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        g1 = jax.jit(jax.grad(cubic))(x)
        np.testing.assert_allclose(g1, 3.0 * x**2, rtol=1e-5)

        # JVP with custom optimization passes
        primals, tangents = jax.jit(
            lambda z: jax.jvp(cubic, (z,), (jnp.ones_like(z),))
        )(x)
        np.testing.assert_allclose(primals, jnp.sum(x**3), rtol=1e-5)
        np.testing.assert_allclose(tangents, jnp.sum(3.0 * x**2), rtol=1e-5)

        # Composition with @enzyme_jax_ir
        @jax.jit
        @enzyme_jax_ir(argv=argv)
        def composed(z):
            (w,) = julia_call(z, source="f(z) = sin.(z) .* z", argv=argv)
            return w + 2.0 * z

        np.testing.assert_allclose(composed(x), jnp.sin(x) * x + 2.0 * x, rtol=1e-5)
        grad_composed = jax.jit(jax.grad(lambda z: jnp.sum(composed(z))))(x)
        expected_grad = jnp.cos(x) * x + jnp.sin(x) + 2.0
        np.testing.assert_allclose(grad_composed, expected_grad, rtol=1e-5)

    def test_dtypes_and_non_diff_args(self):
        # Integer arithmetic
        @jax.jit
        def int_fn(x, y):
            (z,) = julia_call(x, y, source="f(x, y) = x .* 3 .+ y", argv=argv)
            return z

        xi = jnp.array([1, 2, 3], dtype=jnp.int32)
        yi = jnp.array([10, 20, 30], dtype=jnp.int32)
        np.testing.assert_array_equal(int_fn(xi, yi), xi * 3 + yi)

        # Mixed float and integer primal
        @jax.jit
        def mixed_fn(x, n):
            (y,) = julia_call(
                x,
                n,
                source="f(x, n) = x .* n",
                argv=argv,
            )
            return jnp.sum(y**2)

        xf = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        n = jnp.array(4, dtype=jnp.int32)
        np.testing.assert_allclose(mixed_fn(xf, n), jnp.sum((xf * 4.0) ** 2), rtol=1e-5)

        # Complex numbers
        @jax.jit
        def complex_fn(z):
            (w,) = julia_call(z, source="f(z) = z .^ 2 .+ z", argv=argv)
            return w

        zc = jnp.array([1.0 + 2.0j, 3.0 - 1.0j], dtype=jnp.complex64)
        np.testing.assert_allclose(complex_fn(zc), zc**2 + zc, rtol=1e-5)

    def test_out_shapes_and_error_handling(self):
        x = jnp.ones((2, 3), dtype=jnp.float32)

        # Matching out_shapes succeeds
        @jax.jit
        def valid_fn(a):
            (b,) = julia_call(
                a,
                source="f(a) = a .+ 1.0f0",
                out_shapes=jax.core.ShapedArray((2, 3), jnp.float32),
                argv=argv,
            )
            return b

        np.testing.assert_allclose(valid_fn(x), x + 1.0, rtol=1e-5)

        # Shape mismatch raises ValueError
        with self.assertRaises(ValueError):
            julia_call(
                x,
                source="f(a) = a .+ 1.0f0",
                out_shapes=[jax.core.ShapedArray((3, 2), jnp.float32)],
                argv=argv,
            )

        # Dtype mismatch raises ValueError
        with self.assertRaises(ValueError):
            julia_call(
                x,
                source="f(a) = a .+ 1.0f0",
                out_shapes=[jax.core.ShapedArray((2, 3), jnp.int32)],
                argv=argv,
            )

        # Output count mismatch raises ValueError
        with self.assertRaises(ValueError):
            julia_call(
                x,
                source="f(a) = (a, a)",
                out_shapes=[jax.core.ShapedArray((2, 3), jnp.float32)],
                argv=argv,
            )

        # Undefined Julia function raises ValueError
        with self.assertRaises(ValueError):
            julia_call(
                x,
                source="f(a) = a",
                fn="nonexistent_fn",
                argv=argv,
            )


if __name__ == "__main__":
    absltest.main()
