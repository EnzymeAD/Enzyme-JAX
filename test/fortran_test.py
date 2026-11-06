"""Tests for Fortran support via fortran_call."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from enzyme_ad.jax import fortran_call

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

REAL_DTYPE_TO_KIND = {
    jnp.float32: "real32",
    jnp.float64: "real64",
}

COMPLEX_DTYPE_TO_KIND = {
    jnp.complex64: "real32",
    jnp.complex128: "real64",
}


class FortranCallTest(parameterized.TestCase):
    @parameterized.parameters(*REAL_DTYPE_TO_KIND.items())
    def test_explicit_shape_and_layout(self, dtype, kind):
        """Tests explicit-shape column-major arrays and AD for f32 and f64."""
        source = f"""
        subroutine scale_add(out_mat, a, b)
          use, intrinsic :: iso_fortran_env, only: {kind}
          implicit none
          real({kind}), intent(out) :: out_mat(3, 2)
          real({kind}), intent(in) :: a(3, 2), b
          integer :: i, j
          do j = 1, 2
            do i = 1, 3
              out_mat(i, j) = a(i, j) * b + real(i + 10 * j, {kind})
            end do
          end do
        end subroutine scale_add
        """

        @jax.jit
        def fn(a, b):
            (res,) = fortran_call(
                a,
                b,
                out_shapes=[jax.core.ShapedArray((2, 3), dtype)],
                source=source,
                fn="scale_add",
            )
            return res

        a = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
        b = jnp.array(2.5, dtype=dtype)
        expected_offset = jnp.array(
            [[11.0, 12.0, 13.0], [21.0, 22.0, 23.0]], dtype=dtype
        )
        expected = a * b + expected_offset

        out = fn(a, b)
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

        da = jnp.ones_like(a) * 0.5
        db = jnp.array(1.5, dtype=dtype)
        primal, tangent = jax.jvp(fn, (a, b), (da, db))
        np.testing.assert_allclose(primal, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(tangent, da * b + a * db, rtol=1e-6, atol=1e-6)

        primal_vjp, vjp_fn = jax.vjp(fn, a, b)
        np.testing.assert_allclose(primal_vjp, expected, rtol=1e-6, atol=1e-6)
        cot = jnp.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]], dtype=dtype)
        grad_a, grad_b = vjp_fn(cot)
        np.testing.assert_allclose(grad_a, cot * b, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(grad_b, jnp.sum(cot * a), rtol=1e-6, atol=1e-6)

    @parameterized.named_parameters(
        ("module_qualified", "my_fortran_mod::poly_2d"),
        ("unqualified", "poly_2d"),
    )
    def test_module_assumed_shape_descriptors(self, proc_name):
        """Tests assumed-shape module subroutines accessed with or without module prefix."""
        source = """
        module my_fortran_mod
          use, intrinsic :: iso_fortran_env, only: real64
          implicit none
        contains
          subroutine poly_2d(y, x)
            real(real64), intent(out) :: y(:, :)
            real(real64), intent(in) :: x(:, :)
            integer :: i, j, n1, n2
            n1 = size(x, 1)
            n2 = size(x, 2)
            do j = 1, n2
              do i = 1, n1
                y(i, j) = x(i, j)**2 + 3.0_real64 * x(i, j) + real(n1 * 10 + n2, real64)
              end do
            end do
          end subroutine poly_2d
        end module my_fortran_mod
        """

        x = jnp.array([[1.0, -2.0, 0.5], [3.0, 1.5, -1.0]], dtype=jnp.float64)
        out_shape = [jax.core.ShapedArray(x.shape, jnp.float64)]

        # Module procedures can be looked up either via "module_name::proc_name"
        # or via unqualified "proc_name" when unambiguous in the source.
        f = jax.jit(
            lambda arr: fortran_call(
                arr, out_shapes=out_shape, source=source, fn=proc_name
            )[0]
        )
        expected = x**2 + 3.0 * x + 32.0
        np.testing.assert_allclose(f(x), expected, rtol=1e-12, atol=1e-12)

        dx = jnp.ones_like(x) * 0.5
        primal, tangent = jax.jvp(f, (x,), (dx,))
        np.testing.assert_allclose(primal, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            tangent, (2.0 * x + 3.0) * dx, rtol=1e-12, atol=1e-12
        )

        primal_vjp, f_vjp = jax.vjp(f, x)
        np.testing.assert_allclose(primal_vjp, expected, rtol=1e-12, atol=1e-12)
        (cotangent,) = f_vjp(jnp.ones_like(x))
        np.testing.assert_allclose(cotangent, 2.0 * x + 3.0, rtol=1e-12, atol=1e-12)

    def test_scalar_returning_fortran_function(self):
        """Tests a standalone (non-module) Fortran function returning a scalar."""
        source = """
        function weighted_sum(x, y) result(res)
          use, intrinsic :: iso_fortran_env, only: real64
          implicit none
          real(real64), intent(in) :: x(:), y(:)
          real(real64) :: res
          integer :: i
          res = 0.0_real64
          do i = 1, size(x)
            res = res + sin(x(i)) * y(i)**2
          end do
        end function weighted_sum
        """

        @jax.jit
        def fn(x, y):
            (res,) = fortran_call(
                x,
                y,
                out_shapes=[jax.core.ShapedArray((), jnp.float64)],
                source=source,
                fn="weighted_sum",
            )
            return res

        x = jnp.array([0.3, 0.7, 1.1], dtype=jnp.float64)
        y = jnp.array([2.0, -1.5, 0.5], dtype=jnp.float64)
        expected = jnp.sum(jnp.sin(x) * y**2)

        np.testing.assert_allclose(fn(x, y), expected, rtol=1e-12)

        dx = jnp.array([0.1, -0.2, 0.3], dtype=jnp.float64)
        dy = jnp.array([0.4, 0.5, -0.6], dtype=jnp.float64)
        primal, tangent = jax.jvp(fn, (x, y), (dx, dy))
        np.testing.assert_allclose(primal, expected, rtol=1e-12)
        np.testing.assert_allclose(
            tangent,
            jnp.sum(jnp.cos(x) * dx * y**2 + 2.0 * jnp.sin(x) * y * dy),
            rtol=1e-12,
        )

        gx, gy = jax.grad(fn, argnums=(0, 1))(x, y)
        np.testing.assert_allclose(gx, jnp.cos(x) * y**2, rtol=1e-12)
        np.testing.assert_allclose(gy, 2.0 * jnp.sin(x) * y, rtol=1e-12)

    def test_bind_c_subroutine_and_value_scalars(self):
        """Tests calling a Fortran subroutine exported via bind(C, name=...) with VALUE scalars."""
        source = """
        module bind_c_mod
          use, intrinsic :: iso_c_binding, only: c_int, c_double
          implicit none
        contains
          subroutine axpy_fortran(y, a, x, n) bind(C, name="axpy_bind_c")
            integer(c_int), value :: n
            real(c_double), intent(out) :: y(n)
            real(c_double), value :: a
            real(c_double), intent(in) :: x(n)
            integer :: i
            do i = 1, n
              y(i) = a * x(i) + 1.0_c_double
            end do
          end subroutine axpy_fortran
        end module bind_c_mod
        """

        @jax.jit
        def fn(a, x, n):
            # Look up the Fortran subroutine directly via its exported bind(C) label.
            (y,) = fortran_call(
                a,
                x,
                n,
                out_shapes=[jax.core.ShapedArray(x.shape, jnp.float64)],
                source=source,
                fn="axpy_bind_c",
            )
            return y

        a = jnp.array(3.0, dtype=jnp.float64)
        x = jnp.array([1.0, 2.0, 4.0, 8.0], dtype=jnp.float64)
        n = jnp.int32(4)

        np.testing.assert_allclose(fn(a, x, n), a * x + 1.0, rtol=1e-12)

        primal, tangent = jax.jvp(
            lambda a_, x_: fn(a_, x_, n),
            (a, x),
            (jnp.array(2.0, dtype=jnp.float64), jnp.ones_like(x)),
        )
        np.testing.assert_allclose(primal, a * x + 1.0, rtol=1e-12)
        np.testing.assert_allclose(tangent, 2.0 * x + a, rtol=1e-12)

        grad_a, grad_x = jax.grad(
            lambda a_, x_: jnp.sum(fn(a_, x_, n)), argnums=(0, 1)
        )(a, x)
        np.testing.assert_allclose(grad_a, jnp.sum(x), rtol=1e-12)
        np.testing.assert_allclose(grad_x, jnp.full_like(x, a), rtol=1e-12)

    def test_iso_fortran_env_and_intrinsics(self):
        """Tests array intrinsics (matmul, transpose, dot_product, sum, exp)."""
        source = """
        module math_intrinsics_mod
          use, intrinsic :: iso_fortran_env, only: real64, int32
          implicit none
        contains
          subroutine matvec_energy(energy, mat, vec, n)
            real(real64), intent(out) :: energy
            real(real64), intent(in) :: mat(:, :), vec(:)
            integer(int32), intent(in) :: n
            real(real64) :: prod(n)
            prod = matmul(transpose(mat), vec)
            energy = dot_product(prod, prod) + sum(exp(vec * 0.1_real64))
          end subroutine matvec_energy
        end module math_intrinsics_mod
        """

        @jax.jit
        def fn(mat, vec):
            (energy,) = fortran_call(
                mat,
                vec,
                jnp.int32(mat.shape[0]),
                out_shapes=[jax.core.ShapedArray((), jnp.float64)],
                source=source,
                fn="math_intrinsics_mod::matvec_energy",
            )
            return energy

        def ref_fn(mat, vec):
            prod = mat @ vec
            return jnp.dot(prod, prod) + jnp.sum(jnp.exp(vec * 0.1))

        mat = jnp.array([[1.0, 2.0], [3.0, -1.0], [0.5, 1.5]], dtype=jnp.float64)
        vec = jnp.array([2.0, -0.5], dtype=jnp.float64)

        np.testing.assert_allclose(fn(mat, vec), ref_fn(mat, vec), rtol=1e-12)

        dmat = jnp.ones_like(mat) * 0.25
        dvec = jnp.array([0.5, -0.25], dtype=jnp.float64)
        p1, t1 = jax.jvp(fn, (mat, vec), (dmat, dvec))
        p2, t2 = jax.jvp(ref_fn, (mat, vec), (dmat, dvec))
        np.testing.assert_allclose(p1, p2, rtol=1e-12)
        np.testing.assert_allclose(t1, t2, rtol=1e-12)

        g1 = jax.grad(fn, argnums=(0, 1))(mat, vec)
        g2 = jax.grad(ref_fn, argnums=(0, 1))(mat, vec)
        np.testing.assert_allclose(g1[0], g2[0], rtol=1e-12)
        np.testing.assert_allclose(g1[1], g2[1], rtol=1e-12)

    def test_multi_output_partial_cotangent(self):
        """Tests reverse-mode AD when only a subset of subroutine outputs is used."""
        source = """
        subroutine multi_out(out_a, out_b, x)
          use, intrinsic :: iso_fortran_env, only: real64
          implicit none
          real(real64), intent(out) :: out_a(3), out_b(3)
          real(real64), intent(in) :: x(3)
          out_a = x**2
          out_b = cos(x)
        end subroutine multi_out
        """

        @jax.jit
        def only_first_out(x):
            a, _b = fortran_call(
                x,
                out_shapes=[
                    jax.core.ShapedArray((3,), jnp.float64),
                    jax.core.ShapedArray((3,), jnp.float64),
                ],
                source=source,
                fn="multi_out",
            )
            return jnp.sum(a)

        x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
        np.testing.assert_allclose(jax.grad(only_first_out)(x), 2.0 * x, rtol=1e-12)

    @parameterized.parameters(*COMPLEX_DTYPE_TO_KIND.items())
    def test_complex_array_and_scalar(self, dtype, kind):
        """Tests complex64 and complex128 array and scalar arguments."""
        source = f"""
        subroutine cscale(y, z, alpha)
          use, intrinsic :: iso_fortran_env, only: {kind}
          implicit none
          complex({kind}), intent(out) :: y(2)
          complex({kind}), intent(in) :: z(2), alpha
          y = alpha * z
        end subroutine cscale
        """
        z = jnp.array([1.0 + 2.0j, -3.0 + 0.5j], dtype=dtype)
        alpha = jnp.array(0.5 - 1.5j, dtype=dtype)
        (cy,) = jax.jit(
            lambda z_, a_: fortran_call(
                z_,
                a_,
                out_shapes=[jax.core.ShapedArray((2,), dtype)],
                source=source,
                fn="cscale",
            )
        )(z, alpha)
        rtol = 1e-6 if dtype == jnp.complex64 else 1e-12
        np.testing.assert_allclose(cy, alpha * z, rtol=rtol)

    def test_fixed_form(self):
        """Tests fixed-form Fortran compilation via argv=('-ffixed-form',)."""
        fixed_source = (
            "      SUBROUTINE FIXEDSUB(Y, X)\n"
            "      REAL*8 Y(2), X(2)\n"
            "      Y(1) = X(1) + 2.0D0\n"
            "      Y(2) = X(2) * 3.0D0\n"
            "      END SUBROUTINE\n"
        )
        x = jnp.array([4.0, 5.0], dtype=jnp.float64)
        fn = jax.jit(
            lambda arr: fortran_call(
                arr,
                out_shapes=[jax.core.ShapedArray((2,), jnp.float64)],
                source=fixed_source,
                fn="fixedsub",
                argv=("-ffixed-form",),
            )[0]
        )
        np.testing.assert_allclose(fn(x), jnp.array([6.0, 15.0]), rtol=1e-12)
        np.testing.assert_allclose(
            jax.grad(lambda arr: jnp.sum(fn(arr)))(x),
            jnp.array([1.0, 3.0]),
            rtol=1e-12,
        )

    def test_syntax_error_reporting(self):
        """Tests that invalid Fortran syntax raises ValueError."""
        x = jnp.array([4.0, 5.0], dtype=jnp.float64)
        with self.assertRaises(ValueError):
            jax.jit(
                lambda arr: fortran_call(
                    arr,
                    out_shapes=[jax.core.ShapedArray((2,), jnp.float64)],
                    source="subroutine broken(\nend subroutine",
                    fn="broken",
                )
            )(x)

    def test_missing_procedure_error_reporting(self):
        """Tests that referencing a nonexistent Fortran procedure raises ValueError."""
        source = """
        subroutine valid_sub(y, x)
          use, intrinsic :: iso_fortran_env, only: real64
          implicit none
          real(real64), intent(out) :: y(2)
          real(real64), intent(in) :: x(2)
          y = x
        end subroutine valid_sub
        """
        x = jnp.array([4.0, 5.0], dtype=jnp.float64)
        with self.assertRaises(ValueError):
            jax.jit(
                lambda arr: fortran_call(
                    arr,
                    out_shapes=[jax.core.ShapedArray((2,), jnp.float64)],
                    source=source,
                    fn="nonexistent_proc",
                )
            )(x)

    def test_type_mismatch_error_reporting(self):
        """Tests that mismatched JAX and Fortran dummy parameter types raise ValueError."""
        source = """
        subroutine real64_sub(y, x)
          use, intrinsic :: iso_fortran_env, only: real64
          implicit none
          real(real64), intent(out) :: y(2)
          real(real64), intent(in) :: x(2)
          y = x
        end subroutine real64_sub
        """
        x = jnp.array([4.0, 5.0], dtype=jnp.float64)
        with self.assertRaises(ValueError):
            jax.jit(
                lambda arr: fortran_call(
                    arr,
                    out_shapes=[jax.core.ShapedArray((2,), jnp.float32)],
                    source=source,
                    fn="real64_sub",
                )
            )(x)


if __name__ == "__main__":
    absltest.main()
