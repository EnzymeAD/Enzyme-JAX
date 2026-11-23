"""Regression tests for AD requests collected at an outer Enzyme boundary."""

import re
from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from enzyme_ad.jax import enzyme_jax_ir, primitives

jax.config.update("jax_platforms", "cpu")


class DeferredADTest(absltest.TestCase):
    def _run(self, target, values, expected):
        records = []
        original_run = primitives.enzyme_call.run_pass_pipeline

        def record(symbols, source, pipeline):
            result = original_run(symbols, source, pipeline)
            records.append((str(source), pipeline, result[1]))
            return result

        inputs = tuple(jnp.asarray(value, dtype=jnp.float32) for value in values)
        with mock.patch.object(
            primitives.enzyme_call, "run_pass_pipeline", side_effect=record
        ):
            lowered = target.lower(*inputs)
            outer = str(lowered.compiler_ir(dialect="stablehlo"))
            result = lowered.compile()(*inputs)

        np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-6)
        for module in (outer, *(output for _, _, output in records)):
            self.assertNotIn("enzyme.fwddiff", module)
            self.assertNotIn("enzyme.autodiff", module)
            self.assertNotIn("__enzyme_deferred_ad", module)
        return records

    def _assert_collected(self, records, operation):
        # Both derivative requests must reach one complete outer module.
        self.assertLen(records, 1)
        source, pipeline, _ = records[0]
        self.assertEqual(source.count(operation), 2)
        callees = re.findall(r"fn = @([^,\s}]+)", source)
        self.assertLen(callees, 2)
        self.assertEqual(callees[0], callees[1])
        self.assertEqual(source.count(f"func.func private @{callees[0]}("), 1)
        self.assertNotIn("__enzyme_deferred_ad", source)
        self.assertLess(pipeline.index("canonicalize,cse"), pipeline.index(",enzyme,"))

    @staticmethod
    def _jvp_body():
        @enzyme_jax_ir(inner_jit=False)
        def residual(x, y):
            return 0.5 * jnp.sum((x * y) ** 2)

        def body(y, x, dx1, dx2):
            dy = jnp.zeros_like(y)
            _, tangent1 = jax.jvp(residual, (x, y), (dx1, dy))
            _, tangent2 = jax.jvp(residual, (x, y), (dx2, dy))
            return tangent1, tangent2

        return body

    def test_two_jvps_share_primal_and_run_ad_optimization(self):
        target = enzyme_jax_ir(
            defer_ad=True,
            ad_optimization_passes="canonicalize,cse,enzyme-batch",
        )(self._jvp_body())
        records = self._run(target, (3, 2, 1, 2), (18, 36))
        self._assert_collected(records, "enzyme.fwddiff")
        pipeline = records[0][1]
        self.assertLess(pipeline.index("enzyme-batch"), pipeline.index(",enzyme,"))

    def test_two_vjps_accumulate_shared_input_gradients(self):
        @enzyme_jax_ir(inner_jit=False)
        def two_outputs(x, y):
            y2 = y * y
            return x * x + y2, y2

        @enzyme_jax_ir(defer_ad=True)
        def target(x, y, a1, b1, a2, b2):
            _, pullback = jax.vjp(two_outputs, x, y)
            gx1, gy1 = pullback((a1, b1))
            gx2, gy2 = pullback((a2, b2))
            return gx1, gy1, gx2, gy2

        records = self._run(target, (2, 3, 5, 7, 2, 3), (20, 72, 8, 30))
        self._assert_collected(records, "enzyme.autodiff")

    def test_inactive_and_unused_arguments(self):
        @enzyme_jax_ir(inner_jit=False)
        def with_unused(x, y, unused):
            return x * x + y

        @enzyme_jax_ir(defer_ad=True)
        def target(x, y, unused, dx1, dx2):
            def only_x(z):
                return with_unused(z, y, unused)

            _, tangent1 = jax.jvp(only_x, (x,), (dx1,))
            _, tangent2 = jax.jvp(only_x, (x,), (dx2,))
            return tangent1, tangent2

        records = self._run(target, (2, 3, 99, 1, 2), (4, 8))
        self._assert_collected(records, "enzyme.fwddiff")
        self.assertIn("#enzyme.activity<enzyme_const>", records[0][0])

    def test_deferred_lowering_does_not_leak_into_eager_cache(self):
        body = self._jvp_body()
        eager = enzyme_jax_ir()(body)

        def assert_eager(records):
            for source, _, _ in records:
                self.assertNotIn("enzyme.fwddiff", source)
                self.assertNotIn("enzyme.autodiff", source)
                self.assertNotIn("__enzyme_deferred_ad", source)

        assert_eager(self._run(eager, (3, 2, 1, 2), (18, 36)))
        deferred = enzyme_jax_ir(defer_ad=True)(body)
        self._assert_collected(
            self._run(deferred, (3, 2, 1, 2), (18, 36)), "enzyme.fwddiff"
        )
        eager.clear_cache()
        assert_eager(self._run(eager, (3, 2, 1, 2), (18, 36)))
        assert_eager(self._run(eager, ([3], [2], [1], [2]), (18, 36)))

    def test_differentiating_collection_boundary_is_rejected(self):
        @enzyme_jax_ir(defer_ad=True)
        def target(x):
            return x * x

        with self.assertRaisesRegex(NotImplementedError, "collection boundary"):
            jax.grad(target)(jnp.float32(2))


if __name__ == "__main__":
    absltest.main()
