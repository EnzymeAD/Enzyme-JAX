from absl.testing import absltest
from test_utils import EnzymeJaxTest
import jax.numpy as jnp
import jax


class SimpleHMC(EnzymeJaxTest):
    """
    Simple HMC implementation - more robust than BlackJAX NUTS.

    This implements a basic leapfrog integrator which should be
    more numerically stable and deterministic.
    """
    def setUp(self):
        def logdensity_fn(x):
            """Simple target distribution."""
            return -0.5 * jnp.sum(x**2)

        def leapfrog_step(position, momentum, step_size, num_steps):
            """Basic leapfrog integrator - the core of HMC."""

            def single_leapfrog(pos, mom):
                # Half step for momentum
                grad = jax.grad(logdensity_fn)(pos)
                mom_half = mom + 0.5 * step_size * grad

                # Full step for position
                pos_new = pos + step_size * mom_half

                # Half step for momentum
                grad_new = jax.grad(logdensity_fn)(pos_new)
                mom_new = mom_half + 0.5 * step_size * grad_new

                return pos_new, mom_new

            # Apply leapfrog steps
            pos, mom = position, momentum
            for _ in range(num_steps):
                pos, mom = single_leapfrog(pos, mom)

            return pos, mom

        def simple_hmc_step(position):
            """Simple HMC step - deterministic."""
            # Fixed momentum (instead of sampling)
            momentum = jnp.array([0.5, -0.3])  # Fixed, not random

            step_size = 0.1
            num_steps = 5

            final_pos, final_mom = leapfrog_step(position, momentum, step_size, num_steps)

            # For simplicity, always "accept" the proposal
            return final_pos

        initial_position = jnp.array([1.0, 2.0])

        self.ins = [initial_position]
        self.dins = [jnp.ones_like(initial_position)]
        self.douts = simple_hmc_step(*self.ins)

        self.fn = simple_hmc_step
        self.name = "simple_hmc"
        self.count = 100

        # Allow for normal floating-point differences
        self.atol = 1e-6
        self.rtol = 1e-6


class GradientBenchmark(EnzymeJaxTest):
    """
    Just benchmark gradient computation - the core of MCMC.

    This is what MCMC algorithms spend most time on anyway.
    """
    def setUp(self):
        def complex_logdensity(x):
            """More complex but stable target function."""
            # Simpler quadratic with cross-terms (stable)
            return -0.5 * (x[0]**2 + 2*x[1]**2 + x[0]*x[1])

        def gradient_computation(position):
            """Multiple gradient evaluations like in real MCMC."""
            grad1 = jax.grad(complex_logdensity)(position)

            # Simulate multiple evaluations like in leapfrog
            pos2 = position + 0.1 * grad1
            grad2 = jax.grad(complex_logdensity)(pos2)

            pos3 = pos2 + 0.1 * grad2
            grad3 = jax.grad(complex_logdensity)(pos3)

            return grad3

        initial_position = jnp.array([0.5, 1.5])

        self.ins = [initial_position]
        self.dins = [jnp.ones_like(initial_position)]
        self.douts = gradient_computation(*self.ins)

        self.fn = gradient_computation
        self.name = "mcmc_gradients"
        self.count = 500  # Many iterations since it's fast


if __name__ == "__main__":
    from test_utils import fix_paths
    fix_paths()
    absltest.main()
