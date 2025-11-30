from absl.testing import absltest
from test_utils import EnzymeJaxTest
import jax.numpy as jnp
import jax


class BlackJAXNUTS(EnzymeJaxTest):
    """
    BlackJAX NUTS (No-U-Turn Sampler) benchmark.

    This tests automatic differentiation of a full MCMC step including:
    - Hamiltonian dynamics simulation
    - Tree building algorithm
    - Gradient computations for the target distribution
    - Acceptance probability calculations
    """
    def setUp(self):
        try:
            import blackjax
        except ImportError:
            self.skipTest("BlackJAX not available")

        # Initial state
        dim = 10

        # Define target distribution (multivariate normal)
        def logdensity_fn(x):
            """Log density of a 10D multivariate normal."""
            return -0.5 * jnp.sum(x**2)

        # Set up NUTS algorithm (newer BlackJAX API)
        nuts = blackjax.nuts(logdensity_fn, step_size=0.1, inverse_mass_matrix=jnp.eye(dim))
        initial_position = jnp.ones(dim)
        initial_state = nuts.init(initial_position)
        key = jax.random.PRNGKey(42)

        # Function to benchmark: single NUTS step (meaningful MCMC)
        def nuts_step_wrapper(position):
            """Wrapper to avoid PRNG key tangent issues."""
            # Use a fixed key inside the function
            key = jax.random.PRNGKey(42)
            state = nuts.init(position)
            new_state, info = nuts.step(key, state)
            return new_state.position

        self.ins = [initial_position]
        self.dins = [jnp.ones_like(initial_position)]
        self.douts = nuts_step_wrapper(*self.ins)

        self.fn = nuts_step_wrapper
        self.name = "blackjax_nuts"
        self.count = 50  # NUTS steps are expensive, use fewer iterations

        # MCMC algorithms use gradients heavily - test AD performance and correctness
        # Tolerances for float32 (JAX default): looser due to stochastic MCMC + numerical differences
        self.atol = 1e-5  # Suitable for float32, stricter than 1e-3 for float16
        self.rtol = 1e-5


class BlackJAXHMC(EnzymeJaxTest):
    """
    BlackJAX Hamiltonian Monte Carlo benchmark.

    Tests gradient-based MCMC with fixed number of leapfrog steps.
    This is simpler than NUTS but still representative of gradient-based sampling.
    """
    def setUp(self):
        try:
            import blackjax
        except ImportError:
            self.skipTest("BlackJAX not available")

        # Target distribution
        def logdensity_fn(x):
            """Log density of multivariate normal (simplified to avoid linalg)."""
            # Use a simple form to avoid matrix operations
            return -0.5 * (x[0]**2 + x[1]**2 + 0.5 * x[0] * x[1])

        # HMC with fixed parameters (newer BlackJAX API)
        hmc = blackjax.hmc(logdensity_fn, step_size=0.1, num_integration_steps=10, inverse_mass_matrix=jnp.eye(2))

        # Initial state
        initial_position = jnp.array([1.0, 1.0])
        initial_state = hmc.init(initial_position)
        key = jax.random.PRNGKey(123)

        def hmc_step_wrapper(position):
            """Single HMC step wrapper."""
            key = jax.random.PRNGKey(123)
            state = hmc.init(position)
            new_state, info = hmc.step(key, state)
            return new_state.position

        # Set up for EnzymeJaxTest
        self.ins = [initial_position]
        self.dins = [jnp.ones_like(initial_position)]
        self.douts = hmc_step_wrapper(*self.ins)

        self.fn = hmc_step_wrapper
        self.name = "blackjax_hmc"
        self.count = 100  # HMC is faster than NUTS

        # MCMC algorithms use gradients heavily - test AD performance and correctness
        # Tolerances for float32 (JAX default): looser due to stochastic MCMC + numerical differences
        self.atol = 1e-5  # Suitable for float32, stricter than 1e-3 for float16
        self.rtol = 1e-5


if __name__ == "__main__":
    from test_utils import fix_paths
    fix_paths()
    absltest.main()