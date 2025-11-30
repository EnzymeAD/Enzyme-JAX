from absl.testing import absltest
from test_utils import EnzymeJaxTest
import jax.numpy as jnp
import jax


class SimpleBlackJAXGradient(EnzymeJaxTest):
    """
    Simplest possible BlackJAX test - just compute a gradient.

    This isolates whether the issue is with BlackJAX itself
    or with the MCMC stepping logic.
    """
    def setUp(self):
        try:
            import blackjax
        except ImportError:
            self.skipTest("BlackJAX not available")

        def simple_logdensity(x):
            """Dead simple target: just a quadratic."""
            return -0.5 * jnp.sum(x**2)

        def compute_gradient(x):
            """Just compute gradient of log density - no MCMC stepping."""
            return jax.grad(simple_logdensity)(x)

        initial_position = jnp.array([1.0, 2.0])

        self.ins = [initial_position]
        self.dins = [jnp.ones_like(initial_position)]
        self.douts = compute_gradient(initial_position)

        self.fn = compute_gradient
        self.name = "blackjax_gradient_only"
        self.count = 100

        # Standard tolerances
        self.atol = 1e-6
        self.rtol = 1e-6


class SimpleBlackJAXHMC(EnzymeJaxTest):
    """
    Next step: actual BlackJAX HMC step.

    Since gradient computation works, let's try a single HMC step.
    """
    def setUp(self):
        try:
            import blackjax
        except ImportError:
            self.skipTest("BlackJAX not available")

        def simple_logdensity(x):
            """Simple target distribution."""
            return -0.5 * jnp.sum(x**2)

        # Set up BlackJAX HMC with simple parameters
        hmc = blackjax.hmc(simple_logdensity, step_size=0.1, num_integration_steps=5,
                          inverse_mass_matrix=jnp.eye(2))

        def blackjax_hmc_step(position):
            """Single BlackJAX HMC step."""
            key = jax.random.PRNGKey(42)  # Fixed key
            state = hmc.init(position)
            new_state, info = hmc.step(key, state)
            return new_state.position

        initial_position = jnp.array([1.0, 2.0])

        self.ins = [initial_position]
        self.dins = [jnp.ones_like(initial_position)]
        self.douts = blackjax_hmc_step(initial_position)

        self.fn = blackjax_hmc_step
        self.name = "simple_blackjax_hmc"
        self.count = 50

        # Looser tolerances for MCMC with fixed key
        self.atol = 1e-5
        self.rtol = 1e-5


class SimpleBlackJAXLogisticRegression(EnzymeJaxTest):
    """
    More meaningful: BlackJAX on a simple logistic regression.

    Simplified to avoid dynamic_update_slice issues.
    """
    def setUp(self):
        try:
            import blackjax
        except ImportError:
            self.skipTest("BlackJAX not available")

        # Even simpler setup to avoid dynamic updates
        n_features = 2  # Smaller size

        # Fixed simple data (avoid random generation that might cause issues)
        X = jnp.array([[1.0, 0.5], [0.8, -0.3], [-0.5, 1.2], [0.2, 0.9]])
        y = jnp.array([1.0, 1.0, 0.0, 1.0])

        def logistic_log_posterior(beta):
            """Simplified log posterior."""
            logits = X @ beta
            # Simplified likelihood - avoid potential numerical issues
            probs = jax.nn.sigmoid(logits)
            log_likelihood = jnp.sum(y * jnp.log(probs + 1e-8) + (1 - y) * jnp.log(1 - probs + 1e-8))
            log_prior = -0.5 * jnp.sum(beta**2)
            return log_likelihood + log_prior

        # Try with even simpler HMC parameters
        hmc = blackjax.hmc(logistic_log_posterior, step_size=0.1, num_integration_steps=3,
                          inverse_mass_matrix=jnp.eye(n_features))

        def blackjax_logistic_step(beta):
            """Single HMC step for logistic regression."""
            key = jax.random.PRNGKey(123)
            state = hmc.init(beta)
            new_state, info = hmc.step(key, state)
            return new_state.position

        initial_beta = jnp.array([0.1, 0.1])  # Small non-zero start

        self.ins = [initial_beta]
        self.dins = [jnp.ones_like(initial_beta)]
        self.douts = blackjax_logistic_step(initial_beta)

        self.fn = blackjax_logistic_step
        self.name = "blackjax_logistic_regression"
        self.count = 10  # Fewer iterations for debugging

        # Standard tolerances
        self.atol = 1e-5
        self.rtol = 1e-5


if __name__ == "__main__":
    from test_utils import fix_paths
    fix_paths()
    absltest.main()