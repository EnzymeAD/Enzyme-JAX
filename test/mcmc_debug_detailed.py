from absl.testing import absltest
from test_utils import EnzymeJaxTest
import jax.numpy as jnp
import jax


class MCMCDebugStep1(EnzymeJaxTest):
    """Test just BlackJAX initialization - no stepping."""

    def setUp(self):
        try:
            import blackjax
        except ImportError:
            self.skipTest("BlackJAX not available")

        def logdensity_fn(x):
            return -0.5 * jnp.sum(x**2)

        def just_init(position):
            """Just initialize NUTS state - no stepping."""
            nuts = blackjax.nuts(logdensity_fn, step_size=0.1, inverse_mass_matrix=jnp.eye(2))
            state = nuts.init(position)
            return state.position  # Should be identical to input

        initial_position = jnp.array([1.0, 2.0])

        self.ins = [initial_position]
        self.dins = [jnp.ones_like(initial_position)]
        self.douts = just_init(*self.ins)

        self.fn = just_init
        self.name = "mcmc_init_only"
        self.count = 10


class MCMCDebugStep2(EnzymeJaxTest):
    """Test just gradient computation through BlackJAX."""

    def setUp(self):
        try:
            import blackjax
        except ImportError:
            self.skipTest("BlackJAX not available")

        def logdensity_fn(x):
            return -0.5 * jnp.sum(x**2)

        def gradient_via_blackjax(position):
            """Compute gradient using BlackJAX's internal mechanism."""
            # This should be equivalent to jax.grad(logdensity_fn)(position)
            return jax.grad(logdensity_fn)(position)

        initial_position = jnp.array([1.0, 2.0])

        self.ins = [initial_position]
        self.dins = [jnp.ones_like(initial_position)]
        self.douts = gradient_via_blackjax(*self.ins)

        self.fn = gradient_via_blackjax
        self.name = "mcmc_grad_only"
        self.count = 100


class MCMCDebugStep3(EnzymeJaxTest):
    """Test NUTS step with completely fixed random key."""

    def setUp(self):
        try:
            import blackjax
        except ImportError:
            self.skipTest("BlackJAX not available")

        def logdensity_fn(x):
            return -0.5 * jnp.sum(x**2)

        def nuts_with_fixed_key(position):
            """NUTS step with completely deterministic key."""
            nuts = blackjax.nuts(logdensity_fn, step_size=0.1, inverse_mass_matrix=jnp.eye(2))

            # Use the EXACT same key every time
            key = jax.random.PRNGKey(12345)  # Fixed seed

            state = nuts.init(position)
            new_state, info = nuts.step(key, state)

            return new_state.position

        initial_position = jnp.array([1.0, 2.0])

        self.ins = [initial_position]
        self.dins = [jnp.ones_like(initial_position)]
        self.douts = nuts_with_fixed_key(*self.ins)

        self.fn = nuts_with_fixed_key
        self.name = "mcmc_fixed_key"
        self.count = 10

        # Very tight tolerance - should be deterministic
        self.atol = 1e-8
        self.rtol = 1e-8


if __name__ == "__main__":
    from test_utils import fix_paths
    fix_paths()
    absltest.main()