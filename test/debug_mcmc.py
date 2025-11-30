from absl.testing import absltest
from test_utils import EnzymeJaxTest
import jax.numpy as jnp
import jax


class DebugMCMC(EnzymeJaxTest):
    """Debug why MCMC results diverge between JAX and Enzyme-JAX."""

    def setUp(self):
        # Simple deterministic function - no randomness
        def simple_grad_step(x):
            """Simple gradient step - should be deterministic."""
            def logpdf(y):
                return -0.5 * jnp.sum(y**2)

            grad = jax.grad(logpdf)(x)
            return x + 0.1 * grad

        dim = 3  # Small for easy debugging
        initial_x = jnp.ones(dim)

        self.ins = [initial_x]
        self.dins = [jnp.ones_like(initial_x)]
        self.douts = simple_grad_step(*self.ins)

        self.fn = simple_grad_step
        self.name = "debug_grad"
        self.count = 100

        # Tight tolerance - this should be identical
        self.atol = 1e-10
        self.rtol = 1e-10


if __name__ == "__main__":
    from test_utils import fix_paths
    fix_paths()
    absltest.main()