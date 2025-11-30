from absl.testing import absltest
from test_utils import EnzymeJaxTest
import jax.numpy as jnp
import jax


class SimpleMCMC(EnzymeJaxTest):
    """
    Simple MCMC-like benchmark that demonstrates the EnzymeJaxTest pattern.
    This tests automatic differentiation of a typical MCMC step without
    requiring external dependencies like BlackJAX.
    """
    def setUp(self):
        # Define a simple log probability density function (multivariate normal)
        def logpdf(x):
            """Log probability of multivariate normal distribution."""
            return -0.5 * jnp.sum(x**2)

        # Simple Metropolis-Hastings step (without key for simplicity)
        def mcmc_step(x, log_prob):
            """Single MCMC step with gradient computation."""
            # Compute gradient of log probability
            grad_log_prob = jax.grad(logpdf)(x)

            # Simple proposal step (like in Hamiltonian Monte Carlo)
            proposal = x + 0.1 * grad_log_prob

            # Compute acceptance probability (simplified)
            new_log_prob = logpdf(proposal)

            # Return new state (simplified, always accept for deterministic benchmark)
            return proposal, new_log_prob

        # Set up benchmark inputs
        dim = 10  # 10-dimensional problem
        initial_x = jnp.ones(dim)
        initial_log_prob = logpdf(initial_x)

        # EnzymeJaxTest requires these specific attributes
        self.ins = [initial_x, initial_log_prob]
        self.dins = [
            jnp.ones_like(initial_x),  # tangent for x
            jnp.array(1.0),           # tangent for log_prob
        ]

        # Expected output structure
        new_x, new_log_prob = mcmc_step(*self.ins)
        self.douts = (new_x, new_log_prob)

        self.fn = mcmc_step
        self.name = "simple_mcmc"
        self.count = 1000  # Number of timing iterations


class VectorizedMCMC(EnzymeJaxTest):
    """
    Vectorized MCMC benchmark to test batched operations.
    """
    def setUp(self):
        def logpdf_batch(x_batch):
            """Batched log probability computation."""
            return -0.5 * jnp.sum(x_batch**2, axis=-1)

        def mcmc_step_batch(x_batch):
            """Vectorized MCMC step."""
            grad_batch = jax.vmap(jax.grad(lambda x: -0.5 * jnp.sum(x**2)))(x_batch)
            return x_batch + 0.1 * grad_batch

        # Set up batch of MCMC chains
        batch_size = 100
        dim = 5
        x_batch = jnp.ones((batch_size, dim))

        self.ins = [x_batch]
        self.dins = [jnp.ones_like(x_batch)]
        self.douts = mcmc_step_batch(*self.ins)

        self.fn = mcmc_step_batch
        self.name = "vectorized_mcmc"
        self.count = 500


if __name__ == "__main__":
    from test_utils import fix_paths
    fix_paths()
    absltest.main()