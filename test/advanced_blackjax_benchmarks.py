from absl.testing import absltest
from test_utils import EnzymeJaxTest
import jax.numpy as jnp
import jax


class BlackJAXLogisticRegression(EnzymeJaxTest):
    """
    BlackJAX NUTS on Bayesian Logistic Regression.

    This is a realistic use case for BlackJAX - Bayesian inference
    for a logistic regression model with proper priors.
    """
    def setUp(self):
        try:
            import blackjax
        except ImportError:
            self.skipTest("BlackJAX not available")

        # Generate synthetic logistic regression data
        n_data = 100
        n_features = 5
        key = jax.random.PRNGKey(42)

        # Generate features and true coefficients
        X = jax.random.normal(key, (n_data, n_features))
        true_beta = jnp.array([1.5, -0.8, 0.5, 0.0, -1.2])
        true_intercept = 0.3

        # Generate binary outcomes
        logits = X @ true_beta + true_intercept
        probs = jax.nn.sigmoid(logits)
        y = jax.random.bernoulli(key, probs).astype(jnp.float32)

        def logistic_log_posterior(params_flat):
            """Log posterior for Bayesian logistic regression."""
            # Unflatten parameters
            intercept = params_flat[0]
            beta = params_flat[1:]

            # Compute log likelihood (logistic regression)
            logits = X @ beta + intercept
            log_likelihood = jnp.sum(y * logits - jnp.log(1 + jnp.exp(logits)))

            # Log prior (normal priors)
            log_prior_intercept = -0.5 * intercept**2  # N(0, 1)
            log_prior_beta = -0.5 * jnp.sum(beta**2)   # N(0, 1) for each coefficient

            return log_likelihood + log_prior_intercept + log_prior_beta

        # Set up BlackJAX NUTS sampler
        nuts = blackjax.nuts(logistic_log_posterior, step_size=0.01, inverse_mass_matrix=jnp.eye(n_features + 1))

        def blackjax_nuts_step(params_flat):
            """Single BlackJAX NUTS step for logistic regression."""
            key = jax.random.PRNGKey(123)  # Fixed key for determinism
            state = nuts.init(params_flat)
            new_state, info = nuts.step(key, state)
            return new_state.position

        # Initial parameters (flattened: [intercept, beta1, beta2, ...])
        initial_params = jnp.zeros(n_features + 1)

        self.ins = [initial_params]
        self.dins = [jnp.ones_like(initial_params)]
        self.douts = blackjax_nuts_step(initial_params)

        self.fn = blackjax_nuts_step
        self.name = "blackjax_logistic_regression"
        self.count = 50

        # Focus on performance, not correctness (stochastic)
        self.primfilter = lambda x: x
        self.fwdfilter = lambda x: []
        self.revfilter = lambda x: []


class BlackJAXHierarchicalModel(EnzymeJaxTest):
    """
    BlackJAX NUTS on a Hierarchical/Multilevel Model.

    This tests BlackJAX on a more complex model structure that's
    common in Bayesian statistics - different groups with group-level
    and population-level parameters.
    """
    def setUp(self):
        try:
            import blackjax
        except ImportError:
            self.skipTest("BlackJAX not available")

        # Hierarchical model setup
        n_groups = 5
        n_obs_per_group = 10

        # Generate synthetic hierarchical data
        key = jax.random.PRNGKey(42)

        # True population parameters
        mu_pop = 2.0
        sigma_pop = 0.5
        sigma_obs = 0.3

        # True group means (drawn from population distribution)
        true_group_means = mu_pop + sigma_pop * jax.random.normal(key, (n_groups,))

        # Generate observations
        observations = []
        group_indices = []
        for g in range(n_groups):
            group_obs = true_group_means[g] + sigma_obs * jax.random.normal(key, (n_obs_per_group,))
            observations.extend(group_obs)
            group_indices.extend([g] * n_obs_per_group)

        y = jnp.array(observations)
        groups = jnp.array(group_indices)

        def hierarchical_log_posterior(params_flat):
            """Log posterior for hierarchical model."""
            # Parameters: [mu_pop, log_sigma_pop, log_sigma_obs, group_mean_1, ..., group_mean_n]
            mu_pop = params_flat[0]
            log_sigma_pop = params_flat[1]
            log_sigma_obs = params_flat[2]
            group_means = params_flat[3:3+n_groups]

            sigma_pop = jnp.exp(log_sigma_pop)
            sigma_obs = jnp.exp(log_sigma_obs)

            # Log likelihood (observations given group means)
            group_means_expanded = group_means[groups]
            log_likelihood = jnp.sum(-0.5 * ((y - group_means_expanded) / sigma_obs)**2 - jnp.log(sigma_obs))

            # Log prior for group means (hierarchical)
            log_prior_groups = jnp.sum(-0.5 * ((group_means - mu_pop) / sigma_pop)**2 - jnp.log(sigma_pop))

            # Log priors for population parameters
            log_prior_mu = -0.5 * mu_pop**2  # N(0, 1)
            log_prior_sigma_pop = -log_sigma_pop  # Log-normal (exponential on log scale)
            log_prior_sigma_obs = -log_sigma_obs  # Log-normal

            return (log_likelihood + log_prior_groups +
                   log_prior_mu + log_prior_sigma_pop + log_prior_sigma_obs)

        # Set up BlackJAX NUTS
        nuts = blackjax.nuts(hierarchical_log_posterior, step_size=0.01,
                           inverse_mass_matrix=jnp.eye(3 + n_groups))

        def blackjax_hierarchical_step(params_flat):
            """Single NUTS step for hierarchical model."""
            key = jax.random.PRNGKey(456)
            state = nuts.init(params_flat)
            new_state, info = nuts.step(key, state)
            return new_state.position

        # Initial parameters: [mu_pop, log_sigma_pop, log_sigma_obs, group_means...]
        initial_params = jnp.concatenate([
            jnp.array([0.0, 0.0, 0.0]),  # mu_pop, log_sigma_pop, log_sigma_obs
            jnp.zeros(n_groups)  # group means
        ])

        self.ins = [initial_params]
        self.dins = [jnp.ones_like(initial_params)]
        self.douts = blackjax_hierarchical_step(initial_params)

        self.fn = blackjax_hierarchical_step
        self.name = "blackjax_hierarchical"
        self.count = 30  # Complex model, fewer iterations

        # Performance-focused
        self.primfilter = lambda x: x
        self.fwdfilter = lambda x: []
        self.revfilter = lambda x: []


class BlackJAXNonlinearRegression(EnzymeJaxTest):
    """
    BlackJAX HMC on Nonlinear Regression with Complex Function.

    Tests BlackJAX on a nonlinear model that requires gradients through
    complex mathematical functions - more representative of real modeling.
    """
    def setUp(self):
        try:
            import blackjax
        except ImportError:
            self.skipTest("BlackJAX not available")

        # Generate synthetic nonlinear regression data
        n_data = 80
        key = jax.random.PRNGKey(42)

        # True parameters for nonlinear function: y = a * exp(-b * x) * sin(c * x + d) + noise
        true_params = jnp.array([2.5, 0.3, 1.2, 0.8])  # [a, b, c, d]

        x = jnp.linspace(0, 10, n_data)
        y_true = (true_params[0] * jnp.exp(-true_params[1] * x) *
                 jnp.sin(true_params[2] * x + true_params[3]))
        y = y_true + 0.1 * jax.random.normal(key, (n_data,))

        def nonlinear_log_posterior(params):
            """Log posterior for nonlinear regression."""
            a, b, c, d = params

            # Nonlinear model prediction
            y_pred = a * jnp.exp(-b * x) * jnp.sin(c * x + d)

            # Log likelihood (Gaussian noise)
            sigma = 0.2
            log_likelihood = jnp.sum(-0.5 * ((y - y_pred) / sigma)**2)

            # Log priors (weakly informative)
            log_prior = (-0.5 * a**2 - 0.5 * b**2 - 0.5 * c**2 - 0.5 * d**2)

            return log_likelihood + log_prior

        # Set up BlackJAX HMC (instead of NUTS for variety)
        hmc = blackjax.hmc(nonlinear_log_posterior, step_size=0.01, num_integration_steps=10,
                          inverse_mass_matrix=jnp.eye(4))

        def blackjax_nonlinear_step(params):
            """Single HMC step for nonlinear regression."""
            key = jax.random.PRNGKey(789)
            state = hmc.init(params)
            new_state, info = hmc.step(key, state)
            return new_state.position

        # Initial parameters [a, b, c, d]
        initial_params = jnp.array([1.0, 0.5, 1.0, 0.5])

        self.ins = [initial_params]
        self.dins = [jnp.ones_like(initial_params)]
        self.douts = blackjax_nonlinear_step(initial_params)

        self.fn = blackjax_nonlinear_step
        self.name = "blackjax_nonlinear_regression"
        self.count = 40

        # Performance-focused
        self.primfilter = lambda x: x
        self.fwdfilter = lambda x: []
        self.revfilter = lambda x: []


if __name__ == "__main__":
    from test_utils import fix_paths
    fix_paths()
    absltest.main()