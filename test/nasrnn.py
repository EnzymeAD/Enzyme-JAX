from absl.testing import absltest
from test_utils import *
import jax.numpy as jnp
import jax.nn as nn


def combine(x, h, w1, w2):
    t1 = jnp.matmul(x, w1)
    t2 = jnp.matmul(h, w2)
    return t1 + t2


def nas_node(input_state, x, weights, start_idx=0):
    # Extract the 16 weight matrices needed for the NAS cells
    tmp = []
    for i in range(8):
        w1 = weights[f"combine_w1_{i+start_idx}"]
        w2 = weights[f"combine_w2_{i+start_idx}"]
        tmp.append(combine(x, input_state, w1, w2))

    # First level of combinations
    midt = []
    # Add operations
    midt.append(nn.relu(tmp[0]) + nn.sigmoid(tmp[3]))
    midt.append(nn.sigmoid(tmp[1]) + jnp.tanh(tmp[2]))
    # Multiply operations
    midt.append(nn.sigmoid(tmp[4]) * jnp.tanh(tmp[5]))
    midt.append(nn.sigmoid(tmp[6]) * nn.relu(tmp[7]))

    # Second level of combinations
    midt.append(nn.sigmoid(midt[1]) + jnp.tanh(midt[2]))
    midt.append(jnp.tanh(midt[0]) * jnp.tanh(midt[3]))

    # Final combination
    midt.append(jnp.tanh(midt[4]) * jnp.tanh(midt[5]))

    return jnp.tanh(midt[6])


def forward(x_sequence, config, weights):
    seq_len = config["seq_len"]
    hidden_size = config["hidden_size"]

    # Initialize state
    state = weights["initial_state"]

    # Process sequence
    for i in range(seq_len):
        x = x_sequence[i]
        state = nas_node(state, x, weights)

    return state


class NasRNN(EnzymeJaxTest):
    def setUp(self):
        import jax.random

        config = {
            "hidden_size": 512,
            "seq_len": 5,
        }

        hidden_size = config["hidden_size"]
        seq_len = config["seq_len"]

        key = jax.random.PRNGKey(0)
        weights = {}
        dweights = {}

        for i in range(8):
            key, subkey = jax.random.split(key)
            key, subkey2 = jax.random.split(key)
            weights[f"combine_w1_{i}"] = jax.random.uniform(subkey, shape=(hidden_size, hidden_size))
            weights[f"combine_w2_{i}"] = jax.random.uniform(subkey2, shape=(hidden_size, hidden_size))

            dweights[f"combine_w1_{i}"] = jax.random.uniform(subkey, shape=(hidden_size, hidden_size))
            dweights[f"combine_w2_{i}"] = jax.random.uniform(subkey2, shape=(hidden_size, hidden_size))

        # Initial state
        key, subkey = jax.random.split(key)
        weights["initial_state"] = jax.random.uniform(subkey, shape=(hidden_size,))
        key, subkey = jax.random.split(key)
        dweights["initial_state"] = jax.random.uniform(subkey, shape=(hidden_size,))

        # Create input sequence
        key, subkey = jax.random.split(key)
        x_sequence = jax.random.uniform(subkey, shape=(seq_len, hidden_size))
        key, subkey = jax.random.split(key)
        dx_sequence = jax.random.uniform(subkey, shape=(seq_len, hidden_size))

        def partial(func, config):
            def sfn(x_sequence, weights):
                return func(x_sequence, config, weights)
            return sfn

        self.fn = partial(forward, config)
        self.name = "nasrnn"
        self.count = 1000
        self.revprimal = False
        self.AllPipelines = pipelines()
        self.AllBackends = CurBackends

        self.ins = [x_sequence, weights]
        self.dins = [dx_sequence, dweights]
        self.douts = dx_sequence
        self.tol = 5e-5


if __name__ == "__main__":
    from test_utils import fix_paths

    fix_paths()
    absltest.main()
