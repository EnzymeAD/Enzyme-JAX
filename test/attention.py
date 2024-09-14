from test_utils import recursive_check
from absl.testing import absltest
import jax.numpy as jnp
import jax.random
import enzyme_ad.jax as enzyme_jax
from enzyme_ad.jax import JaXPipeline, hlo_opts
from test_utils import *

# Use the same pipelines as in llama.py
pipelines = [
    # ("JaX  ", None, CurBackends),
    ("JaXPipe", JaXPipeline(), CurBackends),
    (
        "HLOOpt",
        JaXPipeline(
            "inline{default-pipeline=canonicalize max-iterations=4},"
            + "canonicalize,cse,enzyme-hlo-opt,cse"
        ),
        CurBackends,
    ),
    ("DefOpt", JaXPipeline(hlo_opts()), CurBackends),
    (
        "EqualitySaturation",
        JaXPipeline(
            "inline{default-pipeline=canonicalize max-iterations=4},"
            + "canonicalize,cse,enzyme-hlo-opt,cse,equality-saturation-pass"
        ),
        CurBackends,
    ),
]

def attention(query, key, value):
    scores = jnp.matmul(query, key.transpose(0, 2, 1))
    exp_scores = jnp.exp(scores)
    sum_exp_scores = jnp.sum(exp_scores, axis=-1, keepdims=True)
    softmax_scores = exp_scores / sum_exp_scores
    return jnp.matmul(softmax_scores, value)

class AttentionTest(EnzymeJaxTest):
    def setUp(self):
        # Initialize with random values as in the llama test
        batch_size = 2
        seq_len = 5
        dim = 3
        
        self.fn = attention
        self.name = 'AttentionTest'
        self.key = jax.random.PRNGKey(0)
        self.query = jax.random.uniform(self.key, shape=(batch_size, seq_len, dim))
        self.key, subkey = jax.random.split(self.key)
        self.key_matrix = jax.random.uniform(subkey, shape=(batch_size, seq_len, dim))
        self.key, subkey = jax.random.split(self.key)
        self.value = jax.random.uniform(subkey, shape=(batch_size, seq_len, dim))

        # Set inputs and outputs for testing
        self.ins = [self.query, self.key_matrix, self.value]
        self.dins = [self.query, self.key_matrix, self.value]  # Use the same inputs for testing derivative checks
        self.douts = [self.value]  # Placeholder, you can modify based on your specific requirements
        self.tol = 1e-5  # Set tolerance as required
        self.AllPipelines = pipelines  # Use the same pipelines as in llama test

    # def forward(self, query, key_matrix, value):
    #     return attention(query, key_matrix, value)
    #
    # def test_attention_pipelines(self):
    #     # Loop through the pipelines and test each one
    #     for pipeline_name, pipeline, backends in self.AllPipelines:
    #         print(f"Testing pipeline: {pipeline_name}")
    #         efunc = enzyme_jax.enzyme_jax_ir(pipeline_options=pipeline)(attention)
    #         jfunc = jax.jit(attention)
    #
    #         eres = efunc(self.query, self.key_matrix, self.value)
    #         jres = jfunc(self.query, self.key_matrix, self.value)


if __name__ == "__main__":
    absltest.main()
