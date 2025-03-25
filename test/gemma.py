"""
Implementation of Gemma using Hugging Face for Enzyme-JAX benchmarking.

Gemma is Google's family of lightweight, state-of-the-art open language models.
It is built using the same research and technology used to create Gemini models.
"""

from transformers import FlaxGemmaForCausalLM, GemmaConfig
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from test_utils import *

# Create a smaller Gemma configuration for benchmarking
# Using reduced parameters to make it more suitable for benchmarking
gemma_config = GemmaConfig(
    vocab_size=256000,
    hidden_size=512,           # Reduced from original
    num_hidden_layers=4,       # Reduced from original
    num_attention_heads=8,     # Reduced from original
    num_key_value_heads=8,     # Must match num_attention_heads for non-MQA
    intermediate_size=2048,    # Reduced from original
    max_position_embeddings=2048,
    hidden_act="gelu",
    attention_bias=True,
    rms_norm_eps=1e-6,
    use_cache=True,
)

# Initialize model with the custom config
model = FlaxGemmaForCausalLM(gemma_config)

# Define the Gemma forward pass
def gemma_forward(input_ids, attention_mask=None):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits

class GemmaTest(EnzymeJaxTest):
    def setUp(self):
        # Create a synthetic input (token ids)
        sequence_length = 64
        batch_size = 1
        
        # Random token ids between 0 and vocab_size-1
        input_ids = np.random.randint(
            0, gemma_config.vocab_size, size=(batch_size, sequence_length), dtype=np.int32
        )
        
        # All 1s for attention mask (no padding)
        attention_mask = np.ones((batch_size, sequence_length), dtype=np.int32)
        
        # Run the Gemma forward pass with the processed input
        self.logits = gemma_forward(input_ids, attention_mask)
        
        # Setup for the test harness
        self.fn = gemma_forward
        self.name = "gemma"
        self.count = 50 
        self.revprimal = False
        self.AllPipelines = pipelines()
        self.AllBackends = CurBackends
        
        # Input and output setup
        self.ins = [input_ids, attention_mask]
        self.dins = [input_ids, attention_mask]
        self.douts = [self.logits]
        self.tol = 5e-5

# Running the tests with absltest
if __name__ == "__main__":
    from test_utils import fix_paths
    
    fix_paths()
    absltest.main()
