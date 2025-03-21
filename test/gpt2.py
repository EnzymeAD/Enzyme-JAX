"""
Implementation of GPT-2 using Hugging Face for Enzyme-JAX benchmarking.
"""

from transformers import FlaxGPT2LMHeadModel, GPT2Tokenizer
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from test_utils import *

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = FlaxGPT2LMHeadModel.from_pretrained("gpt2")

# GPT-2 requires padding on the left, ensure it's handled correctly
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

# Define the GPT-2 forward pass
def gpt2_forward(input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # Return the logits (predictions for each token)
    return outputs.logits

# Define the test class for GPT-2 using absltest
class GPT2TransformerTest(EnzymeJaxTest):
    def setUp(self):
        # Sample text input
        text = "The quick brown fox jumps over the lazy dog. This is a sample text for GPT-2 transformer test."
        
        # Tokenize and prepare inputs
        inputs = tokenizer(text, return_tensors="np", padding="max_length", max_length=32)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Run the GPT-2 forward pass with the processed input
        self.logits = gpt2_forward(input_ids, attention_mask)

        # Setup for the test harness
        self.fn = gpt2_forward
        self.name = "gpt2"
        self.count = 200  # Number of iterations
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
