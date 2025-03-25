from transformers import FlaxBertModel, BertTokenizer
from PIL import Image
import jax
import requests
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
import jax.random
from enzyme_ad.jax import JaXPipeline, hlo_opts
from test_utils import *
import llama


# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = FlaxBertModel.from_pretrained("bert-base-uncased")

# Define the BERT forward pass
def bert_forward(input_ids, attention_mask):
    """ BERT forward pass """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.last_hidden_state

# Define the test class for BERT using absltest
class BertTransformerTest(EnzymeJaxTest):
    def setUp(self):
        # Sample text input
        text = "This is a sample text for BERT transformer test."
        inputs = tokenizer(text, return_tensors="np")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Run the BERT forward pass with the processed input
        self.hidden_state = bert_forward(input_ids, attention_mask)

        # Setup for the test harness
        self.fn = bert_forward
        self.name = "bert"
        self.count = 50
        self.revprimal = False
        self.AllPipelines = pipelines()
        self.AllBackends = CurBackends

        # Input and output setup
        self.ins = [input_ids, attention_mask]
        self.dins = [input_ids, attention_mask]
        self.douts = [self.hidden_state]
        self.tol = 5e-3

# Running the tests with absltest
if __name__ == "__main__":
    absltest.main()
