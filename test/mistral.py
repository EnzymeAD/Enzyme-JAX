from transformers import FlaxMistralModel, MistralConfig, BertTokenizer
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
import jax.random
from enzyme_ad.jax import JaXPipeline, hlo_opts
from test_utils import *
import llama
from typing import Tuple
import jax.numpy as jnp

def create_custom_mistral_config(
    vocab_size=50257,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=2048,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    initializer_range=0.02,
    **kwargs
):
    return MistralConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        initializer_range=initializer_range,
        **kwargs
    )

def create_custom_mistral_model(config: MistralConfig, input_shape: Tuple = (1, 1), seed: int = 0):
    return FlaxMistralModel(config, input_shape=input_shape, seed=seed)

pipelines = [
    ("JaX", None, CurBackends),
    ("JaXPipe", JaXPipeline(), CurBackends),
    (
        "HLOOpt",
        JaXPipeline(
            "inline{default-pipeline=canonicalize max-iterations=4},"
            + "canonicalize,cse,enzyme-hlo-opt,cse"
        ),
        CurBackends,
    ),
    ("PartOpt", JaXPipeline(llama.partialopt), CurBackends),
    ("DefOpt", JaXPipeline(hlo_opts()), CurBackends),
    (
        "EqSat",
        JaXPipeline(
            "inline{default-pipeline=canonicalize max-iterations=4},"
            + "equality-saturation-pass"
        ),
        CurBackends,
    ),
]

mistral_config = create_custom_mistral_config(
    hidden_size=512,
    num_hidden_layers=8,
    num_attention_heads=8,
)

model = create_custom_mistral_model(mistral_config)

# Load the tokenizer (maybe we should try a different one)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def mistral_forward(input_ids, attention_mask):
    """ Mistral forward pass """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.last_hidden_state

class MistralTransformerTest(EnzymeJaxTest):
    def setUp(self):
        text = "This is a sample text for Mistral transformer test."
        inputs = tokenizer(text, return_tensors="np")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        self.hidden_state = mistral_forward(input_ids, attention_mask)

        self.fn = mistral_forward
        self.name = "mistral"
        self.count = 1000
        self.revprimal = False
        self.AllPipelines = pipelines
        self.AllBackends = CurBackends

        self.ins = [input_ids, attention_mask]
        self.dins = [input_ids, attention_mask]
        self.douts = [self.hidden_state]
        self.tol = 5e-5

if __name__ == "__main__":
    absltest.main()
