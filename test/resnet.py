"""
Implementation of ResNet using Hugging Face for Enzyme-JAX benchmarking.

ResNet is a CNN architecture that differs from transformers through its use of residual connections.
While it does use convolutions, its computational patterns provide a good contrast to transformers.
"""

from transformers import FlaxResNetModel, ResNetConfig
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from test_utils import *

# Create a ResNet configuration for benchmarking
# Using standard ResNet-50 architecture
config = ResNetConfig(
    num_channels=3,
    embedding_size=64,
    hidden_sizes=[256, 512, 1024, 2048],  # Standard ResNet-50 sizes
    depths=[3, 4, 6, 3],                  # Standard ResNet-50 block counts
    layer_type="bottleneck",               # Standard ResNet-50 uses bottleneck blocks
    hidden_act="relu",
    downsample_in_first_stage=False
)

# Initialize the model with the config
model = FlaxResNetModel(config)

# Define the ResNet forward pass
def resnet_forward(pixel_values):
    outputs = model(pixel_values=pixel_values)
    return outputs.pooler_output

# Define the test class for ResNet
class ResNetTest(EnzymeJaxTest):
    def setUp(self):
        # Create a random input tensor (batch_size, channels, height, width)
        # Using a standard input size
        pixel_values = np.random.random((1, 3, 224, 224)).astype(np.float32)
        
        # Run the ResNet forward pass with the input
        self.pooler_output = resnet_forward(pixel_values)
        
        # Setup for the test harness
        self.fn = resnet_forward
        self.name = "resnet"
        self.count = 50
        self.revprimal = False
        self.AllPipelines = pipelines()
        self.AllBackends = CurBackends
        
        # Input and output setup
        self.ins = [pixel_values]
        self.dins = [pixel_values]  # Using same for gradient computation
        self.douts = [self.pooler_output]
        self.tol = 5e-5

# Running the tests with absltest
if __name__ == "__main__":
    from test_utils import fix_paths
    
    fix_paths()
    absltest.main()
