from transformers import AutoImageProcessor, FlaxViTForImageClassification, ViTConfig
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

# Load the image processor and model
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = FlaxViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

# Define the ViT forward pass
def vit_forward(pixel_values):
    """ Vision Transformer forward pass """
    outputs = model(pixel_values=pixel_values)
    return outputs.logits

# Define the test class for ViT using absltest
class VisionTransformerTest(EnzymeJaxTest):
    def setUp(self):
        # Create a ViTConfig object instead of using a dictionary
        self.config = ViTConfig.from_pretrained("google/vit-base-patch16-224")

        # Sample image from COCO dataset
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        pil_image = Image.open(requests.get(url, stream=True).raw)
        np_image = np.array(pil_image)

        # Process the image
        inputs = processor(images=np_image, return_tensors="np")
        pixel_values = inputs["pixel_values"]  # Extract the actual NumPy array

        # Run the ViT forward pass with the processed input
        self.logits = vit_forward(pixel_values)

        # Setup for the test harness
        self.fn = vit_forward
        self.name = "vit"
        self.count = 10
        self.revprimal = False
        self.AllPipelines = pipelines()
        self.AllBackends = CurBackends

        # Input and output setup
        self.ins = [pixel_values]
        self.dins = [pixel_values]
        self.douts = [self.logits]
        self.tol = 5e-5

# Running the tests with absltest
if __name__ == "__main__":
    absltest.main()
