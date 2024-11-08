from absl import app
import jax.numpy as jnp
import jax.random
import jax.lax
import enzyme_ad.jax as enzyme_jax
from enzyme_ad.jax import (
    enzyme_jax_ir,
    NewXLAPipeline,
    OldXLAPipeline,
    JaXPipeline,
    hlo_opts,
)
import numpy as np
import timeit
from test_utils import *

argv = ("-I/usr/include/c++/11", "-I/usr/include/x86_64-linux-gnu/c++/11")

import jax.numpy as np
import numpy as onp
from jax import jit
from jax import random
from jax import lax

def main(argv):
    import os
    os.environ["KERAS_BACKEND"] = "jax"
    import keras
    print(keras.config.backend())
    keras.config.set_floatx('float32')
    keras.config.set_epsilon(1e-07)
    keras.config.set_backend('jax')
    keras.config.set_image_data_format('channels_last')
    print(keras.config.backend())
    
    import benchmark.stable_diffusion
    import benchmark.mistral
    import benchmark.gemma
    import benchmark.bert
    benchfns = [
        ("stable_diffusion_predict", benchmark.stable_diffusion.stable_diffusion_predict_run),
        ("stable_diffusion_fit", benchmark.stable_diffusion.stable_diffusion_fit_run),
        ("mistral_predict", benchmark.mistral.mistral_predict_run),
        ("mistral_fit", benchmark.mistral.mistral_fit_run),
        ("gemma_predict", benchmark.gemma.gemma_predict_run),
        ("gemma_fit", benchmark.gemma.gemma_fit_run),
        ("bert_predict", benchmark.bert.bert_predict_run),
        ("bert_fit", benchmark.bert.bert_fit_run),
    ]

    for (bname, bench) in benchfns:
        for (name, pipe, _) in pipelines:
            print("Running ", name, " ", bname)
            os.environ.pop("ENZYME_JAX", None)
            os.environ.pop("ENZYME_JAX_PRE", None)
            if pipe is not None:
                os.environ["ENZYME_JAX"] = pipe.pass_pipeline()
            benchmark.benchmark(bench)
            print("Done Running ", name, " ", bname)


if __name__ == "__main__":
    app.run(main)
