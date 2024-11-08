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
    Both = [False, True]
    benchfns = [
        ("stable_diffusion_predict", benchmark.stable_diffusion.stable_diffusion_predict_run, Both),
        ("stable_diffusion_fit", benchmark.stable_diffusion.stable_diffusion_fit_run, Both),
        ("mistral_predict", benchmark.mistral.mistral_predict_run, Both),
        ("mistral_fit", benchmark.mistral.mistral_fit_run, Both),
        ("gemma_predict", benchmark.gemma.gemma_predict_run, Both),
        ("gemma_fit", benchmark.gemma.gemma_fit_run, Both),
        ("bert_predict", benchmark.bert.bert_predict_run, Both),
        ("bert_fit", benchmark.bert.bert_fit_run, Both),
    ]

    for (bname, bench, ADs) in benchfns:
        for AD in ADs:
            for (name, pipe, _) in pipelines:
                if pipe is None and AD:
                    continue
                print("Running ", name, " ", bname, " AD=", AD)
                os.environ.pop("ENZYME_JAX", None)
                os.environ.pop("ENZYME_JAX_PRE", None)
                if pipe is not None:
                    os.environ["ENZYME_JAX"] = pipe.pass_pipeline()
                if AD:
                    os.environ["ENZYME_JAX_PRE"] = 1
                benchmark.benchmark(bench)
                print("Done Running ", name, " ", bname, " AD=", AD)


if __name__ == "__main__":
    app.run(main)
