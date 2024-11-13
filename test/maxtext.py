# Steps for getting results here
# Run:
#   1) pip install https://github.com/wsmoses/maxtext
#   2) bazel build -c opt //:wheel
#   3) pip install ./bazel-bin/*whl
#   4) python test/maxtext.py

from absl.testing import absltest
import jax.numpy as jnp
from datetime import datetime
import os
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



class MaxText(EnzymeJaxTest):
    def setUp(self):
        import MaxText
        import MaxText.pyconfig

        MaxText.pyconfig.initialize(
            [
                None,
                "test/maxtext_configs/base.yml",
                "dataset_type=synthetic",
                "steps=10",
            ]
        )

    def test(self):
        from enzyme_ad.jax import enzyme_jax_ir
        import MaxText
        import MaxText.pyconfig
        import MaxText.train

        config = MaxText.pyconfig.config

        for name, pipeline, _ in pipelines():
            print("name=", name)

            def rewrite(fn, **kwargs):
                if pipeline is None:
                    return fn
                else:
                    kw = kwargs.copy()
                    return enzyme_jax_ir(
                        pipeline_options=pipeline,
                        argv=argv,
                        inner_jit=False,
                        jit_options=kw,
                    )(fn)

            res1 = MaxText.train.train_loop(config, prejit=rewrite)
            print("name=", name, res1)
            step_time_seconds = res1['scalar']['perf/step_time_seconds']

            self.dump_to_csv(self.csv_filename, name.strip(), "Training", [step_time_seconds])

if __name__ == "__main__":
    from test_utils import fix_paths
    fix_paths()
    absltest.main()
