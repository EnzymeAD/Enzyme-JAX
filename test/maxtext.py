# Steps for getting results here
# Run:
#   1) pip install https://github.com/wsmoses/maxtext
#   2) bazel build -c opt //:wheel
#   3) pip install ./bazel-bin/*whl
#   4) python test/maxtext.py

from absl.testing import absltest
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

argv = ("-I/usr/include/c++/11", "-I/usr/include/x86_64-linux-gnu/c++/11")

import jax.numpy as np
import numpy as onp
from jax import jit
from jax import random
from jax import lax

pipelines = [
    ("JaX  ", None),
    ("JaXPipe", JaXPipeline()),
    (
        "HLOOpt",
        JaXPipeline(
            "inline{default-pipeline=canonicalize max-iterations=4},"
            + "canonicalize,cse,enzyme-hlo-opt,cse"
        ),
    ),
    ("PartOpt", JaXPipeline(partialopt)),
    ("DefOpt", JaXPipeline(hlo_opts())),
]


class MaxText(absltest.TestCase):
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
        import MaxText
        import MaxText.pyconfig
        import MaxText.train

        config = MaxText.pyconfig.config

        for name, pipeline in pipelines:
            print("name=", name)

            def rewrite(fn):
                if pipeline is None:
                    return fn
                else:
                    return enzyme_jax_ir(pipeline_options=pipeline, argv=argv)(fn)

            res1 = MaxText.train.train_loop(config, prejit=rewrite)
            print("name=", name, res1)


if __name__ == "__main__":
    absltest.main()
