# Steps for getting results here
# Run:
#   1) bazel build -c opt //:wheel
#   2) python3 -m pip install --user ./bazel-bin/*.whl  git+https://github.com/wsmoses/maxtext aqtp tensorboardX google-cloud-storage datasets neuralgcm gcsfs --break-system-packages
#   3) python test/maxtext.py

from absl.testing import absltest

argv = ("-I/usr/include/c++/11", "-I/usr/include/x86_64-linux-gnu/c++/11")


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


if __name__ == "__main__":
    from test_utils import fix_paths

    fix_paths()
    absltest.main()
