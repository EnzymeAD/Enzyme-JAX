from absl import app


def main(argv):
    from test_utils import pipelines

    import jax

    print(jax.devices())

    import os

    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    os.environ["KERAS_BACKEND"] = "jax"

    cwd = os.getcwd()
    print(cwd)
    os.environ["KAGGLEHUB_CACHE"] = os.path.join(cwd, "kagglecache")
    os.environ["KAGGLEHUB_VERBOSITY"] = "error"

    import keras

    print(keras.config.backend())
    keras.config.set_floatx("float32")
    keras.config.set_epsilon(1e-07)
    keras.config.set_backend("jax")
    keras.config.set_image_data_format("channels_last")
    print(keras.config.backend())
    assert keras.config.backend() == "jax"

    import benchmark.stable_diffusion
    import benchmark.mistral
    import benchmark.gemma
    import benchmark.bert
    import benchmark.sam

    Both = [False, True]
    benchfns = []

    # breaks on gpu =/
    if False:
        benchfns += [
            (
                "stable_diffusion_predict",
                benchmark.stable_diffusion.stable_diffusion_predict_run,
                Both,
            ),
            (
                "stable_diffusion_fit",
                benchmark.stable_diffusion.stable_diffusion_fit_run,
                Both,
            ),
        ]

    # Uses too much ram on the 4070 in CI
    if False:
        benchfns += [
            # Seems fine on gpu
            ("mistral_predict", benchmark.mistral.mistral_predict_run, Both),
            ("mistral_fit", benchmark.mistral.mistral_fit_run, Both),
        ]

    # requires model download, skipping
    if False:
        benchfns += [
            ("gemma_predict", benchmark.gemma.gemma_predict_run, Both),
            ("gemma_fit", benchmark.gemma.gemma_fit_run, Both),
        ]

    if True:
        benchfns += [
            ("bert_predict", benchmark.bert.bert_predict_run, Both),
        ]

    # OOM's the 4070 :'(
    if False:
        benchfns += [
            ("bert_fit", benchmark.bert.bert_fit_run, Both),
        ]
    # also oom
    if False:
        benchfns += [
            ("sam_predict", benchmark.sam.sam_predict_run, Both),
            ("sam_fit", benchmark.sam.sam_fit_run, Both),
        ]

    num_tests = 5
    num_tests = 1

    for bname, bench, ADs in benchfns:
        for AD in ADs:
            for name, pipe, dev in pipelines():
                if pipe is None and AD:
                    continue
                if bname.endswith("predict") and AD:
                    continue
                # Oom's the gpu ci
                if bname == "bert_predict" and name == "IPartOpt":
                    continue

                os.environ.pop("ENZYME_JAX", None)
                os.environ.pop("ENZYME_JAX_PRE", None)
                if pipe is not None:
                    os.environ["ENZYME_JAX"] = pipe.pass_pipeline()
                if AD:
                    os.environ["ENZYME_JAX_PRE"] = "1"
                for i in range(num_tests):
                    print("Running ", name, " ", bname, " AD=", AD, " dev=", dev)
                    benchmark.benchmark(bench)
                    print("Done Running ", name, " ", bname, " AD=", AD, " dev=", dev)


if __name__ == "__main__":
    import platform

    # Deps not available on macos
    if platform.system() != "Darwin":
        from test_utils import fix_paths

        fix_paths()
        app.run(main)
