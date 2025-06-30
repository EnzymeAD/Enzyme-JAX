from bench_utils import *

class JaxMD(EnzymeJaxBenchmark):
    def setup(self):
        print("Setting up JaxMD benchmark...")

if __name__ == "__main__":
    import jax
    jax.config.update("jax_enable_x64", True)

    benchmarker = JaxMD()
    benchmarker.run()
