from bench_utils import *

class NeuralGCM(EnzymeJaxBenchmark):
    def setup(self):
        print("Setting up NeuralGCM benchmark...")

if __name__ == "__main__":
    import jax
    jax.config.update("jax_enable_x64", True)

    benchmarker = NeuralGCM()
    benchmarker.run()
