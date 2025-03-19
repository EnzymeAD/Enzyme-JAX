from absl.testing import absltest
import jax.numpy as jnp
import jax.random
import jax.lax
import enzyme_ad.jax as enzyme_jax

def test(x, y, z, w):
    a, b = enzyme_jax.hlo_call(
        x,
        y,
        source="""
module {
func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<2x2xf32>) {
%cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
%0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
%1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<f32>
%2 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<2x2xf32>
%3 = stablehlo.multiply %0, %2 : tensor<2x2xf32>
%4 = stablehlo.transpose %3, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
return %1, %4 : tensor<f32>, tensor<2x2xf32>
}
}
""",
    )
    return a, b

class Simple(absltest.TestCase):
    def test_simple_random(self):
        jfunc = jax.jit(test)

        efunc = enzyme_jax.enzyme_jax_ir(pipeline_options=enzyme_jax.JaXPipeline("equality-saturation-pass"),)(test)
        
        ka, kb, kc, kd = jax.random.split(jax.random.PRNGKey(0), num=4)
        a = jnp.array(jnp.ones((2, 2)))
        b = jnp.array(2.7)
        c = jax.random.uniform(kc, shape=(100, 1000,2))
        d = jax.random.uniform(kd, shape=(100, 1000,2))

        eres = efunc(a, b, c, d)
        print("enzyme primal", eres)

if __name__ == "__main__":
    absltest.main()
