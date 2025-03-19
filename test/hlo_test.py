import jax
import jax.numpy as jnp
from enzyme_ad.jax import hlo_call, enzyme_jax_ir, optimize_module

@jax.jit
def do_something(mat, scalar):
    a, b = hlo_call(
        mat,
        scalar,
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


res = do_something(jnp.array(jnp.ones((2, 2))), jnp.array(2.7))
print(res)
