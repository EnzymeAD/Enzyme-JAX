from absl.testing import absltest
from test_utils import EnzymeJaxTest, no_newxla
from enzyme_ad.jax import hlo_call


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


class HLOFFI(EnzymeJaxTest):
    def setUp(self):
        import jax.numpy as jnp

        self.ins = [
            jnp.array(jnp.ones((2, 2))),
            jnp.array(2.7),
        ]
        self.dins = [
            jnp.full((2, 2), 5.0),
            jnp.array(3.1),
        ]
        self.douts = [jnp.array(3.4), jnp.full((2, 2), 7.0)]

        self.primfilter = no_newxla
        self.fwdfilter = no_newxla
        self.revfilter = no_newxla

        self.fn = do_something

        self.name = "hlo_ffi"

        self.atol = 1e-4
        self.rtol = 0.0


if __name__ == "__main__":
    from test_utils import fix_paths

    fix_paths()

    absltest.main()
