from absl.testing import absltest
import jax
import jax.numpy as jnp
from enzyme_ad.jax import hlo_call, enzyme_jax_ir
from test_utils import *
        
def do_something(mat, scalar):
    a, b = hlo_call(
        ones,
        source="""
module {
func.func @myfun(%arg0: tensor<2x2xf64>, %arg1: tensor<f64>) -> (tensor<f64>, tensor<2x2xf64>) {
%cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
%0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf64>) -> tensor<2x2xf64>
%1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x2xf64>, tensor<f64>) -> tensor<f64>
%2 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f64>) -> tensor<2x2xf64>
%3 = stablehlo.multiply %0, %2 : tensor<2x2xf64>
%4 = stablehlo.transpose %3, dims = [1, 0] : (tensor<2x2xf64>) -> tensor<2x2xf64>
return %1, %4 : tensor<f64>, tensor<2x2xf64>
}
}
""",
        fn="myfn",
    )
    return a, b

class HLOFFI(EnzymeJaxTest):
    def setUp(self):
        import jax.numpy as jnp

        self.ins = [
            jnp.array(jnp.ones(2,2)),
            jnp.array(2.7),
        ]
        self.dins = [
            jnp.array(5 * jnp.ones(2,2)),
            jnp.array(3.1),
        ]
        self.douts = [jnp.array(3.4), jnp.array(7 * jnp.ones(2.2))]

        self.primfilter = no_newxla
        self.fwdfilter = no_newxla
        self.revfilter = no_newxla

        self.fn = do_something

        self.name = "hlo_ffi"

if __name__ == "__main__":
    from test_utils import fix_paths

    fix_paths()

    absltest.main()