// RUN: enzymexlamlir-opt %s --enzyme-batch --arith-raise | %stablehlo-translate - --interpret

module {
  func.func private @relu_broadcast_scalar(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %pred = stablehlo.compare GE, %arg0, %cst : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %result = "stablehlo.if"(%pred) ({
        stablehlo.return %arg0 : tensor<f64>
    }, {
        %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        stablehlo.return %cst_0 : tensor<f64>
    }) :  (tensor<i1>) -> tensor<f64>
    return %result : tensor<f64>
  }
  func.func @main() {
    %arg0 = stablehlo.constant dense<[[42.0, -42.0], [0.0, 1.0]]> : tensor<2x2xf64>
    %0 = enzyme.batch @relu_broadcast_scalar(%arg0) {batch_shape = array<i64: 2, 2>} : (tensor<2x2xf64>) -> tensor<2x2xf64>
    check.expect_eq_const %0, dense<[[42.0, 0.0], [0.0, 1.0]]> : tensor<2x2xf64>
    return
  }
}
