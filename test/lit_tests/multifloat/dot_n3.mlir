// dot at N=3 reaches multiFloatMul/Add via the DotGeneralToMulReducePattern
// pre-pass (dotGeneralToReduce=true is default), which decomposes the dot to
// elementwise mul + reduce before multifloat conversion runs.
//
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=3 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=3 concat-dimension=last" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @do_dot(%a: tensor<4xf64>, %b: tensor<4xf64>) -> tensor<f64> {
  %r = stablehlo.dot_general %a, %b, contracting_dims = [0] x [0] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
  return %r : tensor<f64>
}

func.func @main() attributes {enzyme.no_multifloat} {
  // dot([1,2,3,4], [5,6,7,8]) = 5 + 12 + 21 + 32 = 70
  %a = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf64>
  %b = stablehlo.constant dense<[5.0, 6.0, 7.0, 8.0]> : tensor<4xf64>
  %expected = stablehlo.constant dense<70.0> : tensor<f64>

  %r = func.call @do_dot(%a, %b) : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
  "check.expect_close"(%r, %expected) {max_ulp_difference = 8 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  return
}
