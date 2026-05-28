// See reduce_precise_n3.mlir for full explanation. N=4 has ~96 bits of
// mantissa, even more headroom for cancellation/mixed-magnitude cases.
//
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=4 concat-dimension=first precise-reduce=true" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=4 concat-dimension=last precise-reduce=true" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @do_reduce4(%a: tensor<4xf64>) -> tensor<f64> {
  %init = stablehlo.constant dense<0.0> : tensor<f64>
  %r = "stablehlo.reduce"(%a, %init) ({
    ^bb0(%acc: tensor<f64>, %v: tensor<f64>):
      %s = stablehlo.add %acc, %v : tensor<f64>
      stablehlo.return %s : tensor<f64>
  }) {dimensions = array<i64: 0>} : (tensor<4xf64>, tensor<f64>) -> tensor<f64>
  return %r : tensor<f64>
}

func.func @do_reduce3(%a: tensor<3xf64>) -> tensor<f64> {
  %init = stablehlo.constant dense<0.0> : tensor<f64>
  %r = "stablehlo.reduce"(%a, %init) ({
    ^bb0(%acc: tensor<f64>, %v: tensor<f64>):
      %s = stablehlo.add %acc, %v : tensor<f64>
      stablehlo.return %s : tensor<f64>
  }) {dimensions = array<i64: 0>} : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
  return %r : tensor<f64>
}

func.func @main() attributes {enzyme.no_multifloat} {
  %a1 = stablehlo.constant dense<[1.0e10, 1.0, 0.1, 0.01]> : tensor<4xf64>
  %e1 = stablehlo.constant dense<10000000001.110001> : tensor<f64>
  %r1 = func.call @do_reduce4(%a1) : (tensor<4xf64>) -> tensor<f64>
  "check.expect_close"(%r1, %e1) {max_ulp_difference = 4 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  %a2 = stablehlo.constant dense<[1.0e30, 1.0, -1.0e30]> : tensor<3xf64>
  %e2 = stablehlo.constant dense<1.0> : tensor<f64>
  %r2 = func.call @do_reduce3(%a2) : (tensor<3xf64>) -> tensor<f64>
  "check.expect_close"(%r2, %e2) {max_ulp_difference = 4 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  return
}
