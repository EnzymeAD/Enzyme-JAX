// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=4 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=4 concat-dimension=last" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @do_reduce(%a: tensor<5xf64>) -> tensor<f64> {
  %init = stablehlo.constant dense<0.0> : tensor<f64>
  %r = "stablehlo.reduce"(%a, %init) ({
    ^bb0(%acc: tensor<f64>, %v: tensor<f64>):
      %s = stablehlo.add %acc, %v : tensor<f64>
      stablehlo.return %s : tensor<f64>
  }) {dimensions = array<i64: 0>} : (tensor<5xf64>, tensor<f64>) -> tensor<f64>
  return %r : tensor<f64>
}

func.func @main() attributes {enzyme.no_multifloat} {
  %a1 = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf64>
  %e1 = stablehlo.constant dense<15.0> : tensor<f64>
  %r1 = func.call @do_reduce(%a1) : (tensor<5xf64>) -> tensor<f64>
  "check.expect_close"(%r1, %e1) {max_ulp_difference = 8 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  %a2 = stablehlo.constant dense<[1.0e10, 1.0, 0.1, 0.01, 1.0e-10]> : tensor<5xf64>
  %e2 = stablehlo.constant dense<10000000001.110001> : tensor<f64>
  %r2 = func.call @do_reduce(%a2) : (tensor<5xf64>) -> tensor<f64>
  "check.expect_close"(%r2, %e2) {max_ulp_difference = 8 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  return
}
