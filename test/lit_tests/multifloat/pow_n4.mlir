// See pow_n3.mlir for explanation.
//
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=4 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=4 concat-dimension=last" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @do_pow(%a: tensor<5xf64>, %b: tensor<5xf64>) -> tensor<5xf64> {
  %r = stablehlo.power %a, %b : tensor<5xf64>
  return %r : tensor<5xf64>
}

func.func @main() attributes {enzyme.no_multifloat} {
  %a = stablehlo.constant dense<[2.0, 10.0, 1.0, 0.5, 5.0]> : tensor<5xf64>
  %b = stablehlo.constant dense<[3.0, 0.5, 100.0, -2.0, 0.0]> : tensor<5xf64>
  %expected = stablehlo.constant dense<[8.0, 3.1622776601683795, 1.0, 4.0, 1.0]> : tensor<5xf64>

  %r = func.call @do_pow(%a, %b) : (tensor<5xf64>, tensor<5xf64>) -> tensor<5xf64>
  "check.expect_close"(%r, %expected) {max_ulp_difference = 32 : ui64} : (tensor<5xf64>, tensor<5xf64>) -> ()

  return
}
