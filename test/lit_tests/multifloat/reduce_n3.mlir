// Default (non-precise) reduce only preserves leading-limb precision across
// elements — small terms get lost when summed with a much larger one. So the
// default-RUN cases use balanced magnitudes. The precise-reduce=true cases
// exercise the full N-limb-accurate reducer body (my refactored path).
//
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=3 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=3 concat-dimension=last" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=3 concat-dimension=first precise-reduce=true" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=3 concat-dimension=last precise-reduce=true" | stablehlo-translate - --interpret --allow-unregistered-dialect

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
  // Balanced-magnitude case — works under both precise and non-precise reduce.
  %a = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf64>
  %expected = stablehlo.constant dense<15.0> : tensor<f64>
  %r = func.call @do_reduce(%a) : (tensor<5xf64>) -> tensor<f64>
  "check.expect_close"(%r, %expected) {max_ulp_difference = 8 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  return
}
