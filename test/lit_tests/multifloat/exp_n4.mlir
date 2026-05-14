// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=4 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=4 concat-dimension=last" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @do_exp(%a: tensor<6xf64>) -> tensor<6xf64> {
  %r = stablehlo.exponential %a : tensor<6xf64>
  return %r : tensor<6xf64>
}

func.func @main() attributes {enzyme.no_multifloat} {
  %a = stablehlo.constant dense<[0.0, 1.0, -1.0, 2.0, 10.0, -5.0]> : tensor<6xf64>
  %expected = stablehlo.constant dense<[1.0, 2.718281828459045, 0.36787944117144233, 7.38905609893065, 22026.465794806718, 0.006737946999085467]> : tensor<6xf64>

  %r = func.call @do_exp(%a) : (tensor<6xf64>) -> tensor<6xf64>
  // N=4 has ~96 bits; tighter ULP bound.
  "check.expect_close"(%r, %expected) {max_ulp_difference = 16 : ui64} : (tensor<6xf64>, tensor<6xf64>) -> ()

  return
}
