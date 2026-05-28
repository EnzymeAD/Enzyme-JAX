// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=3 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=3 concat-dimension=last" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @do_exp(%a: tensor<6xf64>) -> tensor<6xf64> {
  %r = stablehlo.exponential %a : tensor<6xf64>
  return %r : tensor<6xf64>
}

func.func @main() attributes {enzyme.no_multifloat} {
  // Argument range chosen to stay within f32 limb representability
  // (|2^n| < f32_max ≈ 3.4e38 means |x*log2(e)| < ~127, so |x| < ~88).
  %a = stablehlo.constant dense<[0.0, 1.0, -1.0, 2.0, 10.0, -5.0]> : tensor<6xf64>
  %expected = stablehlo.constant dense<[1.0, 2.718281828459045, 0.36787944117144233, 7.38905609893065, 22026.465794806718, 0.006737946999085467]> : tensor<6xf64>

  %r = func.call @do_exp(%a) : (tensor<6xf64>) -> tensor<6xf64>
  // N=3 has ~72 bits of precision; loose ULP bound for the polynomial+cubing chain.
  "check.expect_close"(%r, %expected) {max_ulp_difference = 64 : ui64} : (tensor<6xf64>, tensor<6xf64>) -> ()

  return
}
