// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=3 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=3 concat-dimension=last" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @do_cos(%a: tensor<6xf64>) -> tensor<6xf64> {
  %r = stablehlo.cosine %a : tensor<6xf64>
  return %r : tensor<6xf64>
}

func.func @main() attributes {enzyme.no_multifloat} {
  // cos(0), cos(0.5), cos(1), cos(π/4), cos(π/3), cos(-π/3)
  %a = stablehlo.constant dense<[0.0, 0.5, 1.0, 0.7853981633974483, 1.0471975511965976, -1.0471975511965976]> : tensor<6xf64>
  %expected = stablehlo.constant dense<[1.0, 0.8775825618903728, 0.5403023058681398, 0.7071067811865476, 0.5000000000000001, 0.5000000000000001]> : tensor<6xf64>

  %r = func.call @do_cos(%a) : (tensor<6xf64>) -> tensor<6xf64>
  "check.expect_close"(%r, %expected) {max_ulp_difference = 64 : ui64} : (tensor<6xf64>, tensor<6xf64>) -> ()

  return
}
