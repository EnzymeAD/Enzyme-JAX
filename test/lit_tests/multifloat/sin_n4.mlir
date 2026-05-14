// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=4 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=4 concat-dimension=last" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @do_sin(%a: tensor<6xf64>) -> tensor<6xf64> {
  %r = stablehlo.sine %a : tensor<6xf64>
  return %r : tensor<6xf64>
}

func.func @main() attributes {enzyme.no_multifloat} {
  %a = stablehlo.constant dense<[0.0, 0.5, 1.0, 0.7853981633974483, 1.5707963267948966, -1.0471975511965976]> : tensor<6xf64>
  %expected = stablehlo.constant dense<[0.0, 0.479425538604203, 0.8414709848078965, 0.7071067811865475, 1.0, -0.8660254037844386]> : tensor<6xf64>

  %r = func.call @do_sin(%a) : (tensor<6xf64>) -> tensor<6xf64>
  "check.expect_close"(%r, %expected) {max_ulp_difference = 16 : ui64} : (tensor<6xf64>, tensor<6xf64>) -> ()

  return
}
