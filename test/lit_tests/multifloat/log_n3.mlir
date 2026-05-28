// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=3 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=3 concat-dimension=last" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @do_log(%a: tensor<5xf64>) -> tensor<5xf64> {
  %r = stablehlo.log %a : tensor<5xf64>
  return %r : tensor<5xf64>
}

func.func @main() attributes {enzyme.no_multifloat} {
  %a = stablehlo.constant dense<[1.0, 2.0, 10.0, 0.5, 100.0]> : tensor<5xf64>
  %expected = stablehlo.constant dense<[0.0, 0.6931471805599453, 2.302585092994046, -0.6931471805599453, 4.605170185988092]> : tensor<5xf64>

  %r = func.call @do_log(%a) : (tensor<5xf64>) -> tensor<5xf64>
  "check.expect_close"(%r, %expected) {max_ulp_difference = 32 : ui64} : (tensor<5xf64>, tensor<5xf64>) -> ()

  return
}
