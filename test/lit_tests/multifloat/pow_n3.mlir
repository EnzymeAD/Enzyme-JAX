// pow(x, y) at N=3 — composed as exp(y * log(|x|)) with boundary handling.
// Negative bases return NaN (parity-check via Floor not N-generalized yet).
//
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=3 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=3 concat-dimension=last" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @do_pow(%a: tensor<5xf64>, %b: tensor<5xf64>) -> tensor<5xf64> {
  %r = stablehlo.power %a, %b : tensor<5xf64>
  return %r : tensor<5xf64>
}

func.func @main() attributes {enzyme.no_multifloat} {
  // pow(2,3)=8, pow(10,0.5)=sqrt(10), pow(1,100)=1, pow(0.5,-2)=4, pow(5,0)=1
  %a = stablehlo.constant dense<[2.0, 10.0, 1.0, 0.5, 5.0]> : tensor<5xf64>
  %b = stablehlo.constant dense<[3.0, 0.5, 100.0, -2.0, 0.0]> : tensor<5xf64>
  %expected = stablehlo.constant dense<[8.0, 3.1622776601683795, 1.0, 4.0, 1.0]> : tensor<5xf64>

  %r = func.call @do_pow(%a, %b) : (tensor<5xf64>, tensor<5xf64>) -> tensor<5xf64>
  // Pow chains log + exp; expected error compounds. ~72 bit precision at N=3.
  "check.expect_close"(%r, %expected) {max_ulp_difference = 128 : ui64} : (tensor<5xf64>, tensor<5xf64>) -> ()

  return
}
