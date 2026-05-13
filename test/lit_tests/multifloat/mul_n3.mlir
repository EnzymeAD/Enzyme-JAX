// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=3 concat-dimension=first" %s | FileCheck %s --check-prefix=FIRST
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=3 concat-dimension=last" %s | FileCheck %s --check-prefix=LAST
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=3 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=3 concat-dimension=last" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @do_mul(%a: tensor<4xf64>, %b: tensor<4xf64>) -> tensor<4xf64> {
  %r = stablehlo.multiply %a, %b : tensor<4xf64>
  return %r : tensor<4xf64>
}

// FIRST-LABEL: func.func @do_mul
// FIRST: stablehlo.concatenate {{.*}} dim = 0 : (tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
// FIRST: stablehlo.multiply %{{.*}} : tensor<1x4xf32>

// LAST-LABEL: func.func @do_mul
// LAST: stablehlo.concatenate {{.*}} : (tensor<4x1xf32>, tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x3xf32>
// LAST: stablehlo.multiply %{{.*}} : tensor<4x1xf32>

func.func @main() attributes {enzyme.no_multifloat} {
  %a = stablehlo.constant dense<[1.0, 0.1, 1.0e15, 2.0]> : tensor<4xf64>
  %b = stablehlo.constant dense<[2.0, 0.2, 1.0, -2.0]> : tensor<4xf64>
  %expected = stablehlo.constant dense<[2.0, 0.020000000000000004, 1.0e15, -4.0]> : tensor<4xf64>

  %r = func.call @do_mul(%a, %b) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  "check.expect_close"(%r, %expected) {max_ulp_difference = 4 : ui64} : (tensor<4xf64>, tensor<4xf64>) -> ()

  return
}
