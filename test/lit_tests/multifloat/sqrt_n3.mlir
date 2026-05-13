// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=3 concat-dimension=first" %s | FileCheck %s --check-prefix=FIRST
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=3 concat-dimension=last" %s | FileCheck %s --check-prefix=LAST
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=3 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=3 concat-dimension=last" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @do_sqrt(%a: tensor<4xf64>) -> tensor<4xf64> {
  %r = stablehlo.sqrt %a : tensor<4xf64>
  return %r : tensor<4xf64>
}

// 2 Newton steps at N=3 (K = N-1).
// FIRST-LABEL: func.func @do_sqrt
// FIRST: stablehlo.rsqrt %{{.*}} : tensor<1x4xf32>
// FIRST: stablehlo.concatenate {{.*}} dim = 0 : (tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>

// LAST-LABEL: func.func @do_sqrt
// LAST: stablehlo.rsqrt %{{.*}} : tensor<4x1xf32>
// LAST: stablehlo.concatenate {{.*}} : (tensor<4x1xf32>, tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x3xf32>

func.func @main() attributes {enzyme.no_multifloat} {
  %a = stablehlo.constant dense<[1.0, 0.1, 1.0e15, 2.0]> : tensor<4xf64>
  %expected = stablehlo.constant dense<[1.0, 0.31622776601683794, 31622776.60168379, 1.4142135623730951]> : tensor<4xf64>

  %r = func.call @do_sqrt(%a) : (tensor<4xf64>) -> tensor<4xf64>
  "check.expect_close"(%r, %expected) {max_ulp_difference = 8 : ui64} : (tensor<4xf64>, tensor<4xf64>) -> ()

  return
}
