// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%arg0: tensor<2x2xf32>) -> tensor<1x2xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = "stablehlo.reduce_window"(%arg0, %cst) <{window_dimensions = array<i64: 2, 1>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<2x2xf32>, tensor<f32>) -> tensor<1x2xf32>
    return %0 : tensor<1x2xf32>
  }
}

// CHECK: module {
// CHECK-NEXT:   func.func @main(%arg0: tensor<2x2xf32>) -> tensor<1x2xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
// CHECK-NEXT:     %1 = stablehlo.reshape %0 : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK-NEXT:     return %1 : tensor<1x2xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
