// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=dus_dus" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func @inner_covers_outer(%arg0: tensor<4x1x8x8xf32>, %arg1: tensor<4x1x4x8xf32>, %arg2: tensor<4x1x4x4xf32>) -> tensor<4x1x8x8xf32> {
    %c = stablehlo.constant dense<4> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %c_0, %c_0, %c_0, %c_0 : (tensor<4x1x8x8xf32>, tensor<4x1x4x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x1x8x8xf32>
    %1 = stablehlo.dynamic_update_slice %0, %arg2, %c_0, %c_0, %c_0, %c : (tensor<4x1x8x8xf32>, tensor<4x1x4x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x1x8x8xf32>
    return %1 : tensor<4x1x8x8xf32>
  }
}

// CHECK: func.func @inner_covers_outer(%arg0: tensor<4x1x8x8xf32>, %arg1: tensor<4x1x4x8xf32>, %arg2: tensor<4x1x4x4xf32>) -> tensor<4x1x8x8xf32> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<4> : tensor<i32>
// CHECK-NEXT:   %0 = stablehlo.dynamic_update_slice %arg1, %arg2, %c, %c, %c, %c_0 : (tensor<4x1x4x8xf32>, tensor<4x1x4x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x1x4x8xf32>
// CHECK-NEXT:   %1 = stablehlo.dynamic_update_slice %arg0, %0, %c, %c, %c, %c : (tensor<4x1x8x8xf32>, tensor<4x1x4x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x1x8x8xf32>
// CHECK-NEXT:   return %1 : tensor<4x1x8x8xf32>
// CHECK-NEXT: }

module {
  func.func @outer_covers_inner(%arg0: tensor<4x1x8x8xf32>, %arg1: tensor<4x1x4x6xf32>, %arg2: tensor<4x1x4x3xf32>) -> tensor<4x1x8x8xf32> {
    %c = stablehlo.constant dense<4> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %c_1 = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.dynamic_update_slice %arg0, %arg2, %c_0, %c_0, %c_0, %c : (tensor<4x1x8x8xf32>, tensor<4x1x4x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x1x8x8xf32>
    %1 = stablehlo.dynamic_update_slice %0, %arg1, %c_0, %c_0, %c_0, %c_1 : (tensor<4x1x8x8xf32>, tensor<4x1x4x6xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x1x8x8xf32>
    return %1 : tensor<4x1x8x8xf32>
  }
}

// CHECK: func.func @outer_covers_inner(%arg0: tensor<4x1x8x8xf32>, %arg1: tensor<4x1x4x6xf32>, %arg2: tensor<4x1x4x3xf32>) -> tensor<4x1x8x8xf32> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:   %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %c, %c, %c, %c_0 : (tensor<4x1x8x8xf32>, tensor<4x1x4x6xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x1x8x8xf32>
// CHECK-NEXT:   return %0 : tensor<4x1x8x8xf32>
// CHECK-NEXT: }
