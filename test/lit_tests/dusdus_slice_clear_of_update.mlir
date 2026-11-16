// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// A slice of the inner of two updates, clear of the outer one, reads the same
// from either. DUSDUSSubsuming used to move it onto the outer update even when
// it could not replace that update, and SliceOfDynamicUpdate moved it back,
// forever. The slice stays where it is.

func.func @slice_clear_of_update(%base: tensor<4x4xf64>, %u1: tensor<3x1xf64>, %u2: tensor<3x1xf64>) -> (tensor<4x4xf64>, tensor<4x1xf64>) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c2 = stablehlo.constant dense<2> : tensor<i32>
  %d2 = stablehlo.dynamic_update_slice %base, %u1, %c0, %c0 : (tensor<4x4xf64>, tensor<3x1xf64>, tensor<i32>, tensor<i32>) -> tensor<4x4xf64>
  %d = stablehlo.dynamic_update_slice %d2, %u2, %c0, %c2 : (tensor<4x4xf64>, tensor<3x1xf64>, tensor<i32>, tensor<i32>) -> tensor<4x4xf64>
  %s = stablehlo.slice %d2 [0:4, 0:1] : (tensor<4x4xf64>) -> tensor<4x1xf64>
  return %d, %s : tensor<4x4xf64>, tensor<4x1xf64>
}

// CHECK-LABEL: func.func @slice_clear_of_update(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4x4xf64>, %[[V1:[^ ,:)]+]]: tensor<3x1xf64>, %[[V2:[^ ,:)]+]]: tensor<3x1xf64>) -> (tensor<4x4xf64>, tensor<4x1xf64>) {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.constant dense<2> : tensor<i32>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V0]], %[[V1]], %[[V3]], %[[V3]] : (tensor<4x4xf64>, tensor<3x1xf64>, tensor<i32>, tensor<i32>) -> tensor<4x4xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V5]], %[[V2]], %[[V3]], %[[V4]] : (tensor<4x4xf64>, tensor<3x1xf64>, tensor<i32>, tensor<i32>) -> tensor<4x4xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.slice %[[V5]] [0:4, 0:1] : (tensor<4x4xf64>) -> tensor<4x1xf64>
// CHECK-NEXT: return %[[V6]], %[[V7]] : tensor<4x4xf64>, tensor<4x1xf64>
// CHECK-NEXT: }
