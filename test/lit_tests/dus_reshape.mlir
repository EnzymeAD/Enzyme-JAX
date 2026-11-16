// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=dus_reshape" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Rows 3 and 4 of the merged view are row 1, columns 0 and 1 of %x.
func.func @merge(%x: tensor<2x4x3xf64>, %u: tensor<2x2xf64>) -> tensor<2x12xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c3 = stablehlo.constant dense<3> : tensor<i32>
  %r = stablehlo.reshape %x : (tensor<2x4x3xf64>) -> tensor<2x12xf64>
  %d = stablehlo.dynamic_update_slice %r, %u, %c0, %c3 : (tensor<2x12xf64>, tensor<2x2xf64>, tensor<i32>, tensor<i32>) -> tensor<2x12xf64>
  return %d : tensor<2x12xf64>
}

// CHECK-LABEL: func.func @merge(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<2x4x3xf64>, %[[V1:[^ ,:)]+]]: tensor<2x2xf64>) -> tensor<2x12xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<2x2xf64>) -> tensor<2x1x2xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V0]], %[[V4]], %[[V2]], %[[V3]], %[[V2]] : (tensor<2x4x3xf64>, tensor<2x1x2xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x4x3xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.reshape %[[V5]] : (tensor<2x4x3xf64>) -> tensor<2x12xf64>
// CHECK-NEXT: return %[[V6]] : tensor<2x12xf64>
// CHECK-NEXT: }

// One row of the split view is a run of the merged dimension.
func.func @split(%x: tensor<2x12xf64>, %u: tensor<1x1x3xf64>, %i: tensor<i32>) -> tensor<2x4x3xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c2 = stablehlo.constant dense<2> : tensor<i32>
  %r = stablehlo.reshape %x : (tensor<2x12xf64>) -> tensor<2x4x3xf64>
  %d = stablehlo.dynamic_update_slice %r, %u, %i, %c2, %c0 : (tensor<2x4x3xf64>, tensor<1x1x3xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x4x3xf64>
  return %d : tensor<2x4x3xf64>
}

// CHECK-LABEL: func.func @split(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<2x12xf64>, %[[V1:[^ ,:)]+]]: tensor<1x1x3xf64>, %[[V2:[^ ,:)]+]]: tensor<i32>) -> tensor<2x4x3xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<6> : tensor<i32>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<1x1x3xf64>) -> tensor<1x3xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V0]], %[[V4]], %[[V2]], %[[V3]] : (tensor<2x12xf64>, tensor<1x3xf64>, tensor<i32>, tensor<i32>) -> tensor<2x12xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.reshape %[[V5]] : (tensor<2x12xf64>) -> tensor<2x4x3xf64>
// CHECK-NEXT: return %[[V6]] : tensor<2x4x3xf64>
// CHECK-NEXT: }

// The update crosses a row of %x: not a box of %x.
func.func @crosses_row(%x: tensor<2x4x3xf64>, %u: tensor<2x2xf64>) -> tensor<2x12xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c2 = stablehlo.constant dense<2> : tensor<i32>
  %r = stablehlo.reshape %x : (tensor<2x4x3xf64>) -> tensor<2x12xf64>
  %d = stablehlo.dynamic_update_slice %r, %u, %c0, %c2 : (tensor<2x12xf64>, tensor<2x2xf64>, tensor<i32>, tensor<i32>) -> tensor<2x12xf64>
  return %d : tensor<2x12xf64>
}

// CHECK-LABEL: func.func @crosses_row(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<2x4x3xf64>, %[[V1:[^ ,:)]+]]: tensor<2x2xf64>) -> tensor<2x12xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<2> : tensor<i32>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<2x4x3xf64>) -> tensor<2x12xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V4]], %[[V1]], %[[V2]], %[[V3]] : (tensor<2x12xf64>, tensor<2x2xf64>, tensor<i32>, tensor<i32>) -> tensor<2x12xf64>
// CHECK-NEXT: return %[[V5]] : tensor<2x12xf64>
// CHECK-NEXT: }

// A unit dimension only: left to reshape_dus.
func.func @unit_dim(%x: tensor<4x3xf64>, %u: tensor<1x2x3xf64>) -> tensor<1x4x3xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %r = stablehlo.reshape %x : (tensor<4x3xf64>) -> tensor<1x4x3xf64>
  %d = stablehlo.dynamic_update_slice %r, %u, %c0, %c0, %c0 : (tensor<1x4x3xf64>, tensor<1x2x3xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x4x3xf64>
  return %d : tensor<1x4x3xf64>
}

// CHECK-LABEL: func.func @unit_dim(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4x3xf64>, %[[V1:[^ ,:)]+]]: tensor<1x2x3xf64>) -> tensor<1x4x3xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<4x3xf64>) -> tensor<1x4x3xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V3]], %[[V1]], %[[V2]], %[[V2]], %[[V2]] : (tensor<1x4x3xf64>, tensor<1x2x3xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x4x3xf64>
// CHECK-NEXT: return %[[V4]] : tensor<1x4x3xf64>
// CHECK-NEXT: }
