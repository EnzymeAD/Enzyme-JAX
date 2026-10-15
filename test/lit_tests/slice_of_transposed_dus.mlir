// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=slice_of_dynamic_update" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// The slice reads, through the transpose, only what the update wrote.
func.func @inside(%x: tensor<4x8x6xf64>, %u: tensor<4x2x3xf64>) -> tensor<4x3x2xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c2 = stablehlo.constant dense<2> : tensor<i32>
  %c3 = stablehlo.constant dense<3> : tensor<i32>
  %d = stablehlo.dynamic_update_slice %x, %u, %c0, %c2, %c3 : (tensor<4x8x6xf64>, tensor<4x2x3xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x8x6xf64>
  %t = stablehlo.transpose %d, dims = [0, 2, 1] : (tensor<4x8x6xf64>) -> tensor<4x6x8xf64>
  %s = stablehlo.slice %t [0:4, 3:6, 2:4] : (tensor<4x6x8xf64>) -> tensor<4x3x2xf64>
  return %s : tensor<4x3x2xf64>
}

// CHECK-LABEL: func.func @inside(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4x8x6xf64>, %[[V1:[^ ,:)]+]]: tensor<4x2x3xf64>) -> tensor<4x3x2xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.slice %[[V1]] [0:4, 0:2, 0:3] : (tensor<4x2x3xf64>) -> tensor<4x2x3xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.transpose %[[V2]], dims = [0, 2, 1] : (tensor<4x2x3xf64>) -> tensor<4x3x2xf64>
// CHECK-NEXT: return %[[V3]] : tensor<4x3x2xf64>
// CHECK-NEXT: }

// The slice reads, through the transpose, none of what the update wrote.
func.func @disjoint(%x: tensor<4x8x6xf64>, %u: tensor<4x2x3xf64>) -> tensor<4x3x2xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c2 = stablehlo.constant dense<2> : tensor<i32>
  %c3 = stablehlo.constant dense<3> : tensor<i32>
  %d = stablehlo.dynamic_update_slice %x, %u, %c0, %c2, %c3 : (tensor<4x8x6xf64>, tensor<4x2x3xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x8x6xf64>
  %t = stablehlo.transpose %d, dims = [0, 2, 1] : (tensor<4x8x6xf64>) -> tensor<4x6x8xf64>
  %s = stablehlo.slice %t [0:4, 0:3, 5:7] : (tensor<4x6x8xf64>) -> tensor<4x3x2xf64>
  return %s : tensor<4x3x2xf64>
}

// CHECK-LABEL: func.func @disjoint(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4x8x6xf64>, %[[V1:[^ ,:)]+]]: tensor<4x2x3xf64>) -> tensor<4x3x2xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.slice %[[V0]] [0:4, 5:7, 0:3] : (tensor<4x8x6xf64>) -> tensor<4x2x3xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.transpose %[[V2]], dims = [0, 2, 1] : (tensor<4x2x3xf64>) -> tensor<4x3x2xf64>
// CHECK-NEXT: return %[[V3]] : tensor<4x3x2xf64>
// CHECK-NEXT: }

// Stays: the slice reads both written and unwritten elements.
func.func @straddles(%x: tensor<4x8x6xf64>, %u: tensor<4x2x3xf64>) -> tensor<4x3x2xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c2 = stablehlo.constant dense<2> : tensor<i32>
  %c3 = stablehlo.constant dense<3> : tensor<i32>
  %d = stablehlo.dynamic_update_slice %x, %u, %c0, %c2, %c3 : (tensor<4x8x6xf64>, tensor<4x2x3xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x8x6xf64>
  %t = stablehlo.transpose %d, dims = [0, 2, 1] : (tensor<4x8x6xf64>) -> tensor<4x6x8xf64>
  %s = stablehlo.slice %t [0:4, 2:5, 2:4] : (tensor<4x6x8xf64>) -> tensor<4x3x2xf64>
  return %s : tensor<4x3x2xf64>
}

// CHECK-LABEL: func.func @straddles(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4x8x6xf64>, %[[V1:[^ ,:)]+]]: tensor<4x2x3xf64>) -> tensor<4x3x2xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<2> : tensor<i32>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.constant dense<3> : tensor<i32>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V0]], %[[V1]], %[[V2]], %[[V3]], %[[V4]] : (tensor<4x8x6xf64>, tensor<4x2x3xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x8x6xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.transpose %[[V5]], dims = [0, 2, 1] : (tensor<4x8x6xf64>) -> tensor<4x6x8xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.slice %[[V6]] [0:4, 2:5, 2:4] : (tensor<4x6x8xf64>) -> tensor<4x3x2xf64>
// CHECK-NEXT: return %[[V7]] : tensor<4x3x2xf64>
// CHECK-NEXT: }

// Stays: the transpose has another use, which still needs it whole.
func.func @transpose_two_uses(%x: tensor<4x8x6xf64>, %u: tensor<4x2x3xf64>) -> (tensor<4x3x2xf64>, tensor<4x6x8xf64>) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c2 = stablehlo.constant dense<2> : tensor<i32>
  %c3 = stablehlo.constant dense<3> : tensor<i32>
  %d = stablehlo.dynamic_update_slice %x, %u, %c0, %c2, %c3 : (tensor<4x8x6xf64>, tensor<4x2x3xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x8x6xf64>
  %t = stablehlo.transpose %d, dims = [0, 2, 1] : (tensor<4x8x6xf64>) -> tensor<4x6x8xf64>
  %s = stablehlo.slice %t [0:4, 3:6, 2:4] : (tensor<4x6x8xf64>) -> tensor<4x3x2xf64>
  return %s, %t : tensor<4x3x2xf64>, tensor<4x6x8xf64>
}

// CHECK-LABEL: func.func @transpose_two_uses(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4x8x6xf64>, %[[V1:[^ ,:)]+]]: tensor<4x2x3xf64>) -> (tensor<4x3x2xf64>, tensor<4x6x8xf64>) {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<2> : tensor<i32>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.constant dense<3> : tensor<i32>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V0]], %[[V1]], %[[V2]], %[[V3]], %[[V4]] : (tensor<4x8x6xf64>, tensor<4x2x3xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x8x6xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.transpose %[[V5]], dims = [0, 2, 1] : (tensor<4x8x6xf64>) -> tensor<4x6x8xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.slice %[[V6]] [0:4, 3:6, 2:4] : (tensor<4x6x8xf64>) -> tensor<4x3x2xf64>
// CHECK-NEXT: return %[[V7]], %[[V6]] : tensor<4x3x2xf64>, tensor<4x6x8xf64>
// CHECK-NEXT: }
