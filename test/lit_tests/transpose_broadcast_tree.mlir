// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=transpose_elementwise(1);transpose_broadcast_in_dim_to_broadcast_in_dim<16>" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// A contraction unrolled into a sum of products of broadcasts, used both as
// it is and transposed: the transposed use is the sum rebuilt on transposed
// broadcasts, and no transpose is left.
func.func @used_both_ways(%b: tensor<2x3xf64>, %x: tensor<4x3x2xf64>, %y: tensor<4x3x3xf64>) -> (tensor<4x3x3xf64>, tensor<4x3x3xf64>) {
  %b0 = stablehlo.slice %b [0:1, 0:3] : (tensor<2x3xf64>) -> tensor<1x3xf64>
  %x0 = stablehlo.slice %x [0:4, 0:3, 0:1] : (tensor<4x3x2xf64>) -> tensor<4x3x1xf64>
  %bx0 = stablehlo.broadcast_in_dim %x0, dims = [0, 2, 1] : (tensor<4x3x1xf64>) -> tensor<4x3x3xf64>
  %bb0 = stablehlo.broadcast_in_dim %b0, dims = [2, 1] : (tensor<1x3xf64>) -> tensor<4x3x3xf64>
  %m0 = stablehlo.multiply %bb0, %bx0 : tensor<4x3x3xf64>
  %b1 = stablehlo.slice %b [1:2, 0:3] : (tensor<2x3xf64>) -> tensor<1x3xf64>
  %x1 = stablehlo.slice %x [0:4, 0:3, 1:2] : (tensor<4x3x2xf64>) -> tensor<4x3x1xf64>
  %bx1 = stablehlo.broadcast_in_dim %x1, dims = [0, 2, 1] : (tensor<4x3x1xf64>) -> tensor<4x3x3xf64>
  %bb1 = stablehlo.broadcast_in_dim %b1, dims = [2, 1] : (tensor<1x3xf64>) -> tensor<4x3x3xf64>
  %m1 = stablehlo.multiply %bb1, %bx1 : tensor<4x3x3xf64>
  %s = stablehlo.add %m0, %m1 : tensor<4x3x3xf64>
  %t = stablehlo.transpose %s, dims = [0, 2, 1] : (tensor<4x3x3xf64>) -> tensor<4x3x3xf64>
  %u = stablehlo.multiply %s, %y : tensor<4x3x3xf64>
  return %t, %u : tensor<4x3x3xf64>, tensor<4x3x3xf64>
}

// CHECK-LABEL: func.func @used_both_ways(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<2x3xf64>, %[[V1:[^ ,:)]+]]: tensor<4x3x2xf64>, %[[V2:[^ ,:)]+]]: tensor<4x3x3xf64>) -> (tensor<4x3x3xf64>, tensor<4x3x3xf64>) {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.slice %[[V0]] [0:1, 0:3] : (tensor<2x3xf64>) -> tensor<1x3xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.slice %[[V1]] [0:4, 0:3, 0:1] : (tensor<4x3x2xf64>) -> tensor<4x3x1xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V4]], dims = [0, 2, 1] : (tensor<4x3x1xf64>) -> tensor<4x3x3xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V4]], dims = [0, 1, 2] : (tensor<4x3x1xf64>) -> tensor<4x3x3xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V3]], dims = [2, 1] : (tensor<1x3xf64>) -> tensor<4x3x3xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V3]], dims = [1, 2] : (tensor<1x3xf64>) -> tensor<4x3x3xf64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.multiply %[[V7]], %[[V5]] : tensor<4x3x3xf64>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = stablehlo.multiply %[[V8]], %[[V6]] : tensor<4x3x3xf64>
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = stablehlo.slice %[[V0]] [1:2, 0:3] : (tensor<2x3xf64>) -> tensor<1x3xf64>
// CHECK-NEXT: %[[V12:[^ ,:)]+]] = stablehlo.slice %[[V1]] [0:4, 0:3, 1:2] : (tensor<4x3x2xf64>) -> tensor<4x3x1xf64>
// CHECK-NEXT: %[[V13:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V12]], dims = [0, 2, 1] : (tensor<4x3x1xf64>) -> tensor<4x3x3xf64>
// CHECK-NEXT: %[[V14:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V12]], dims = [0, 1, 2] : (tensor<4x3x1xf64>) -> tensor<4x3x3xf64>
// CHECK-NEXT: %[[V15:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V11]], dims = [2, 1] : (tensor<1x3xf64>) -> tensor<4x3x3xf64>
// CHECK-NEXT: %[[V16:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V11]], dims = [1, 2] : (tensor<1x3xf64>) -> tensor<4x3x3xf64>
// CHECK-NEXT: %[[V17:[^ ,:)]+]] = stablehlo.multiply %[[V15]], %[[V13]] : tensor<4x3x3xf64>
// CHECK-NEXT: %[[V18:[^ ,:)]+]] = stablehlo.multiply %[[V16]], %[[V14]] : tensor<4x3x3xf64>
// CHECK-NEXT: %[[V19:[^ ,:)]+]] = stablehlo.add %[[V9]], %[[V17]] : tensor<4x3x3xf64>
// CHECK-NEXT: %[[V20:[^ ,:)]+]] = stablehlo.add %[[V10]], %[[V18]] : tensor<4x3x3xf64>
// CHECK-NEXT: %[[V21:[^ ,:)]+]] = stablehlo.multiply %[[V19]], %[[V2]] : tensor<4x3x3xf64>
// CHECK-NEXT: return %[[V20]], %[[V21]] : tensor<4x3x3xf64>, tensor<4x3x3xf64>
// CHECK-NEXT: }

// A leaf that is not a broadcast would be a transposed copy: the transpose
// stays where the sum is used twice.
func.func @data_leaf(%y: tensor<4x3x3xf64>, %z: tensor<4x3x3xf64>) -> (tensor<4x3x3xf64>, tensor<4x3x3xf64>) {
  %two = stablehlo.constant dense<2.0> : tensor<4x3x3xf64>
  %s = stablehlo.multiply %y, %two : tensor<4x3x3xf64>
  %t = stablehlo.transpose %s, dims = [0, 2, 1] : (tensor<4x3x3xf64>) -> tensor<4x3x3xf64>
  %u = stablehlo.multiply %s, %z : tensor<4x3x3xf64>
  return %t, %u : tensor<4x3x3xf64>, tensor<4x3x3xf64>
}

// CHECK-LABEL: func.func @data_leaf(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4x3x3xf64>, %[[V1:[^ ,:)]+]]: tensor<4x3x3xf64>) -> (tensor<4x3x3xf64>, tensor<4x3x3xf64>) {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<2.000000e+00> : tensor<4x3x3xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.multiply %[[V0]], %[[V2]] : tensor<4x3x3xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.transpose %[[V3]], dims = [0, 2, 1] : (tensor<4x3x3xf64>) -> tensor<4x3x3xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.multiply %[[V3]], %[[V1]] : tensor<4x3x3xf64>
// CHECK-NEXT: return %[[V4]], %[[V5]] : tensor<4x3x3xf64>, tensor<4x3x3xf64>
// CHECK-NEXT: }
