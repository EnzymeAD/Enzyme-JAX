// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=cse_reduce_transposed" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Two reductions of the same product of B and x, each read through
// broadcasts of a different layout (the second's input is a transpose of
// the first's, the reduced dimension staying put): the second is a
// transpose of the first.
func.func @transposed(%B: tensor<3x4xf64>, %x: tensor<8x4x3xf64>) -> (tensor<8x4x4xf64>, tensor<4x8x4xf64>) {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %b0 = stablehlo.broadcast_in_dim %B, dims = [1, 3] : (tensor<3x4xf64>) -> tensor<8x3x4x4xf64>
  %x0 = stablehlo.broadcast_in_dim %x, dims = [0, 2, 1] : (tensor<8x4x3xf64>) -> tensor<8x3x4x4xf64>
  %m0 = stablehlo.multiply %b0, %x0 : tensor<8x3x4x4xf64>
  %r0 = stablehlo.reduce(%m0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x3x4x4xf64>, tensor<f64>) -> tensor<8x4x4xf64>
  %b1 = stablehlo.broadcast_in_dim %B, dims = [1, 0] : (tensor<3x4xf64>) -> tensor<4x3x8x4xf64>
  %x1 = stablehlo.broadcast_in_dim %x, dims = [2, 3, 1] : (tensor<8x4x3xf64>) -> tensor<4x3x8x4xf64>
  %m1 = stablehlo.multiply %b1, %x1 : tensor<4x3x8x4xf64>
  %r1 = stablehlo.reduce(%m1 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x3x8x4xf64>, tensor<f64>) -> tensor<4x8x4xf64>
  return %r0, %r1 : tensor<8x4x4xf64>, tensor<4x8x4xf64>
}

// CHECK-LABEL: func.func @transposed(
// CHECK-SAME: %[[B:[^ ,:)]+]]: tensor<3x4xf64>, %[[X:[^ ,:)]+]]: tensor<8x4x3xf64>) -> (tensor<8x4x4xf64>, tensor<4x8x4xf64>) {
// CHECK-NEXT: %[[Z:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[B0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[B]], dims = [1, 3] : (tensor<3x4xf64>) -> tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[X0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[X]], dims = [0, 2, 1] : (tensor<8x4x3xf64>) -> tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[M:[^ ,:)]+]] = stablehlo.multiply %[[B0]], %[[X0]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.reduce(%[[M]] init: %[[Z]]) applies stablehlo.add across dimensions = [1] : (tensor<8x3x4x4xf64>, tensor<f64>) -> tensor<8x4x4xf64>
// CHECK-NEXT: %[[T:[^ ,:)]+]] = stablehlo.transpose %[[R]], dims = [2, 0, 1] : (tensor<8x4x4xf64>) -> tensor<4x8x4xf64>
// CHECK-NEXT: return %[[R]], %[[T]] : tensor<8x4x4xf64>, tensor<4x8x4xf64>
// CHECK-NEXT: }

// The same values read with x's dimension of 4 and B's column coinciding
// in one input (an elementwise product along it) and separate in the
// other (an outer product): different computations, left alone.
func.func @different(%B: tensor<3x4xf64>, %x: tensor<3x1x8x4x3xf64>) -> (tensor<3x8x4x4xf64>, tensor<3x8x4x4xf64>) {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %b0 = stablehlo.broadcast_in_dim %B, dims = [1, 3] : (tensor<3x4xf64>) -> tensor<3x3x8x4x4xf64>
  %x0 = stablehlo.broadcast_in_dim %x, dims = [0, 3, 2, 4, 1] : (tensor<3x1x8x4x3xf64>) -> tensor<3x3x8x4x4xf64>
  %m0 = stablehlo.multiply %b0, %x0 : tensor<3x3x8x4x4xf64>
  %r0 = stablehlo.reduce(%m0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<3x3x8x4x4xf64>, tensor<f64>) -> tensor<3x8x4x4xf64>
  %x1 = stablehlo.broadcast_in_dim %x, dims = [0, 4, 2, 3, 1] : (tensor<3x1x8x4x3xf64>) -> tensor<3x3x8x4x4xf64>
  %m1 = stablehlo.multiply %b0, %x1 : tensor<3x3x8x4x4xf64>
  %r1 = stablehlo.reduce(%m1 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<3x3x8x4x4xf64>, tensor<f64>) -> tensor<3x8x4x4xf64>
  return %r0, %r1 : tensor<3x8x4x4xf64>, tensor<3x8x4x4xf64>
}

// CHECK-LABEL: func.func @different(
// CHECK-SAME: %[[B:[^ ,:)]+]]: tensor<3x4xf64>, %[[X:[^ ,:)]+]]: tensor<3x1x8x4x3xf64>) -> (tensor<3x8x4x4xf64>, tensor<3x8x4x4xf64>) {
// CHECK-NEXT: %[[Z:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[B0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[B]], dims = [1, 3] : (tensor<3x4xf64>) -> tensor<3x3x8x4x4xf64>
// CHECK-NEXT: %[[X0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[X]], dims = [0, 3, 2, 4, 1] : (tensor<3x1x8x4x3xf64>) -> tensor<3x3x8x4x4xf64>
// CHECK-NEXT: %[[M0:[^ ,:)]+]] = stablehlo.multiply %[[B0]], %[[X0]] : tensor<3x3x8x4x4xf64>
// CHECK-NEXT: %[[R0:[^ ,:)]+]] = stablehlo.reduce(%[[M0]] init: %[[Z]]) applies stablehlo.add across dimensions = [1] : (tensor<3x3x8x4x4xf64>, tensor<f64>) -> tensor<3x8x4x4xf64>
// CHECK-NEXT: %[[X1:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[X]], dims = [0, 4, 2, 3, 1] : (tensor<3x1x8x4x3xf64>) -> tensor<3x3x8x4x4xf64>
// CHECK-NEXT: %[[M1:[^ ,:)]+]] = stablehlo.multiply %[[B0]], %[[X1]] : tensor<3x3x8x4x4xf64>
// CHECK-NEXT: %[[R1:[^ ,:)]+]] = stablehlo.reduce(%[[M1]] init: %[[Z]]) applies stablehlo.add across dimensions = [1] : (tensor<3x3x8x4x4xf64>, tensor<f64>) -> tensor<3x8x4x4xf64>
// CHECK-NEXT: return %[[R0]], %[[R1]] : tensor<3x8x4x4xf64>, tensor<3x8x4x4xf64>
// CHECK-NEXT: }
