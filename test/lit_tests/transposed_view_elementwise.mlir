// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=transposed_view_elementwise" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s
// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=transpose_elementwise(1)" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s --check-prefix=PUSH

// A product of a contraction's result (a reduce of broadcasts, which can be
// had in any layout) and a slice of the transposed argument D: computed in
// D's layout, with the transpose on the result.
func.func @view(%D: tensor<8x8x4x4xf64>, %g: tensor<3x8x4x4xf64>, %B: tensor<4x3xf64>) -> tensor<4x8x4x4xf64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = stablehlo.broadcast_in_dim %g, dims = [0, 2, 4, 1] : (tensor<3x8x4x4xf64>) -> tensor<3x4x8x4x4xf64>
  %1 = stablehlo.broadcast_in_dim %B, dims = [3, 0] : (tensor<4x3xf64>) -> tensor<3x4x8x4x4xf64>
  %2 = stablehlo.multiply %0, %1 : tensor<3x4x8x4x4xf64>
  %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3x4x8x4x4xf64>, tensor<f64>) -> tensor<4x8x4x4xf64>
  %4 = stablehlo.transpose %D, dims = [3, 0, 1, 2] : (tensor<8x8x4x4xf64>) -> tensor<4x8x8x4xf64>
  %5 = stablehlo.slice %4 [0:4, 0:8, 4:8, 0:4] : (tensor<4x8x8x4xf64>) -> tensor<4x8x4x4xf64>
  %6 = stablehlo.multiply %3, %5 : tensor<4x8x4x4xf64>
  return %6 : tensor<4x8x4x4xf64>
}

// CHECK-LABEL: func.func @view(
// CHECK-SAME: %[[D:[^ ,:)]+]]: tensor<8x8x4x4xf64>, %[[G:[^ ,:)]+]]: tensor<3x8x4x4xf64>, %[[B:[^ ,:)]+]]: tensor<4x3xf64>) -> tensor<4x8x4x4xf64> {
// CHECK-NEXT: %[[Z:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[G0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[G]], dims = [0, 2, 4, 1] : (tensor<3x8x4x4xf64>) -> tensor<3x4x8x4x4xf64>
// CHECK-NEXT: %[[B0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[B]], dims = [3, 0] : (tensor<4x3xf64>) -> tensor<3x4x8x4x4xf64>
// CHECK-NEXT: %[[M:[^ ,:)]+]] = stablehlo.multiply %[[G0]], %[[B0]] : tensor<3x4x8x4x4xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.reduce(%[[M]] init: %[[Z]]) applies stablehlo.add across dimensions = [0] : (tensor<3x4x8x4x4xf64>, tensor<f64>) -> tensor<4x8x4x4xf64>
// CHECK-NEXT: %[[RT:[^ ,:)]+]] = stablehlo.transpose %[[R]], dims = [1, 2, 3, 0] : (tensor<4x8x4x4xf64>) -> tensor<8x4x4x4xf64>
// CHECK-NEXT: %[[S:[^ ,:)]+]] = stablehlo.slice %[[D]] [0:8, 4:8, 0:4, 0:4] : (tensor<8x8x4x4xf64>) -> tensor<8x4x4x4xf64>
// CHECK-NEXT: %[[P:[^ ,:)]+]] = stablehlo.multiply %[[RT]], %[[S]] : tensor<8x4x4x4xf64>
// CHECK-NEXT: %[[T:[^ ,:)]+]] = stablehlo.transpose %[[P]], dims = [3, 0, 1, 2] : (tensor<8x4x4x4xf64>) -> tensor<4x8x4x4xf64>
// CHECK-NEXT: return %[[T]] : tensor<4x8x4x4xf64>
// CHECK-NEXT: }

// The other operand is an argument too: no layout is free, left alone.
func.func @two_views(%A: tensor<8x4xf64>, %C: tensor<4x8xf64>) -> tensor<4x8xf64> {
  %0 = stablehlo.transpose %A, dims = [1, 0] : (tensor<8x4xf64>) -> tensor<4x8xf64>
  %1 = stablehlo.add %0, %C : tensor<4x8xf64>
  return %1 : tensor<4x8xf64>
}

// CHECK-LABEL: func.func @two_views(
// CHECK-SAME: %[[A:[^ ,:)]+]]: tensor<8x4xf64>, %[[C:[^ ,:)]+]]: tensor<4x8xf64>) -> tensor<4x8xf64> {
// CHECK-NEXT: %[[AT:[^ ,:)]+]] = stablehlo.transpose %[[A]], dims = [1, 0] : (tensor<8x4xf64>) -> tensor<4x8xf64>
// CHECK-NEXT: %[[S:[^ ,:)]+]] = stablehlo.add %[[AT]], %[[C]] : tensor<4x8xf64>
// CHECK-NEXT: return %[[S]] : tensor<4x8xf64>
// CHECK-NEXT: }

// Both operands read through the same transpose: left to the transpose
// patterns (transpose_elementwise pushes a result transpose onto both).
func.func @same_transpose(%A: tensor<8x4xf64>, %C: tensor<8x4xf64>) -> tensor<4x8xf64> {
  %0 = stablehlo.transpose %A, dims = [1, 0] : (tensor<8x4xf64>) -> tensor<4x8xf64>
  %1 = stablehlo.transpose %C, dims = [1, 0] : (tensor<8x4xf64>) -> tensor<4x8xf64>
  %2 = stablehlo.add %0, %1 : tensor<4x8xf64>
  return %2 : tensor<4x8xf64>
}

// CHECK-LABEL: func.func @same_transpose(
// CHECK-SAME: %[[A:[^ ,:)]+]]: tensor<8x4xf64>, %[[C:[^ ,:)]+]]: tensor<8x4xf64>) -> tensor<4x8xf64> {
// CHECK-NEXT: %[[AT:[^ ,:)]+]] = stablehlo.transpose %[[A]], dims = [1, 0] : (tensor<8x4xf64>) -> tensor<4x8xf64>
// CHECK-NEXT: %[[CT:[^ ,:)]+]] = stablehlo.transpose %[[C]], dims = [1, 0] : (tensor<8x4xf64>) -> tensor<4x8xf64>
// CHECK-NEXT: %[[S:[^ ,:)]+]] = stablehlo.add %[[AT]], %[[CT]] : tensor<4x8xf64>
// CHECK-NEXT: return %[[S]] : tensor<4x8xf64>
// CHECK-NEXT: }

// transpose_elementwise: the transpose of a sum of an argument and a
// broadcast stays on the sum (the sum computes in the argument's layout);
// the transpose of a sum of two arguments is pushed onto both as before.
func.func @push(%A: tensor<4x8xf64>, %C: tensor<4x8xf64>, %b: tensor<8xf64>) -> (tensor<8x4xf64>, tensor<8x4xf64>) {
  %0 = stablehlo.broadcast_in_dim %b, dims = [1] : (tensor<8xf64>) -> tensor<4x8xf64>
  %1 = stablehlo.add %A, %0 : tensor<4x8xf64>
  %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<4x8xf64>) -> tensor<8x4xf64>
  %3 = stablehlo.add %A, %C : tensor<4x8xf64>
  %4 = stablehlo.transpose %3, dims = [1, 0] : (tensor<4x8xf64>) -> tensor<8x4xf64>
  return %2, %4 : tensor<8x4xf64>, tensor<8x4xf64>
}

// PUSH-LABEL: func.func @push(
// PUSH-SAME: %[[A:[^ ,:)]+]]: tensor<4x8xf64>, %[[C:[^ ,:)]+]]: tensor<4x8xf64>, %[[B:[^ ,:)]+]]: tensor<8xf64>) -> (tensor<8x4xf64>, tensor<8x4xf64>) {
// PUSH-NEXT: %[[CT:[^ ,:)]+]] = stablehlo.transpose %[[C]], dims = [1, 0] : (tensor<4x8xf64>) -> tensor<8x4xf64>
// PUSH-NEXT: %[[AT:[^ ,:)]+]] = stablehlo.transpose %[[A]], dims = [1, 0] : (tensor<4x8xf64>) -> tensor<8x4xf64>
// PUSH-NEXT: %[[B0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[B]], dims = [1] : (tensor<8xf64>) -> tensor<4x8xf64>
// PUSH-NEXT: %[[S1:[^ ,:)]+]] = stablehlo.add %[[A]], %[[B0]] : tensor<4x8xf64>
// PUSH-NEXT: %[[T1:[^ ,:)]+]] = stablehlo.transpose %[[S1]], dims = [1, 0] : (tensor<4x8xf64>) -> tensor<8x4xf64>
// PUSH-NEXT: %[[S2:[^ ,:)]+]] = stablehlo.add %[[AT]], %[[CT]] : tensor<8x4xf64>
// PUSH-NEXT: return %[[T1]], %[[S2]] : tensor<8x4xf64>, tensor<8x4xf64>
// PUSH-NEXT: }
