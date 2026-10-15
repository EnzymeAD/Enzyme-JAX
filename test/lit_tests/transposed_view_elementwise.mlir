// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=transposed_view_elementwise" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s
// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=transpose_elementwise(1)" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s --check-prefix=PUSH
// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=transposed_view_elementwise;transpose_elementwise(1);transpose_transpose" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s --check-prefix=BOTH

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
// broadcast stays on the sum (the sum computes in the argument's layout), and
// so does the transpose of a sum of two arguments (pushed, it would copy both).
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
// PUSH-NEXT: %[[B0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[B]], dims = [1] : (tensor<8xf64>) -> tensor<4x8xf64>
// PUSH-NEXT: %[[S1:[^ ,:)]+]] = stablehlo.add %[[A]], %[[B0]] : tensor<4x8xf64>
// PUSH-NEXT: %[[T1:[^ ,:)]+]] = stablehlo.transpose %[[S1]], dims = [1, 0] : (tensor<4x8xf64>) -> tensor<8x4xf64>
// PUSH-NEXT: %[[S2:[^ ,:)]+]] = stablehlo.add %[[A]], %[[C]] : tensor<4x8xf64>
// PUSH-NEXT: %[[T2:[^ ,:)]+]] = stablehlo.transpose %[[S2]], dims = [1, 0] : (tensor<4x8xf64>) -> tensor<8x4xf64>
// PUSH-NEXT: return %[[T1]], %[[T2]] : tensor<8x4xf64>, tensor<8x4xf64>
// PUSH-NEXT: }

// The two patterns together: the sum %5 absorbs the transpose (it cancels
// the one on the product %2), so transpose_elementwise does not push the
// transpose of %8 onto %7, a product of a view, which
// transposed_view_elementwise would move back; the transpose of %5 is
// pushed in, where it cancels. One transpose is left, on the sum of the
// two products, computed in the views' layout.
func.func @chain(%v1: tensor<2x3x3xf64>, %v2: tensor<2x3x3xf64>, %b: tensor<1xf64>, %c: tensor<1xf64>, %g: tensor<2x3x3xf64>) -> tensor<2x3x3xf64> {
  %0 = stablehlo.broadcast_in_dim %b, dims = [0] : (tensor<1xf64>) -> tensor<2x3x3xf64>
  %1 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
  %2 = stablehlo.multiply %v1, %1 : tensor<2x3x3xf64>
  %3 = stablehlo.transpose %2, dims = [0, 2, 1] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
  %4 = stablehlo.broadcast_in_dim %c, dims = [0] : (tensor<1xf64>) -> tensor<2x3x3xf64>
  %5 = stablehlo.add %4, %3 : tensor<2x3x3xf64>
  %6 = stablehlo.transpose %5, dims = [0, 2, 1] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
  %7 = stablehlo.multiply %v2, %1 : tensor<2x3x3xf64>
  %8 = stablehlo.add %6, %7 : tensor<2x3x3xf64>
  %9 = stablehlo.transpose %8, dims = [0, 2, 1] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
  %10 = stablehlo.add %9, %g : tensor<2x3x3xf64>
  return %10 : tensor<2x3x3xf64>
}

// BOTH-LABEL: func.func @chain(
// BOTH-SAME: %[[V1:[^ ,:)]+]]: tensor<2x3x3xf64>, %[[V2:[^ ,:)]+]]: tensor<2x3x3xf64>, %[[B:[^ ,:)]+]]: tensor<1xf64>, %[[C:[^ ,:)]+]]: tensor<1xf64>, %[[G:[^ ,:)]+]]: tensor<2x3x3xf64>) -> tensor<2x3x3xf64> {
// BOTH-NEXT: %[[B0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[B]], dims = [0] : (tensor<1xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[BT:[^ ,:)]+]] = stablehlo.transpose %[[B0]], dims = [0, 2, 1] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[M1:[^ ,:)]+]] = stablehlo.multiply %[[V1]], %[[BT]] : tensor<2x3x3xf64>
// BOTH-NEXT: %[[C0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[C]], dims = [0] : (tensor<1xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[CT:[^ ,:)]+]] = stablehlo.transpose %[[C0]], dims = [0, 2, 1] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[S1:[^ ,:)]+]] = stablehlo.add %[[CT]], %[[M1]] : tensor<2x3x3xf64>
// BOTH-NEXT: %[[I:[^ ,:)]+]] = stablehlo.transpose %[[S1]], dims = [0, 1, 2] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[M2:[^ ,:)]+]] = stablehlo.multiply %[[V2]], %[[BT]] : tensor<2x3x3xf64>
// BOTH-NEXT: %[[S2:[^ ,:)]+]] = stablehlo.add %[[I]], %[[M2]] : tensor<2x3x3xf64>
// BOTH-NEXT: %[[T:[^ ,:)]+]] = stablehlo.transpose %[[S2]], dims = [0, 2, 1] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[R:[^ ,:)]+]] = stablehlo.add %[[T]], %[[G]] : tensor<2x3x3xf64>
// BOTH-NEXT: return %[[R]] : tensor<2x3x3xf64>
// BOTH-NEXT: }

// Three products of views of the same argument, each computed in its layout
// with the transpose on its result: the sums absorb the transposes (each
// cancels one on a product), so transposed_view_elementwise moves them to
// the result of the sum of the three, and transpose_elementwise does not
// push that one onto the three products (a copy of each, instead of one).
func.func @sum(%w: tensor<2x9x3xf64>, %s: tensor<16xf64>, %g: tensor<2x3x3xf64>) -> tensor<2x3x3xf64> {
  %v1 = stablehlo.slice %w [0:2, 0:3, 0:3] : (tensor<2x9x3xf64>) -> tensor<2x3x3xf64>
  %v2 = stablehlo.slice %w [0:2, 3:6, 0:3] : (tensor<2x9x3xf64>) -> tensor<2x3x3xf64>
  %v3 = stablehlo.slice %w [0:2, 6:9, 0:3] : (tensor<2x9x3xf64>) -> tensor<2x3x3xf64>
  %t1 = stablehlo.transpose %v1, dims = [0, 2, 1] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
  %t2 = stablehlo.transpose %v2, dims = [0, 2, 1] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
  %t3 = stablehlo.transpose %v3, dims = [0, 2, 1] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
  %s1 = stablehlo.slice %s [1:2] : (tensor<16xf64>) -> tensor<1xf64>
  %s2 = stablehlo.slice %s [2:3] : (tensor<16xf64>) -> tensor<1xf64>
  %s3 = stablehlo.slice %s [3:4] : (tensor<16xf64>) -> tensor<1xf64>
  %b1 = stablehlo.broadcast_in_dim %s1, dims = [0] : (tensor<1xf64>) -> tensor<2x3x3xf64>
  %m1 = stablehlo.multiply %t1, %b1 : tensor<2x3x3xf64>
  %b2 = stablehlo.broadcast_in_dim %s2, dims = [0] : (tensor<1xf64>) -> tensor<2x3x3xf64>
  %m2 = stablehlo.multiply %t2, %b2 : tensor<2x3x3xf64>
  %a1 = stablehlo.add %m1, %m2 : tensor<2x3x3xf64>
  %b3 = stablehlo.broadcast_in_dim %s3, dims = [0] : (tensor<1xf64>) -> tensor<2x3x3xf64>
  %m3 = stablehlo.multiply %t3, %b3 : tensor<2x3x3xf64>
  %a2 = stablehlo.add %a1, %m3 : tensor<2x3x3xf64>
  %a3 = stablehlo.add %a2, %g : tensor<2x3x3xf64>
  return %a3 : tensor<2x3x3xf64>
}

// BOTH-LABEL: func.func @sum(
// BOTH-SAME: %[[W:[^ ,:)]+]]: tensor<2x9x3xf64>, %[[S:[^ ,:)]+]]: tensor<16xf64>, %[[G:[^ ,:)]+]]: tensor<2x3x3xf64>) -> tensor<2x3x3xf64> {
// BOTH-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.slice %[[W]] [0:2, 0:3, 0:3] : (tensor<2x9x3xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.slice %[[W]] [0:2, 3:6, 0:3] : (tensor<2x9x3xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.slice %[[W]] [0:2, 6:9, 0:3] : (tensor<2x9x3xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[S1:[^ ,:)]+]] = stablehlo.slice %[[S]] [1:2] : (tensor<16xf64>) -> tensor<1xf64>
// BOTH-NEXT: %[[S2:[^ ,:)]+]] = stablehlo.slice %[[S]] [2:3] : (tensor<16xf64>) -> tensor<1xf64>
// BOTH-NEXT: %[[S3:[^ ,:)]+]] = stablehlo.slice %[[S]] [3:4] : (tensor<16xf64>) -> tensor<1xf64>
// BOTH-NEXT: %[[B1:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[S1]], dims = [0] : (tensor<1xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[BT1:[^ ,:)]+]] = stablehlo.transpose %[[B1]], dims = [0, 2, 1] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[M1:[^ ,:)]+]] = stablehlo.multiply %[[V1]], %[[BT1]] : tensor<2x3x3xf64>
// BOTH-NEXT: %[[I1:[^ ,:)]+]] = stablehlo.transpose %[[M1]], dims = [0, 1, 2] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[B2:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[S2]], dims = [0] : (tensor<1xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[BT2:[^ ,:)]+]] = stablehlo.transpose %[[B2]], dims = [0, 2, 1] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[M2:[^ ,:)]+]] = stablehlo.multiply %[[V2]], %[[BT2]] : tensor<2x3x3xf64>
// BOTH-NEXT: %[[I2:[^ ,:)]+]] = stablehlo.transpose %[[M2]], dims = [0, 1, 2] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[A1:[^ ,:)]+]] = stablehlo.add %[[I1]], %[[I2]] : tensor<2x3x3xf64>
// BOTH-NEXT: %[[B3:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[S3]], dims = [0] : (tensor<1xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[BT3:[^ ,:)]+]] = stablehlo.transpose %[[B3]], dims = [0, 2, 1] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[M3:[^ ,:)]+]] = stablehlo.multiply %[[V3]], %[[BT3]] : tensor<2x3x3xf64>
// BOTH-NEXT: %[[A2:[^ ,:)]+]] = stablehlo.add %[[A1]], %[[M3]] : tensor<2x3x3xf64>
// BOTH-NEXT: %[[T:[^ ,:)]+]] = stablehlo.transpose %[[A2]], dims = [0, 2, 1] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[R:[^ ,:)]+]] = stablehlo.add %[[T]], %[[G]] : tensor<2x3x3xf64>
// BOTH-NEXT: return %[[R]] : tensor<2x3x3xf64>
// BOTH-NEXT: }
