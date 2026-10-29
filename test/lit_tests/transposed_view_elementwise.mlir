// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=transposed_view_elementwise" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s
// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=transposed_view_elementwise;transpose_transpose" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s --check-prefix=BOTH

// A product of a contraction's result (a reduce of broadcasts, which can be
// had in any layout) and a slice of the transposed argument D, feeding the
// next contraction's broadcast: computed in D's layout, with the transpose
// on the result, where the broadcast takes it.
func.func @view(%D: tensor<8x8x4x4xf64>, %g: tensor<3x8x4x4xf64>, %B: tensor<4x3xf64>) -> tensor<4x8x4x4x3xf64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = stablehlo.broadcast_in_dim %g, dims = [0, 2, 4, 1] : (tensor<3x8x4x4xf64>) -> tensor<3x4x8x4x4xf64>
  %1 = stablehlo.broadcast_in_dim %B, dims = [3, 0] : (tensor<4x3xf64>) -> tensor<3x4x8x4x4xf64>
  %2 = stablehlo.multiply %0, %1 : tensor<3x4x8x4x4xf64>
  %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3x4x8x4x4xf64>, tensor<f64>) -> tensor<4x8x4x4xf64>
  %4 = stablehlo.transpose %D, dims = [3, 0, 1, 2] : (tensor<8x8x4x4xf64>) -> tensor<4x8x8x4xf64>
  %5 = stablehlo.slice %4 [0:4, 0:8, 4:8, 0:4] : (tensor<4x8x8x4xf64>) -> tensor<4x8x4x4xf64>
  %6 = stablehlo.multiply %3, %5 : tensor<4x8x4x4xf64>
  %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2, 3] : (tensor<4x8x4x4xf64>) -> tensor<4x8x4x4x3xf64>
  return %7 : tensor<4x8x4x4x3xf64>
}

// CHECK-LABEL: func.func @view(
// CHECK-SAME: %[[D:[^ ,:)]+]]: tensor<8x8x4x4xf64>, %[[G:[^ ,:)]+]]: tensor<3x8x4x4xf64>, %[[B:[^ ,:)]+]]: tensor<4x3xf64>) -> tensor<4x8x4x4x3xf64> {
// CHECK-NEXT: %[[Z:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[G0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[G]], dims = [0, 2, 4, 1] : (tensor<3x8x4x4xf64>) -> tensor<3x4x8x4x4xf64>
// CHECK-NEXT: %[[B0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[B]], dims = [3, 0] : (tensor<4x3xf64>) -> tensor<3x4x8x4x4xf64>
// CHECK-NEXT: %[[M:[^ ,:)]+]] = stablehlo.multiply %[[G0]], %[[B0]] : tensor<3x4x8x4x4xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.reduce(%[[M]] init: %[[Z]]) applies stablehlo.add across dimensions = [0] : (tensor<3x4x8x4x4xf64>, tensor<f64>) -> tensor<4x8x4x4xf64>
// CHECK-NEXT: %[[RT:[^ ,:)]+]] = stablehlo.transpose %[[R]], dims = [1, 2, 3, 0] : (tensor<4x8x4x4xf64>) -> tensor<8x4x4x4xf64>
// CHECK-NEXT: %[[S:[^ ,:)]+]] = stablehlo.slice %[[D]] [0:8, 4:8, 0:4, 0:4] : (tensor<8x8x4x4xf64>) -> tensor<8x4x4x4xf64>
// CHECK-NEXT: %[[P:[^ ,:)]+]] = stablehlo.multiply %[[RT]], %[[S]] : tensor<8x4x4x4xf64>
// CHECK-NEXT: %[[T:[^ ,:)]+]] = stablehlo.transpose %[[P]], dims = [3, 0, 1, 2] : (tensor<8x4x4x4xf64>) -> tensor<4x8x4x4xf64>
// CHECK-NEXT: %[[O:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[T]], dims = [0, 1, 2, 3] : (tensor<4x8x4x4xf64>) -> tensor<4x8x4x4x3xf64>
// CHECK-NEXT: return %[[O]] : tensor<4x8x4x4x3xf64>
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

// The set's transpose_elementwise: the transpose of a sum of an argument and
// a broadcast, taken by the broadcast that reads it, stays on the sum (the
// sum computes in the argument's layout), and so does the transpose of a sum
// of two arguments (pushed, it would copy both).
func.func @push(%A: tensor<4x8xf64>, %C: tensor<4x8xf64>, %b: tensor<8xf64>) -> (tensor<8x4x2xf64>, tensor<8x4x2xf64>) {
  %0 = stablehlo.broadcast_in_dim %b, dims = [1] : (tensor<8xf64>) -> tensor<4x8xf64>
  %1 = stablehlo.add %A, %0 : tensor<4x8xf64>
  %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<4x8xf64>) -> tensor<8x4xf64>
  %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<8x4xf64>) -> tensor<8x4x2xf64>
  %4 = stablehlo.add %A, %C : tensor<4x8xf64>
  %5 = stablehlo.transpose %4, dims = [1, 0] : (tensor<4x8xf64>) -> tensor<8x4xf64>
  %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<8x4xf64>) -> tensor<8x4x2xf64>
  return %3, %6 : tensor<8x4x2xf64>, tensor<8x4x2xf64>
}

// CHECK-LABEL: func.func @push(
// CHECK-SAME: %[[A:[^ ,:)]+]]: tensor<4x8xf64>, %[[C:[^ ,:)]+]]: tensor<4x8xf64>, %[[B:[^ ,:)]+]]: tensor<8xf64>) -> (tensor<8x4x2xf64>, tensor<8x4x2xf64>) {
// CHECK-NEXT: %[[B0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[B]], dims = [1] : (tensor<8xf64>) -> tensor<4x8xf64>
// CHECK-NEXT: %[[S1:[^ ,:)]+]] = stablehlo.add %[[A]], %[[B0]] : tensor<4x8xf64>
// CHECK-NEXT: %[[T1:[^ ,:)]+]] = stablehlo.transpose %[[S1]], dims = [1, 0] : (tensor<4x8xf64>) -> tensor<8x4xf64>
// CHECK-NEXT: %[[O1:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[T1]], dims = [0, 1] : (tensor<8x4xf64>) -> tensor<8x4x2xf64>
// CHECK-NEXT: %[[S2:[^ ,:)]+]] = stablehlo.add %[[A]], %[[C]] : tensor<4x8xf64>
// CHECK-NEXT: %[[T2:[^ ,:)]+]] = stablehlo.transpose %[[S2]], dims = [1, 0] : (tensor<4x8xf64>) -> tensor<8x4xf64>
// CHECK-NEXT: %[[O2:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[T2]], dims = [0, 1] : (tensor<8x4xf64>) -> tensor<8x4x2xf64>
// CHECK-NEXT: return %[[O1]], %[[O2]] : tensor<8x4x2xf64>, tensor<8x4x2xf64>
// CHECK-NEXT: }

// The sum %5 absorbs the transpose (it cancels the one on the product %2),
// so transpose_elementwise does not push the transpose of %8 onto %7, a
// product of a view, which transposed_view_elementwise would move back; the
// transpose of %5 is pushed in, where it cancels on the product %2 (an
// identity transpose is left on it: pushed onto the view %v1, it would be a
// copy as big). One transpose is left, on the sum of the two products,
// computed in the views' layout.
func.func @chain(%v1: tensor<2x3x3xf64>, %v2: tensor<2x3x3xf64>, %b: tensor<1xf64>, %c: tensor<1xf64>) -> tensor<2x3x3x4xf64> {
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
  %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<2x3x3xf64>) -> tensor<2x3x3x4xf64>
  return %10 : tensor<2x3x3x4xf64>
}

// Three products of views of the same argument, each computed in its layout
// with the transpose on its result; the sums of the three read them so
// (the set has no pattern taking a transpose off every operand of a sum),
// and transpose_elementwise leaves them there: pushed onto the products, a
// transpose would land on the views.
func.func @sum(%w: tensor<2x9x3xf64>, %s: tensor<16xf64>) -> tensor<2x3x3x4xf64> {
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
  %a3 = stablehlo.broadcast_in_dim %a2, dims = [0, 1, 2] : (tensor<2x3x3xf64>) -> tensor<2x3x3x4xf64>
  return %a3 : tensor<2x3x3x4xf64>
}

// The product's consumers do not take a transpose (a reduce reads it as it
// is), so the transpose is a copy either way: of the view's window, as big
// as the product, if left there, or of the product. Computed in D's layout,
// with the transpose on the result.
func.func @copy_on_result(%D: tensor<8x8x4x4xf64>, %g: tensor<3x8x4x4xf64>, %B: tensor<4x3xf64>, %C: tensor<4x8x4xf64>) -> tensor<8x4x4xf64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = stablehlo.broadcast_in_dim %g, dims = [0, 2, 4, 1] : (tensor<3x8x4x4xf64>) -> tensor<3x4x8x4x4xf64>
  %1 = stablehlo.broadcast_in_dim %B, dims = [3, 0] : (tensor<4x3xf64>) -> tensor<3x4x8x4x4xf64>
  %2 = stablehlo.multiply %0, %1 : tensor<3x4x8x4x4xf64>
  %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3x4x8x4x4xf64>, tensor<f64>) -> tensor<4x8x4x4xf64>
  %4 = stablehlo.transpose %D, dims = [3, 0, 1, 2] : (tensor<8x8x4x4xf64>) -> tensor<4x8x8x4xf64>
  %5 = stablehlo.slice %4 [0:4, 0:8, 4:8, 0:4] : (tensor<4x8x8x4xf64>) -> tensor<4x8x4x4xf64>
  %6 = stablehlo.multiply %3, %5 : tensor<4x8x4x4xf64>
  %7 = stablehlo.broadcast_in_dim %C, dims = [0, 1, 2] : (tensor<4x8x4xf64>) -> tensor<4x8x4x4xf64>
  %8 = stablehlo.multiply %6, %7 : tensor<4x8x4x4xf64>
  %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<4x8x4x4xf64>, tensor<f64>) -> tensor<8x4x4xf64>
  return %9 : tensor<8x4x4xf64>
}

// CHECK-LABEL: func.func @copy_on_result(
// CHECK-SAME: %[[D:[^ ,:)]+]]: tensor<8x8x4x4xf64>, %[[G:[^ ,:)]+]]: tensor<3x8x4x4xf64>, %[[B:[^ ,:)]+]]: tensor<4x3xf64>, %[[C:[^ ,:)]+]]: tensor<4x8x4xf64>) -> tensor<8x4x4xf64> {
// CHECK-NEXT: %[[Z:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[G0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[G]], dims = [0, 2, 4, 1] : (tensor<3x8x4x4xf64>) -> tensor<3x4x8x4x4xf64>
// CHECK-NEXT: %[[B0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[B]], dims = [3, 0] : (tensor<4x3xf64>) -> tensor<3x4x8x4x4xf64>
// CHECK-NEXT: %[[M:[^ ,:)]+]] = stablehlo.multiply %[[G0]], %[[B0]] : tensor<3x4x8x4x4xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.reduce(%[[M]] init: %[[Z]]) applies stablehlo.add across dimensions = [0] : (tensor<3x4x8x4x4xf64>, tensor<f64>) -> tensor<4x8x4x4xf64>
// CHECK-NEXT: %[[RT:[^ ,:)]+]] = stablehlo.transpose %[[R]], dims = [1, 2, 3, 0] : (tensor<4x8x4x4xf64>) -> tensor<8x4x4x4xf64>
// CHECK-NEXT: %[[S:[^ ,:)]+]] = stablehlo.slice %[[D]] [0:8, 4:8, 0:4, 0:4] : (tensor<8x8x4x4xf64>) -> tensor<8x4x4x4xf64>
// CHECK-NEXT: %[[P:[^ ,:)]+]] = stablehlo.multiply %[[RT]], %[[S]] : tensor<8x4x4x4xf64>
// CHECK-NEXT: %[[T:[^ ,:)]+]] = stablehlo.transpose %[[P]], dims = [3, 0, 1, 2] : (tensor<8x4x4x4xf64>) -> tensor<4x8x4x4xf64>
// CHECK-NEXT: %[[C0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[C]], dims = [0, 1, 2] : (tensor<4x8x4xf64>) -> tensor<4x8x4x4xf64>
// CHECK-NEXT: %[[P2:[^ ,:)]+]] = stablehlo.multiply %[[T]], %[[C0]] : tensor<4x8x4x4xf64>
// CHECK-NEXT: %[[R2:[^ ,:)]+]] = stablehlo.reduce(%[[P2]] init: %[[Z]]) applies stablehlo.add across dimensions = [0] : (tensor<4x8x4x4xf64>, tensor<f64>) -> tensor<8x4x4xf64>
// CHECK-NEXT: return %[[R2]] : tensor<8x4x4xf64>
// CHECK-NEXT: }

// The set's transpose_elementwise, the other way round: a transpose of a
// product of a view's window that the reduce does not take stays on the
// product (pushed onto the window, it would be a copy as big).
func.func @keep_on_result(%D: tensor<8x8x4x4xf64>, %b: tensor<8xf64>) -> tensor<8x4x4xf64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = stablehlo.slice %D [0:8, 4:8, 0:4, 0:4] : (tensor<8x8x4x4xf64>) -> tensor<8x4x4x4xf64>
  %1 = stablehlo.broadcast_in_dim %b, dims = [0] : (tensor<8xf64>) -> tensor<8x4x4x4xf64>
  %2 = stablehlo.multiply %0, %1 : tensor<8x4x4x4xf64>
  %3 = stablehlo.transpose %2, dims = [3, 0, 1, 2] : (tensor<8x4x4x4xf64>) -> tensor<4x8x4x4xf64>
  %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<4x8x4x4xf64>, tensor<f64>) -> tensor<8x4x4xf64>
  return %4 : tensor<8x4x4xf64>
}

// CHECK-LABEL: func.func @keep_on_result(
// CHECK-SAME: %[[D:[^ ,:)]+]]: tensor<8x8x4x4xf64>, %[[B:[^ ,:)]+]]: tensor<8xf64>) -> tensor<8x4x4xf64> {
// CHECK-NEXT: %[[Z:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[S:[^ ,:)]+]] = stablehlo.slice %[[D]] [0:8, 4:8, 0:4, 0:4] : (tensor<8x8x4x4xf64>) -> tensor<8x4x4x4xf64>
// CHECK-NEXT: %[[B0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[B]], dims = [0] : (tensor<8xf64>) -> tensor<8x4x4x4xf64>
// CHECK-NEXT: %[[P:[^ ,:)]+]] = stablehlo.multiply %[[S]], %[[B0]] : tensor<8x4x4x4xf64>
// CHECK-NEXT: %[[T:[^ ,:)]+]] = stablehlo.transpose %[[P]], dims = [3, 0, 1, 2] : (tensor<8x4x4x4xf64>) -> tensor<4x8x4x4xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.reduce(%[[T]] init: %[[Z]]) applies stablehlo.add across dimensions = [0] : (tensor<4x8x4x4xf64>, tensor<f64>) -> tensor<8x4x4xf64>
// CHECK-NEXT: return %[[R]] : tensor<8x4x4xf64>
// CHECK-NEXT: }

// BOTH-LABEL: func.func @chain(
// BOTH-SAME: %[[V1:[^ ,:)]+]]: tensor<2x3x3xf64>, %[[V2:[^ ,:)]+]]: tensor<2x3x3xf64>, %[[B:[^ ,:)]+]]: tensor<1xf64>, %[[C:[^ ,:)]+]]: tensor<1xf64>) -> tensor<2x3x3x4xf64> {
// BOTH-NEXT: %[[B0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[B]], dims = [0] : (tensor<1xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[BT:[^ ,:)]+]] = stablehlo.transpose %[[B0]], dims = [0, 2, 1] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[M1:[^ ,:)]+]] = stablehlo.multiply %[[V1]], %[[BT]] : tensor<2x3x3xf64>
// BOTH-NEXT: %[[I:[^ ,:)]+]] = stablehlo.transpose %[[M1]], dims = [0, 1, 2] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[C0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[C]], dims = [0] : (tensor<1xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[CT:[^ ,:)]+]] = stablehlo.transpose %[[C0]], dims = [0, 2, 1] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[S1:[^ ,:)]+]] = stablehlo.add %[[CT]], %[[I]] : tensor<2x3x3xf64>
// BOTH-NEXT: %[[M2:[^ ,:)]+]] = stablehlo.multiply %[[V2]], %[[BT]] : tensor<2x3x3xf64>
// BOTH-NEXT: %[[S2:[^ ,:)]+]] = stablehlo.add %[[S1]], %[[M2]] : tensor<2x3x3xf64>
// BOTH-NEXT: %[[T:[^ ,:)]+]] = stablehlo.transpose %[[S2]], dims = [0, 2, 1] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[O:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[T]], dims = [0, 1, 2] : (tensor<2x3x3xf64>) -> tensor<2x3x3x4xf64>
// BOTH-NEXT: return %[[O]] : tensor<2x3x3x4xf64>
// BOTH-NEXT: }

// BOTH-LABEL: func.func @sum(
// BOTH-SAME: %[[W:[^ ,:)]+]]: tensor<2x9x3xf64>, %[[S:[^ ,:)]+]]: tensor<16xf64>) -> tensor<2x3x3x4xf64> {
// BOTH-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.slice %[[W]] [0:2, 0:3, 0:3] : (tensor<2x9x3xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.slice %[[W]] [0:2, 3:6, 0:3] : (tensor<2x9x3xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.slice %[[W]] [0:2, 6:9, 0:3] : (tensor<2x9x3xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[S1:[^ ,:)]+]] = stablehlo.slice %[[S]] [1:2] : (tensor<16xf64>) -> tensor<1xf64>
// BOTH-NEXT: %[[S2:[^ ,:)]+]] = stablehlo.slice %[[S]] [2:3] : (tensor<16xf64>) -> tensor<1xf64>
// BOTH-NEXT: %[[S3:[^ ,:)]+]] = stablehlo.slice %[[S]] [3:4] : (tensor<16xf64>) -> tensor<1xf64>
// BOTH-NEXT: %[[B1:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[S1]], dims = [0] : (tensor<1xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[BT1:[^ ,:)]+]] = stablehlo.transpose %[[B1]], dims = [0, 2, 1] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[M1:[^ ,:)]+]] = stablehlo.multiply %[[V1]], %[[BT1]] : tensor<2x3x3xf64>
// BOTH-NEXT: %[[T1:[^ ,:)]+]] = stablehlo.transpose %[[M1]], dims = [0, 2, 1] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[B2:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[S2]], dims = [0] : (tensor<1xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[BT2:[^ ,:)]+]] = stablehlo.transpose %[[B2]], dims = [0, 2, 1] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[M2:[^ ,:)]+]] = stablehlo.multiply %[[V2]], %[[BT2]] : tensor<2x3x3xf64>
// BOTH-NEXT: %[[T2:[^ ,:)]+]] = stablehlo.transpose %[[M2]], dims = [0, 2, 1] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[A1:[^ ,:)]+]] = stablehlo.add %[[T1]], %[[T2]] : tensor<2x3x3xf64>
// BOTH-NEXT: %[[B3:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[S3]], dims = [0] : (tensor<1xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[BT3:[^ ,:)]+]] = stablehlo.transpose %[[B3]], dims = [0, 2, 1] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[M3:[^ ,:)]+]] = stablehlo.multiply %[[V3]], %[[BT3]] : tensor<2x3x3xf64>
// BOTH-NEXT: %[[T3:[^ ,:)]+]] = stablehlo.transpose %[[M3]], dims = [0, 2, 1] : (tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
// BOTH-NEXT: %[[A2:[^ ,:)]+]] = stablehlo.add %[[A1]], %[[T3]] : tensor<2x3x3xf64>
// BOTH-NEXT: %[[O:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[A2]], dims = [0, 1, 2] : (tensor<2x3x3xf64>) -> tensor<2x3x3x4xf64>
// BOTH-NEXT: return %[[O]] : tensor<2x3x3x4xf64>
// BOTH-NEXT: }

// A chain summing products of transposed views of memory (one view read
// through a slice of a whole-buffer transpose) and one more view: the
// transpose taken off a view moves to the chain's result, and is not pushed
// back onto the views (two copies of memory for the result's one), so the
// patterns converge.
func.func @views_through_chain(%x: tensor<4x2x3x3xf64>, %buf: tensor<4x2x64xf64>, %mem: tensor<216xf64>, %b0: tensor<4x1xf64>, %b1: tensor<4x1xf64>, %b2: tensor<4x1xf64>) -> tensor<4x2x3x3xf64> {
  %v0 = stablehlo.transpose %x, dims = [0, 1, 3, 2] : (tensor<4x2x3x3xf64>) -> tensor<4x2x3x3xf64>
  %s1 = stablehlo.slice %buf [0:4, 0:2, 0:18] : (tensor<4x2x64xf64>) -> tensor<4x2x18xf64>
  %r1 = stablehlo.reshape %s1 : (tensor<4x2x18xf64>) -> tensor<4x2x6x3xf64>
  %t1 = stablehlo.transpose %r1, dims = [0, 1, 3, 2] : (tensor<4x2x6x3xf64>) -> tensor<4x2x3x6xf64>
  %v1 = stablehlo.slice %t1 [0:4, 0:2, 0:3, 3:6] : (tensor<4x2x3x6xf64>) -> tensor<4x2x3x3xf64>
  %s2 = stablehlo.slice %buf [0:4, 0:2, 0:27] : (tensor<4x2x64xf64>) -> tensor<4x2x27xf64>
  %r2 = stablehlo.reshape %s2 : (tensor<4x2x27xf64>) -> tensor<4x2x9x3xf64>
  %t2 = stablehlo.transpose %r2, dims = [0, 1, 3, 2] : (tensor<4x2x9x3xf64>) -> tensor<4x2x3x9xf64>
  %v2 = stablehlo.slice %t2 [0:4, 0:2, 0:3, 6:9] : (tensor<4x2x3x9xf64>) -> tensor<4x2x3x3xf64>
  %rm = stablehlo.reshape %mem : (tensor<216xf64>) -> tensor<4x2x9x3xf64>
  %tm = stablehlo.transpose %rm, dims = [0, 1, 3, 2] : (tensor<4x2x9x3xf64>) -> tensor<4x2x3x9xf64>
  %v3 = stablehlo.slice %tm [0:4, 0:2, 0:3, 0:3] : (tensor<4x2x3x9xf64>) -> tensor<4x2x3x3xf64>
  %c0 = stablehlo.broadcast_in_dim %b0, dims = [0, 1] : (tensor<4x1xf64>) -> tensor<4x2x3x3xf64>
  %m0 = stablehlo.multiply %v0, %c0 : tensor<4x2x3x3xf64>
  %c1 = stablehlo.broadcast_in_dim %b1, dims = [0, 1] : (tensor<4x1xf64>) -> tensor<4x2x3x3xf64>
  %m1 = stablehlo.multiply %v1, %c1 : tensor<4x2x3x3xf64>
  %a1 = stablehlo.add %m0, %m1 : tensor<4x2x3x3xf64>
  %c2 = stablehlo.broadcast_in_dim %b2, dims = [0, 1] : (tensor<4x1xf64>) -> tensor<4x2x3x3xf64>
  %m2 = stablehlo.multiply %v2, %c2 : tensor<4x2x3x3xf64>
  %a2 = stablehlo.add %a1, %m2 : tensor<4x2x3x3xf64>
  %a3 = stablehlo.add %a2, %v3 : tensor<4x2x3x3xf64>
  return %a3 : tensor<4x2x3x3xf64>
}

// CHECK-LABEL: func.func @views_through_chain(
// CHECK-SAME: %[[A0:[^ ,:)]+]]: tensor<4x2x3x3xf64>, %[[A1:[^ ,:)]+]]: tensor<4x2x64xf64>, %[[A2:[^ ,:)]+]]: tensor<216xf64>, %[[A3:[^ ,:)]+]]: tensor<4x1xf64>, %[[A4:[^ ,:)]+]]: tensor<4x1xf64>, %[[A5:[^ ,:)]+]]: tensor<4x1xf64>) -> tensor<4x2x3x3xf64> {
// CHECK-NEXT: %[[V0:[^ ,:)]+]] = stablehlo.transpose %[[A0]], dims = [0, 1, 3, 2] : (tensor<4x2x3x3xf64>) -> tensor<4x2x3x3xf64>
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.transpose %[[V0]], dims = [0, 1, 3, 2] : (tensor<4x2x3x3xf64>) -> tensor<4x2x3x3xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.slice %[[A1]] [0:4, 0:2, 0:18] : (tensor<4x2x64xf64>) -> tensor<4x2x18xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V2]] : (tensor<4x2x18xf64>) -> tensor<4x2x6x3xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.transpose %[[V3]], dims = [0, 1, 3, 2] : (tensor<4x2x6x3xf64>) -> tensor<4x2x3x6xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.slice %[[V4]] [0:4, 0:2, 0:3, 3:6] : (tensor<4x2x3x6xf64>) -> tensor<4x2x3x3xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.transpose %[[V5]], dims = [0, 1, 3, 2] : (tensor<4x2x3x3xf64>) -> tensor<4x2x3x3xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.slice %[[A1]] [0:4, 0:2, 0:27] : (tensor<4x2x64xf64>) -> tensor<4x2x27xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.reshape %[[V7]] : (tensor<4x2x27xf64>) -> tensor<4x2x9x3xf64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.transpose %[[V8]], dims = [0, 1, 3, 2] : (tensor<4x2x9x3xf64>) -> tensor<4x2x3x9xf64>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = stablehlo.slice %[[V9]] [0:4, 0:2, 0:3, 6:9] : (tensor<4x2x3x9xf64>) -> tensor<4x2x3x3xf64>
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = stablehlo.transpose %[[V10]], dims = [0, 1, 3, 2] : (tensor<4x2x3x3xf64>) -> tensor<4x2x3x3xf64>
// CHECK-NEXT: %[[V12:[^ ,:)]+]] = stablehlo.reshape %[[A2]] : (tensor<216xf64>) -> tensor<4x2x9x3xf64>
// CHECK-NEXT: %[[V13:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[A3]], dims = [0, 1] : (tensor<4x1xf64>) -> tensor<4x2x3x3xf64>
// CHECK-NEXT: %[[V14:[^ ,:)]+]] = stablehlo.transpose %[[V13]], dims = [0, 1, 3, 2] : (tensor<4x2x3x3xf64>) -> tensor<4x2x3x3xf64>
// CHECK-NEXT: %[[V15:[^ ,:)]+]] = stablehlo.multiply %[[V1]], %[[V14]] : tensor<4x2x3x3xf64>
// CHECK-NEXT: %[[V16:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[A4]], dims = [0, 1] : (tensor<4x1xf64>) -> tensor<4x2x3x3xf64>
// CHECK-NEXT: %[[V17:[^ ,:)]+]] = stablehlo.transpose %[[V16]], dims = [0, 1, 3, 2] : (tensor<4x2x3x3xf64>) -> tensor<4x2x3x3xf64>
// CHECK-NEXT: %[[V18:[^ ,:)]+]] = stablehlo.multiply %[[V6]], %[[V17]] : tensor<4x2x3x3xf64>
// CHECK-NEXT: %[[V19:[^ ,:)]+]] = stablehlo.add %[[V15]], %[[V18]] : tensor<4x2x3x3xf64>
// CHECK-NEXT: %[[V20:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[A5]], dims = [0, 1] : (tensor<4x1xf64>) -> tensor<4x2x3x3xf64>
// CHECK-NEXT: %[[V21:[^ ,:)]+]] = stablehlo.transpose %[[V20]], dims = [0, 1, 3, 2] : (tensor<4x2x3x3xf64>) -> tensor<4x2x3x3xf64>
// CHECK-NEXT: %[[V22:[^ ,:)]+]] = stablehlo.multiply %[[V11]], %[[V21]] : tensor<4x2x3x3xf64>
// CHECK-NEXT: %[[V23:[^ ,:)]+]] = stablehlo.add %[[V19]], %[[V22]] : tensor<4x2x3x3xf64>
// CHECK-NEXT: %[[V24:[^ ,:)]+]] = stablehlo.slice %[[V12]] [0:4, 0:2, 0:3, 0:3] : (tensor<4x2x9x3xf64>) -> tensor<4x2x3x3xf64>
// CHECK-NEXT: %[[V25:[^ ,:)]+]] = stablehlo.add %[[V23]], %[[V24]] : tensor<4x2x3x3xf64>
// CHECK-NEXT: %[[V26:[^ ,:)]+]] = stablehlo.transpose %[[V25]], dims = [0, 1, 3, 2] : (tensor<4x2x3x3xf64>) -> tensor<4x2x3x3xf64>
// CHECK-NEXT: return %[[V26]] : tensor<4x2x3x3xf64>
// CHECK-NEXT: }

// BOTH-LABEL: func.func @views_through_chain(
// BOTH-SAME: %[[A0:[^ ,:)]+]]: tensor<4x2x3x3xf64>, %[[A1:[^ ,:)]+]]: tensor<4x2x64xf64>, %[[A2:[^ ,:)]+]]: tensor<216xf64>, %[[A3:[^ ,:)]+]]: tensor<4x1xf64>, %[[A4:[^ ,:)]+]]: tensor<4x1xf64>, %[[A5:[^ ,:)]+]]: tensor<4x1xf64>) -> tensor<4x2x3x3xf64> {
// BOTH-NEXT: %[[V0:[^ ,:)]+]] = stablehlo.slice %[[A1]] [0:4, 0:2, 0:18] : (tensor<4x2x64xf64>) -> tensor<4x2x18xf64>
// BOTH-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<4x2x18xf64>) -> tensor<4x2x6x3xf64>
// BOTH-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.transpose %[[V1]], dims = [0, 1, 3, 2] : (tensor<4x2x6x3xf64>) -> tensor<4x2x3x6xf64>
// BOTH-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.slice %[[V2]] [0:4, 0:2, 0:3, 3:6] : (tensor<4x2x3x6xf64>) -> tensor<4x2x3x3xf64>
// BOTH-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.transpose %[[V3]], dims = [0, 1, 3, 2] : (tensor<4x2x3x3xf64>) -> tensor<4x2x3x3xf64>
// BOTH-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.slice %[[A1]] [0:4, 0:2, 0:27] : (tensor<4x2x64xf64>) -> tensor<4x2x27xf64>
// BOTH-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.reshape %[[V5]] : (tensor<4x2x27xf64>) -> tensor<4x2x9x3xf64>
// BOTH-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.transpose %[[V6]], dims = [0, 1, 3, 2] : (tensor<4x2x9x3xf64>) -> tensor<4x2x3x9xf64>
// BOTH-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.slice %[[V7]] [0:4, 0:2, 0:3, 6:9] : (tensor<4x2x3x9xf64>) -> tensor<4x2x3x3xf64>
// BOTH-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.transpose %[[V8]], dims = [0, 1, 3, 2] : (tensor<4x2x3x3xf64>) -> tensor<4x2x3x3xf64>
// BOTH-NEXT: %[[V10:[^ ,:)]+]] = stablehlo.reshape %[[A2]] : (tensor<216xf64>) -> tensor<4x2x9x3xf64>
// BOTH-NEXT: %[[V11:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[A3]], dims = [0, 1] : (tensor<4x1xf64>) -> tensor<4x2x3x3xf64>
// BOTH-NEXT: %[[V12:[^ ,:)]+]] = stablehlo.transpose %[[V11]], dims = [0, 1, 3, 2] : (tensor<4x2x3x3xf64>) -> tensor<4x2x3x3xf64>
// BOTH-NEXT: %[[V13:[^ ,:)]+]] = stablehlo.multiply %[[A0]], %[[V12]] : tensor<4x2x3x3xf64>
// BOTH-NEXT: %[[V14:[^ ,:)]+]] = stablehlo.transpose %[[V13]], dims = [0, 1, 2, 3] : (tensor<4x2x3x3xf64>) -> tensor<4x2x3x3xf64>
// BOTH-NEXT: %[[V15:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[A4]], dims = [0, 1] : (tensor<4x1xf64>) -> tensor<4x2x3x3xf64>
// BOTH-NEXT: %[[V16:[^ ,:)]+]] = stablehlo.transpose %[[V15]], dims = [0, 1, 3, 2] : (tensor<4x2x3x3xf64>) -> tensor<4x2x3x3xf64>
// BOTH-NEXT: %[[V17:[^ ,:)]+]] = stablehlo.multiply %[[V4]], %[[V16]] : tensor<4x2x3x3xf64>
// BOTH-NEXT: %[[V18:[^ ,:)]+]] = stablehlo.add %[[V14]], %[[V17]] : tensor<4x2x3x3xf64>
// BOTH-NEXT: %[[V19:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[A5]], dims = [0, 1] : (tensor<4x1xf64>) -> tensor<4x2x3x3xf64>
// BOTH-NEXT: %[[V20:[^ ,:)]+]] = stablehlo.transpose %[[V19]], dims = [0, 1, 3, 2] : (tensor<4x2x3x3xf64>) -> tensor<4x2x3x3xf64>
// BOTH-NEXT: %[[V21:[^ ,:)]+]] = stablehlo.multiply %[[V9]], %[[V20]] : tensor<4x2x3x3xf64>
// BOTH-NEXT: %[[V22:[^ ,:)]+]] = stablehlo.add %[[V18]], %[[V21]] : tensor<4x2x3x3xf64>
// BOTH-NEXT: %[[V23:[^ ,:)]+]] = stablehlo.slice %[[V10]] [0:4, 0:2, 0:3, 0:3] : (tensor<4x2x9x3xf64>) -> tensor<4x2x3x3xf64>
// BOTH-NEXT: %[[V24:[^ ,:)]+]] = stablehlo.add %[[V22]], %[[V23]] : tensor<4x2x3x3xf64>
// BOTH-NEXT: %[[V25:[^ ,:)]+]] = stablehlo.transpose %[[V24]], dims = [0, 1, 3, 2] : (tensor<4x2x3x3xf64>) -> tensor<4x2x3x3xf64>
// BOTH-NEXT: return %[[V25]] : tensor<4x2x3x3xf64>
// BOTH-NEXT: }
