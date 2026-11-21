// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=reread_barrier" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// Three stages of a contraction, each reading the one before through an
// expanding broadcast: the first two are materialized (nested in their
// readers, they would be recomputed once per element of the expansion);
// the third, read by the return, stays.
func.func @stages(%x: tensor<4096x4x4xf64>, %B: tensor<3x4xf64>) -> tensor<4096x4x4xf64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = stablehlo.broadcast_in_dim %x, dims = [1, 2, 3] : (tensor<4096x4x4xf64>) -> tensor<3x4096x4x4xf64>
  %1 = stablehlo.broadcast_in_dim %B, dims = [0, 3] : (tensor<3x4xf64>) -> tensor<3x4096x4x4xf64>
  %2 = stablehlo.multiply %0, %1 : tensor<3x4096x4x4xf64>
  %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3x4096x4x4xf64>, tensor<f64>) -> tensor<4096x4x4xf64>
  %4 = stablehlo.broadcast_in_dim %3, dims = [1, 2, 3] : (tensor<4096x4x4xf64>) -> tensor<3x4096x4x4xf64>
  %5 = stablehlo.multiply %4, %1 : tensor<3x4096x4x4xf64>
  %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3x4096x4x4xf64>, tensor<f64>) -> tensor<4096x4x4xf64>
  %7 = stablehlo.broadcast_in_dim %6, dims = [1, 2, 3] : (tensor<4096x4x4xf64>) -> tensor<3x4096x4x4xf64>
  %8 = stablehlo.multiply %7, %1 : tensor<3x4096x4x4xf64>
  %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3x4096x4x4xf64>, tensor<f64>) -> tensor<4096x4x4xf64>
  return %9 : tensor<4096x4x4xf64>
}

// CHECK-LABEL: func.func @stages(
// CHECK-SAME: %[[X:[^ ,:)]+]]: tensor<4096x4x4xf64>, %[[B:[^ ,:)]+]]: tensor<3x4xf64>) -> tensor<4096x4x4xf64> {
// CHECK-NEXT: %[[CST:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[BX:.+]] = stablehlo.broadcast_in_dim %[[X]], dims = [1, 2, 3] : (tensor<4096x4x4xf64>) -> tensor<3x4096x4x4xf64>
// CHECK-NEXT: %[[BB:.+]] = stablehlo.broadcast_in_dim %[[B]], dims = [0, 3] : (tensor<3x4xf64>) -> tensor<3x4096x4x4xf64>
// CHECK-NEXT: %[[M1:.+]] = stablehlo.multiply %[[BX]], %[[BB]] : tensor<3x4096x4x4xf64>
// CHECK-NEXT: %[[R1:.+]] = stablehlo.reduce(%[[M1]] init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<3x4096x4x4xf64>, tensor<f64>) -> tensor<4096x4x4xf64>
// CHECK-NEXT: %[[BAR1:.+]] = stablehlo.optimization_barrier %[[R1]] : tensor<4096x4x4xf64>
// CHECK-NEXT: %[[BR1:.+]] = stablehlo.broadcast_in_dim %[[BAR1]], dims = [1, 2, 3] : (tensor<4096x4x4xf64>) -> tensor<3x4096x4x4xf64>
// CHECK-NEXT: %[[M2:.+]] = stablehlo.multiply %[[BR1]], %[[BB]] : tensor<3x4096x4x4xf64>
// CHECK-NEXT: %[[R2:.+]] = stablehlo.reduce(%[[M2]] init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<3x4096x4x4xf64>, tensor<f64>) -> tensor<4096x4x4xf64>
// CHECK-NEXT: %[[BAR2:.+]] = stablehlo.optimization_barrier %[[R2]] : tensor<4096x4x4xf64>
// CHECK-NEXT: %[[BR2:.+]] = stablehlo.broadcast_in_dim %[[BAR2]], dims = [1, 2, 3] : (tensor<4096x4x4xf64>) -> tensor<3x4096x4x4xf64>
// CHECK-NEXT: %[[M3:.+]] = stablehlo.multiply %[[BR2]], %[[BB]] : tensor<3x4096x4x4xf64>
// CHECK-NEXT: %[[R3:.+]] = stablehlo.reduce(%[[M3]] init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<3x4096x4x4xf64>, tensor<f64>) -> tensor<4096x4x4xf64>
// CHECK-NEXT: return %[[R3]] : tensor<4096x4x4xf64>
// CHECK-NEXT: }

// The sums of a contraction's transposed stages, read through a strided
// slice and its broadcast: computed from a reduce, they are materialized;
// their single-use terms are not. A sum of products of arguments is not a
// stage of a contraction (XLA does not recompute it) and stays.
func.func @sliced(%x: tensor<4096x3x8xf64>, %d: tensor<4096x1x8xf64>, %B: tensor<1x3xf64>) -> (tensor<4096x4x3x3xf64>, tensor<4096x4x3xf64>) {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = stablehlo.reduce(%x init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4096x3x8xf64>, tensor<f64>) -> tensor<4096x8xf64>
  %1 = stablehlo.reshape %0 : (tensor<4096x8xf64>) -> tensor<4096x1x8xf64>
  %2 = stablehlo.multiply %1, %d : tensor<4096x1x8xf64>
  %3 = stablehlo.add %2, %d : tensor<4096x1x8xf64>
  %4 = stablehlo.slice %3 [0:4096, 0:1, 0:8:2] : (tensor<4096x1x8xf64>) -> tensor<4096x1x4xf64>
  %5 = stablehlo.reshape %4 : (tensor<4096x1x4xf64>) -> tensor<4096x4xf64>
  %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<4096x4xf64>) -> tensor<4096x4x3xf64>
  %7 = stablehlo.broadcast_in_dim %B, dims = [1, 2] : (tensor<1x3xf64>) -> tensor<4096x4x3xf64>
  %8 = stablehlo.multiply %6, %7 : tensor<4096x4x3xf64>
  %9 = stablehlo.add %8, %6 : tensor<4096x4x3xf64>
  %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<4096x4x3xf64>) -> tensor<4096x4x3x3xf64>
  %11 = stablehlo.broadcast_in_dim %B, dims = [1, 3] : (tensor<1x3xf64>) -> tensor<4096x4x3x3xf64>
  %12 = stablehlo.multiply %10, %11 : tensor<4096x4x3x3xf64>
  %13 = stablehlo.multiply %d, %d : tensor<4096x1x8xf64>
  %14 = stablehlo.add %13, %d : tensor<4096x1x8xf64>
  %15 = stablehlo.slice %14 [0:4096, 0:1, 0:8:2] : (tensor<4096x1x8xf64>) -> tensor<4096x1x4xf64>
  %16 = stablehlo.reshape %15 : (tensor<4096x1x4xf64>) -> tensor<4096x4xf64>
  %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1] : (tensor<4096x4xf64>) -> tensor<4096x4x3xf64>
  return %12, %17 : tensor<4096x4x3x3xf64>, tensor<4096x4x3xf64>
}

// CHECK-LABEL: func.func @sliced(
// CHECK-SAME: %[[X:[^ ,:)]+]]: tensor<4096x3x8xf64>, %[[D:[^ ,:)]+]]: tensor<4096x1x8xf64>, %[[BM:[^ ,:)]+]]: tensor<1x3xf64>) -> (tensor<4096x4x3x3xf64>, tensor<4096x4x3xf64>) {
// CHECK-NEXT: %[[CST:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[R:.+]] = stablehlo.reduce(%[[X]] init: %[[CST]]) applies stablehlo.add across dimensions = [1] : (tensor<4096x3x8xf64>, tensor<f64>) -> tensor<4096x8xf64>
// CHECK-NEXT: %[[RR:.+]] = stablehlo.reshape %[[R]] : (tensor<4096x8xf64>) -> tensor<4096x1x8xf64>
// CHECK-NEXT: %[[M0:.+]] = stablehlo.multiply %[[RR]], %[[D]] : tensor<4096x1x8xf64>
// CHECK-NEXT: %[[S:.+]] = stablehlo.add %[[M0]], %[[D]] : tensor<4096x1x8xf64>
// CHECK-NEXT: %[[BAR:.+]] = stablehlo.optimization_barrier %[[S]] : tensor<4096x1x8xf64>
// CHECK-NEXT: %[[SL:.+]] = stablehlo.slice %[[BAR]] [0:4096, 0:1, 0:8:2] : (tensor<4096x1x8xf64>) -> tensor<4096x1x4xf64>
// CHECK-NEXT: %[[RS:.+]] = stablehlo.reshape %[[SL]] : (tensor<4096x1x4xf64>) -> tensor<4096x4xf64>
// CHECK-NEXT: %[[BR:.+]] = stablehlo.broadcast_in_dim %[[RS]], dims = [0, 1] : (tensor<4096x4xf64>) -> tensor<4096x4x3xf64>
// CHECK-NEXT: %[[BB:.+]] = stablehlo.broadcast_in_dim %[[BM]], dims = [1, 2] : (tensor<1x3xf64>) -> tensor<4096x4x3xf64>
// CHECK-NEXT: %[[M2:.+]] = stablehlo.multiply %[[BR]], %[[BB]] : tensor<4096x4x3xf64>
// CHECK-NEXT: %[[S2:.+]] = stablehlo.add %[[M2]], %[[BR]] : tensor<4096x4x3xf64>
// CHECK-NEXT: %[[BAR2:.+]] = stablehlo.optimization_barrier %[[S2]] : tensor<4096x4x3xf64>
// CHECK-NEXT: %[[BR2:.+]] = stablehlo.broadcast_in_dim %[[BAR2]], dims = [0, 1, 2] : (tensor<4096x4x3xf64>) -> tensor<4096x4x3x3xf64>
// CHECK-NEXT: %[[BB2:.+]] = stablehlo.broadcast_in_dim %[[BM]], dims = [1, 3] : (tensor<1x3xf64>) -> tensor<4096x4x3x3xf64>
// CHECK-NEXT: %[[M3:.+]] = stablehlo.multiply %[[BR2]], %[[BB2]] : tensor<4096x4x3x3xf64>
// CHECK-NEXT: %[[P:.+]] = stablehlo.multiply %[[D]], %[[D]] : tensor<4096x1x8xf64>
// CHECK-NEXT: %[[PS:.+]] = stablehlo.add %[[P]], %[[D]] : tensor<4096x1x8xf64>
// CHECK-NEXT: %[[PL:.+]] = stablehlo.slice %[[PS]] [0:4096, 0:1, 0:8:2] : (tensor<4096x1x8xf64>) -> tensor<4096x1x4xf64>
// CHECK-NEXT: %[[PR:.+]] = stablehlo.reshape %[[PL]] : (tensor<4096x1x4xf64>) -> tensor<4096x4xf64>
// CHECK-NEXT: %[[PB:.+]] = stablehlo.broadcast_in_dim %[[PR]], dims = [0, 1] : (tensor<4096x4xf64>) -> tensor<4096x4x3xf64>
// CHECK-NEXT: return %[[M3]], %[[PB]] : tensor<4096x4x3x3xf64>, tensor<4096x4x3xf64>
// CHECK-NEXT: }

// A stage read 4-fold is a reduction XLA keeps as a kernel of its own; one
// op on an argument is not worth a round trip through memory; a small value
// is read from cache; a value read without expansion is not recomputed.
func.func @once(%x: tensor<4096x4x3xf64>, %y: tensor<4096x4xf64>, %s: tensor<4x4xf64>) -> (tensor<4x4096x4xf64>, tensor<4x4x4x4xf64>, tensor<4096x4xf64>, tensor<4x4096x4xf64>) {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = stablehlo.reduce(%x init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<4096x4x3xf64>, tensor<f64>) -> tensor<4096x4xf64>
  %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<4096x4xf64>) -> tensor<4x4096x4xf64>
  %2 = stablehlo.negate %1 : tensor<4x4096x4xf64>
  %3 = stablehlo.multiply %s, %s : tensor<4x4xf64>
  %4 = stablehlo.add %3, %s : tensor<4x4xf64>
  %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<4x4xf64>) -> tensor<4x4x4xf64>
  %6 = stablehlo.multiply %5, %5 : tensor<4x4x4xf64>
  %7 = stablehlo.broadcast_in_dim %6, dims = [1, 2, 3] : (tensor<4x4x4xf64>) -> tensor<4x4x4x4xf64>
  %8 = stablehlo.add %0, %y : tensor<4096x4xf64>
  %9 = stablehlo.reshape %8 : (tensor<4096x4xf64>) -> tensor<4096x4x1xf64>
  %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<4096x4x1xf64>) -> tensor<4096x4x1xf64>
  %11 = stablehlo.reshape %10 : (tensor<4096x4x1xf64>) -> tensor<4096x4xf64>
  %12 = stablehlo.broadcast_in_dim %8, dims = [1, 2] : (tensor<4096x4xf64>) -> tensor<4x4096x4xf64>
  return %2, %7, %11, %12 : tensor<4x4096x4xf64>, tensor<4x4x4x4xf64>, tensor<4096x4xf64>, tensor<4x4096x4xf64>
}

// CHECK-LABEL: func.func @once(
// CHECK-NOT: optimization_barrier
