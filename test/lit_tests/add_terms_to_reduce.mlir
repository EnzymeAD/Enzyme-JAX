// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=add_terms_to_reduce" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// One stage of a sum-factorized kernel, unrolled: out[e, q] = sum_d x[e, d] * b[d, q],
// with x[e, d] read as every 3rd entry of a row (the terms interleave).
func.func @interleaved(%x: tensor<8x9xf64>, %b: tensor<9xf64>) -> tensor<8x3xf64> {
  %x0 = stablehlo.slice %x [0:8, 0:9:3] : (tensor<8x9xf64>) -> tensor<8x3xf64>
  %x1 = stablehlo.slice %x [0:8, 1:9:3] : (tensor<8x9xf64>) -> tensor<8x3xf64>
  %x2 = stablehlo.slice %x [0:8, 2:9:3] : (tensor<8x9xf64>) -> tensor<8x3xf64>
  %b0 = stablehlo.slice %b [0:3] : (tensor<9xf64>) -> tensor<3xf64>
  %b1 = stablehlo.slice %b [3:6] : (tensor<9xf64>) -> tensor<3xf64>
  %b2 = stablehlo.slice %b [6:9] : (tensor<9xf64>) -> tensor<3xf64>
  %c0 = stablehlo.broadcast_in_dim %b0, dims = [1] : (tensor<3xf64>) -> tensor<8x3xf64>
  %c1 = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<3xf64>) -> tensor<8x3xf64>
  %c2 = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<3xf64>) -> tensor<8x3xf64>
  %t0 = stablehlo.multiply %x0, %c0 : tensor<8x3xf64>
  %t1 = stablehlo.multiply %x1, %c1 : tensor<8x3xf64>
  %t2 = stablehlo.multiply %x2, %c2 : tensor<8x3xf64>
  %s0 = stablehlo.add %t0, %t1 : tensor<8x3xf64>
  %s1 = stablehlo.add %s0, %t2 : tensor<8x3xf64>
  return %s1 : tensor<8x3xf64>
}

// CHECK-LABEL: func.func @interleaved(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x9xf64>, %[[V1:[^ ,:)]+]]: tensor<9xf64>) -> tensor<8x3xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<8x9xf64>) -> tensor<8x3x3xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.transpose %[[V3]], dims = [2, 0, 1] : (tensor<8x3x3xf64>) -> tensor<3x8x3xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<9xf64>) -> tensor<3x3xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V5]], dims = [0, 2] : (tensor<3x3xf64>) -> tensor<3x8x3xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.multiply %[[V4]], %[[V6]] : tensor<3x8x3xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.reduce(%[[V7]] init: %[[V2]]) applies stablehlo.add across dimensions = [0] : (tensor<3x8x3xf64>, tensor<f64>) -> tensor<8x3xf64>
// CHECK-NEXT: return %[[V8]] : tensor<8x3xf64>
// CHECK-NEXT: }

// Terms that read consecutive runs, plus a term built differently that stays outside the reduce.
func.func @runs_and_extra(%x: tensor<12xf64>, %w: tensor<4xf64>, %y: tensor<4xf64>) -> tensor<4xf64> {
  %x0 = stablehlo.slice %x [0:4] : (tensor<12xf64>) -> tensor<4xf64>
  %x1 = stablehlo.slice %x [4:8] : (tensor<12xf64>) -> tensor<4xf64>
  %x2 = stablehlo.slice %x [8:12] : (tensor<12xf64>) -> tensor<4xf64>
  %t0 = stablehlo.multiply %x0, %w : tensor<4xf64>
  %t1 = stablehlo.multiply %x1, %w : tensor<4xf64>
  %t2 = stablehlo.multiply %x2, %w : tensor<4xf64>
  %s0 = stablehlo.add %t0, %y : tensor<4xf64>
  %s1 = stablehlo.add %s0, %t1 : tensor<4xf64>
  %s2 = stablehlo.add %s1, %t2 : tensor<4xf64>
  return %s2 : tensor<4xf64>
}

// CHECK-LABEL: func.func @runs_and_extra(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<12xf64>, %[[V1:[^ ,:)]+]]: tensor<4xf64>, %[[V2:[^ ,:)]+]]: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<12xf64>) -> tensor<3x4xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V1]], dims = [1] : (tensor<4xf64>) -> tensor<3x4xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.multiply %[[V4]], %[[V5]] : tensor<3x4xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.reduce(%[[V6]] init: %[[V3]]) applies stablehlo.add across dimensions = [0] : (tensor<3x4xf64>, tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.add %[[V7]], %[[V2]] : tensor<4xf64>
// CHECK-NEXT: return %[[V8]] : tensor<4xf64>
// CHECK-NEXT: }

// Terms over unrelated values are left alone: stacking them would be a copy.
func.func @unrelated(%a: tensor<4xf64>, %b: tensor<4xf64>, %c: tensor<4xf64>, %w: tensor<4xf64>) -> tensor<4xf64> {
  %t0 = stablehlo.multiply %a, %w : tensor<4xf64>
  %t1 = stablehlo.multiply %b, %w : tensor<4xf64>
  %t2 = stablehlo.multiply %c, %w : tensor<4xf64>
  %s0 = stablehlo.add %t0, %t1 : tensor<4xf64>
  %s1 = stablehlo.add %s0, %t2 : tensor<4xf64>
  return %s1 : tensor<4xf64>
}

// CHECK-LABEL: func.func @unrelated(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4xf64>, %[[V1:[^ ,:)]+]]: tensor<4xf64>, %[[V2:[^ ,:)]+]]: tensor<4xf64>, %[[V3:[^ ,:)]+]]: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.multiply %[[V0]], %[[V3]] : tensor<4xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.multiply %[[V1]], %[[V3]] : tensor<4xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.multiply %[[V2]], %[[V3]] : tensor<4xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.add %[[V4]], %[[V5]] : tensor<4xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.add %[[V7]], %[[V6]] : tensor<4xf64>
// CHECK-NEXT: return %[[V8]] : tensor<4xf64>
// CHECK-NEXT: }

// Two terms are left as an add.
func.func @two_terms(%a: tensor<4xf64>, %b: tensor<4xf64>, %w: tensor<4xf64>) -> tensor<4xf64> {
  %t0 = stablehlo.multiply %a, %w : tensor<4xf64>
  %t1 = stablehlo.multiply %b, %w : tensor<4xf64>
  %s0 = stablehlo.add %t0, %t1 : tensor<4xf64>
  return %s0 : tensor<4xf64>
}

// CHECK-LABEL: func.func @two_terms(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4xf64>, %[[V1:[^ ,:)]+]]: tensor<4xf64>, %[[V2:[^ ,:)]+]]: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.multiply %[[V0]], %[[V2]] : tensor<4xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.multiply %[[V1]], %[[V2]] : tensor<4xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.add %[[V3]], %[[V4]] : tensor<4xf64>
// CHECK-NEXT: return %[[V5]] : tensor<4xf64>
// CHECK-NEXT: }

// A sum of slices a constant step apart is a reduce of a reshape of their tensor.
func.func @plain_slices(%x: tensor<12xf64>) -> tensor<4xf64> {
  %x0 = stablehlo.slice %x [0:4] : (tensor<12xf64>) -> tensor<4xf64>
  %x1 = stablehlo.slice %x [4:8] : (tensor<12xf64>) -> tensor<4xf64>
  %x2 = stablehlo.slice %x [8:12] : (tensor<12xf64>) -> tensor<4xf64>
  %s0 = stablehlo.add %x0, %x1 : tensor<4xf64>
  %s1 = stablehlo.add %s0, %x2 : tensor<4xf64>
  return %s1 : tensor<4xf64>
}

// CHECK-LABEL: func.func @plain_slices(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<12xf64>) -> tensor<4xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<12xf64>) -> tensor<3x4xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reduce(%[[V2]] init: %[[V1]]) applies stablehlo.add across dimensions = [0] : (tensor<3x4xf64>, tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT: return %[[V3]] : tensor<4xf64>
// CHECK-NEXT: }

// Interleaved slices: a reshape, a transpose to put the terms first, then a reduce.
func.func @interleaved_slices(%x: tensor<12xf64>) -> tensor<4xf64> {
  %x0 = stablehlo.slice %x [0:12:3] : (tensor<12xf64>) -> tensor<4xf64>
  %x1 = stablehlo.slice %x [1:12:3] : (tensor<12xf64>) -> tensor<4xf64>
  %x2 = stablehlo.slice %x [2:12:3] : (tensor<12xf64>) -> tensor<4xf64>
  %s0 = stablehlo.add %x0, %x1 : tensor<4xf64>
  %s1 = stablehlo.add %s0, %x2 : tensor<4xf64>
  return %s1 : tensor<4xf64>
}

// CHECK-LABEL: func.func @interleaved_slices(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<12xf64>) -> tensor<4xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<12xf64>) -> tensor<4x3xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.transpose %[[V2]], dims = [1, 0] : (tensor<4x3xf64>) -> tensor<3x4xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reduce(%[[V3]] init: %[[V1]]) applies stablehlo.add across dimensions = [0] : (tensor<3x4xf64>, tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT: return %[[V4]] : tensor<4xf64>
// CHECK-NEXT: }

// Slices that overlap are not one view of their tensor.
func.func @overlapping_slices(%x: tensor<12xf64>) -> tensor<4xf64> {
  %x0 = stablehlo.slice %x [0:4] : (tensor<12xf64>) -> tensor<4xf64>
  %x1 = stablehlo.slice %x [2:6] : (tensor<12xf64>) -> tensor<4xf64>
  %x2 = stablehlo.slice %x [4:8] : (tensor<12xf64>) -> tensor<4xf64>
  %s0 = stablehlo.add %x0, %x1 : tensor<4xf64>
  %s1 = stablehlo.add %s0, %x2 : tensor<4xf64>
  return %s1 : tensor<4xf64>
}

// CHECK-LABEL: func.func @overlapping_slices(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<12xf64>) -> tensor<4xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.slice %[[V0]] [0:4] : (tensor<12xf64>) -> tensor<4xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.slice %[[V0]] [2:6] : (tensor<12xf64>) -> tensor<4xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.slice %[[V0]] [4:8] : (tensor<12xf64>) -> tensor<4xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.add %[[V1]], %[[V2]] : tensor<4xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.add %[[V4]], %[[V3]] : tensor<4xf64>
// CHECK-NEXT: return %[[V5]] : tensor<4xf64>
// CHECK-NEXT: }

// A term used outside the sum keeps its op, so it is not re-rolled.
func.func @shared_term(%a: tensor<4xf64>, %b: tensor<4xf64>, %c: tensor<4xf64>, %w: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
  %t0 = stablehlo.multiply %a, %w : tensor<4xf64>
  %t1 = stablehlo.multiply %b, %w : tensor<4xf64>
  %t2 = stablehlo.multiply %c, %w : tensor<4xf64>
  %s0 = stablehlo.add %t0, %t1 : tensor<4xf64>
  %s1 = stablehlo.add %s0, %t2 : tensor<4xf64>
  return %s1, %t0 : tensor<4xf64>, tensor<4xf64>
}

// CHECK-LABEL: func.func @shared_term(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4xf64>, %[[V1:[^ ,:)]+]]: tensor<4xf64>, %[[V2:[^ ,:)]+]]: tensor<4xf64>, %[[V3:[^ ,:)]+]]: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.multiply %[[V0]], %[[V3]] : tensor<4xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.multiply %[[V1]], %[[V3]] : tensor<4xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.multiply %[[V2]], %[[V3]] : tensor<4xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.add %[[V4]], %[[V5]] : tensor<4xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.add %[[V7]], %[[V6]] : tensor<4xf64>
// CHECK-NEXT: return %[[V8]], %[[V4]] : tensor<4xf64>, tensor<4xf64>
// CHECK-NEXT: }

// A select with a scalar predicate is left alone: stacking does not broadcast scalar operands.
func.func @scalar_select(%p: tensor<i1>, %a: tensor<4xf64>, %b: tensor<4xf64>, %c: tensor<4xf64>, %w: tensor<4xf64>) -> tensor<4xf64> {
  %t0 = stablehlo.select %p, %a, %w : tensor<i1>, tensor<4xf64>
  %t1 = stablehlo.select %p, %b, %w : tensor<i1>, tensor<4xf64>
  %t2 = stablehlo.select %p, %c, %w : tensor<i1>, tensor<4xf64>
  %s0 = stablehlo.add %t0, %t1 : tensor<4xf64>
  %s1 = stablehlo.add %s0, %t2 : tensor<4xf64>
  return %s1 : tensor<4xf64>
}

// CHECK-LABEL: func.func @scalar_select(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<i1>, %[[V1:[^ ,:)]+]]: tensor<4xf64>, %[[V2:[^ ,:)]+]]: tensor<4xf64>, %[[V3:[^ ,:)]+]]: tensor<4xf64>, %[[V4:[^ ,:)]+]]: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.select %[[V0]], %[[V1]], %[[V4]] : tensor<i1>, tensor<4xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.select %[[V0]], %[[V2]], %[[V4]] : tensor<i1>, tensor<4xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.select %[[V0]], %[[V3]], %[[V4]] : tensor<i1>, tensor<4xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.add %[[V5]], %[[V6]] : tensor<4xf64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.add %[[V8]], %[[V7]] : tensor<4xf64>
// CHECK-NEXT: return %[[V9]] : tensor<4xf64>
// CHECK-NEXT: }


// Two contractions read the same views of %x (row i of %b against the columns
// of %x): the shared reshapes are followed, and both sums roll up.
func.func @shared_views(%x: tensor<8x4xf64>, %b: tensor<2x4xf64>) -> (tensor<8xf64>, tensor<8xf64>) {
  %s0 = stablehlo.slice %x [0:8, 0:1] : (tensor<8x4xf64>) -> tensor<8x1xf64>
  %s1 = stablehlo.slice %x [0:8, 1:2] : (tensor<8x4xf64>) -> tensor<8x1xf64>
  %s2 = stablehlo.slice %x [0:8, 2:3] : (tensor<8x4xf64>) -> tensor<8x1xf64>
  %r0 = stablehlo.reshape %s0 : (tensor<8x1xf64>) -> tensor<8xf64>
  %r1 = stablehlo.reshape %s1 : (tensor<8x1xf64>) -> tensor<8xf64>
  %r2 = stablehlo.reshape %s2 : (tensor<8x1xf64>) -> tensor<8xf64>
  %b00 = stablehlo.slice %b [0:1, 0:1] : (tensor<2x4xf64>) -> tensor<1x1xf64>
  %b01 = stablehlo.slice %b [0:1, 1:2] : (tensor<2x4xf64>) -> tensor<1x1xf64>
  %b02 = stablehlo.slice %b [0:1, 2:3] : (tensor<2x4xf64>) -> tensor<1x1xf64>
  %b10 = stablehlo.slice %b [1:2, 0:1] : (tensor<2x4xf64>) -> tensor<1x1xf64>
  %b11 = stablehlo.slice %b [1:2, 1:2] : (tensor<2x4xf64>) -> tensor<1x1xf64>
  %b12 = stablehlo.slice %b [1:2, 2:3] : (tensor<2x4xf64>) -> tensor<1x1xf64>
  %w00 = stablehlo.broadcast_in_dim %b00, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<8x1xf64>
  %w01 = stablehlo.broadcast_in_dim %b01, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<8x1xf64>
  %w02 = stablehlo.broadcast_in_dim %b02, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<8x1xf64>
  %w10 = stablehlo.broadcast_in_dim %b10, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<8x1xf64>
  %w11 = stablehlo.broadcast_in_dim %b11, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<8x1xf64>
  %w12 = stablehlo.broadcast_in_dim %b12, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<8x1xf64>
  %v00 = stablehlo.reshape %w00 : (tensor<8x1xf64>) -> tensor<8xf64>
  %v01 = stablehlo.reshape %w01 : (tensor<8x1xf64>) -> tensor<8xf64>
  %v02 = stablehlo.reshape %w02 : (tensor<8x1xf64>) -> tensor<8xf64>
  %v10 = stablehlo.reshape %w10 : (tensor<8x1xf64>) -> tensor<8xf64>
  %v11 = stablehlo.reshape %w11 : (tensor<8x1xf64>) -> tensor<8xf64>
  %v12 = stablehlo.reshape %w12 : (tensor<8x1xf64>) -> tensor<8xf64>
  %m00 = stablehlo.multiply %v00, %r0 : tensor<8xf64>
  %m01 = stablehlo.multiply %v01, %r1 : tensor<8xf64>
  %m02 = stablehlo.multiply %v02, %r2 : tensor<8xf64>
  %m10 = stablehlo.multiply %v10, %r0 : tensor<8xf64>
  %m11 = stablehlo.multiply %v11, %r1 : tensor<8xf64>
  %m12 = stablehlo.multiply %v12, %r2 : tensor<8xf64>
  %a0 = stablehlo.add %m00, %m01 : tensor<8xf64>
  %y0 = stablehlo.add %a0, %m02 : tensor<8xf64>
  %a1 = stablehlo.add %m10, %m11 : tensor<8xf64>
  %y1 = stablehlo.add %a1, %m12 : tensor<8xf64>
  return %y0, %y1 : tensor<8xf64>, tensor<8xf64>
}

// CHECK-LABEL: func.func @shared_views(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x4xf64>, %[[V1:[^ ,:)]+]]: tensor<2x4xf64>) -> (tensor<8xf64>, tensor<8xf64>) {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.slice %[[V1]] [0:1, 0:3] : (tensor<2x4xf64>) -> tensor<1x3xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reshape %[[V3]] : (tensor<1x3xf64>) -> tensor<1x3x1xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.transpose %[[V4]], dims = [1, 0, 2] : (tensor<1x3x1xf64>) -> tensor<3x1x1xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V5]], dims = [0, 1, 2] : (tensor<3x1x1xf64>) -> tensor<3x8x1xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.reshape %[[V6]] : (tensor<3x8x1xf64>) -> tensor<3x8xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.slice %[[V0]] [0:8, 0:3] : (tensor<8x4xf64>) -> tensor<8x3xf64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.reshape %[[V8]] : (tensor<8x3xf64>) -> tensor<8x3x1xf64>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = stablehlo.transpose %[[V9]], dims = [1, 0, 2] : (tensor<8x3x1xf64>) -> tensor<3x8x1xf64>
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = stablehlo.reshape %[[V10]] : (tensor<3x8x1xf64>) -> tensor<3x8xf64>
// CHECK-NEXT: %[[V12:[^ ,:)]+]] = stablehlo.multiply %[[V7]], %[[V11]] : tensor<3x8xf64>
// CHECK-NEXT: %[[V13:[^ ,:)]+]] = stablehlo.reduce(%[[V12]] init: %[[V2]]) applies stablehlo.add across dimensions = [0] : (tensor<3x8xf64>, tensor<f64>) -> tensor<8xf64>
// CHECK-NEXT: %[[V14:[^ ,:)]+]] = stablehlo.slice %[[V1]] [1:2, 0:3] : (tensor<2x4xf64>) -> tensor<1x3xf64>
// CHECK-NEXT: %[[V15:[^ ,:)]+]] = stablehlo.reshape %[[V14]] : (tensor<1x3xf64>) -> tensor<1x3x1xf64>
// CHECK-NEXT: %[[V16:[^ ,:)]+]] = stablehlo.transpose %[[V15]], dims = [1, 0, 2] : (tensor<1x3x1xf64>) -> tensor<3x1x1xf64>
// CHECK-NEXT: %[[V17:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V16]], dims = [0, 1, 2] : (tensor<3x1x1xf64>) -> tensor<3x8x1xf64>
// CHECK-NEXT: %[[V18:[^ ,:)]+]] = stablehlo.reshape %[[V17]] : (tensor<3x8x1xf64>) -> tensor<3x8xf64>
// CHECK-NEXT: %[[V19:[^ ,:)]+]] = stablehlo.slice %[[V0]] [0:8, 0:3] : (tensor<8x4xf64>) -> tensor<8x3xf64>
// CHECK-NEXT: %[[V20:[^ ,:)]+]] = stablehlo.reshape %[[V19]] : (tensor<8x3xf64>) -> tensor<8x3x1xf64>
// CHECK-NEXT: %[[V21:[^ ,:)]+]] = stablehlo.transpose %[[V20]], dims = [1, 0, 2] : (tensor<8x3x1xf64>) -> tensor<3x8x1xf64>
// CHECK-NEXT: %[[V22:[^ ,:)]+]] = stablehlo.reshape %[[V21]] : (tensor<3x8x1xf64>) -> tensor<3x8xf64>
// CHECK-NEXT: %[[V23:[^ ,:)]+]] = stablehlo.multiply %[[V18]], %[[V22]] : tensor<3x8xf64>
// CHECK-NEXT: %[[V24:[^ ,:)]+]] = stablehlo.reduce(%[[V23]] init: %[[V2]]) applies stablehlo.add across dimensions = [0] : (tensor<3x8xf64>, tensor<f64>) -> tensor<8xf64>
// CHECK-NEXT: return %[[V13]], %[[V24]] : tensor<8xf64>, tensor<8xf64>
// CHECK-NEXT: }

// The terms come in the order 0, 2, 1: ordered by their slices, they stack.
func.func @out_of_order(%x: tensor<8x3xf64>, %y: tensor<8x3xf64>) -> tensor<8x1xf64> {
  %x0 = stablehlo.slice %x [0:8, 0:1] : (tensor<8x3xf64>) -> tensor<8x1xf64>
  %x1 = stablehlo.slice %x [0:8, 1:2] : (tensor<8x3xf64>) -> tensor<8x1xf64>
  %x2 = stablehlo.slice %x [0:8, 2:3] : (tensor<8x3xf64>) -> tensor<8x1xf64>
  %y0 = stablehlo.slice %y [0:8, 0:1] : (tensor<8x3xf64>) -> tensor<8x1xf64>
  %y1 = stablehlo.slice %y [0:8, 1:2] : (tensor<8x3xf64>) -> tensor<8x1xf64>
  %y2 = stablehlo.slice %y [0:8, 2:3] : (tensor<8x3xf64>) -> tensor<8x1xf64>
  %m0 = stablehlo.multiply %x0, %y0 : tensor<8x1xf64>
  %m1 = stablehlo.multiply %x1, %y1 : tensor<8x1xf64>
  %m2 = stablehlo.multiply %x2, %y2 : tensor<8x1xf64>
  %a = stablehlo.add %m0, %m2 : tensor<8x1xf64>
  %r = stablehlo.add %a, %m1 : tensor<8x1xf64>
  return %r : tensor<8x1xf64>
}

// CHECK-LABEL: func.func @out_of_order(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x3xf64>, %[[V1:[^ ,:)]+]]: tensor<8x3xf64>) -> tensor<8x1xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<8x3xf64>) -> tensor<8x3x1xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.transpose %[[V3]], dims = [1, 0, 2] : (tensor<8x3x1xf64>) -> tensor<3x8x1xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<8x3xf64>) -> tensor<8x3x1xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.transpose %[[V5]], dims = [1, 0, 2] : (tensor<8x3x1xf64>) -> tensor<3x8x1xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.multiply %[[V4]], %[[V6]] : tensor<3x8x1xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.reduce(%[[V7]] init: %[[V2]]) applies stablehlo.add across dimensions = [0] : (tensor<3x8x1xf64>, tensor<f64>) -> tensor<8x1xf64>
// CHECK-NEXT: return %[[V8]] : tensor<8x1xf64>
// CHECK-NEXT: }

// Terms that match through more than eight ops still roll up.
func.func @deep(%x: tensor<8x3xf64>) -> tensor<8x1xf64> {
  %x0 = stablehlo.slice %x [0:8, 0:1] : (tensor<8x3xf64>) -> tensor<8x1xf64>
  %x1 = stablehlo.slice %x [0:8, 1:2] : (tensor<8x3xf64>) -> tensor<8x1xf64>
  %x2 = stablehlo.slice %x [0:8, 2:3] : (tensor<8x3xf64>) -> tensor<8x1xf64>
  %c0_0 = stablehlo.negate %x0 : tensor<8x1xf64>
  %c0_1 = stablehlo.exponential %c0_0 : tensor<8x1xf64>
  %c0_2 = stablehlo.negate %c0_1 : tensor<8x1xf64>
  %c0_3 = stablehlo.exponential %c0_2 : tensor<8x1xf64>
  %c0_4 = stablehlo.negate %c0_3 : tensor<8x1xf64>
  %c0_5 = stablehlo.exponential %c0_4 : tensor<8x1xf64>
  %c0_6 = stablehlo.negate %c0_5 : tensor<8x1xf64>
  %c0_7 = stablehlo.exponential %c0_6 : tensor<8x1xf64>
  %c0_8 = stablehlo.negate %c0_7 : tensor<8x1xf64>
  %c0_9 = stablehlo.exponential %c0_8 : tensor<8x1xf64>
  %c1_0 = stablehlo.negate %x1 : tensor<8x1xf64>
  %c1_1 = stablehlo.exponential %c1_0 : tensor<8x1xf64>
  %c1_2 = stablehlo.negate %c1_1 : tensor<8x1xf64>
  %c1_3 = stablehlo.exponential %c1_2 : tensor<8x1xf64>
  %c1_4 = stablehlo.negate %c1_3 : tensor<8x1xf64>
  %c1_5 = stablehlo.exponential %c1_4 : tensor<8x1xf64>
  %c1_6 = stablehlo.negate %c1_5 : tensor<8x1xf64>
  %c1_7 = stablehlo.exponential %c1_6 : tensor<8x1xf64>
  %c1_8 = stablehlo.negate %c1_7 : tensor<8x1xf64>
  %c1_9 = stablehlo.exponential %c1_8 : tensor<8x1xf64>
  %c2_0 = stablehlo.negate %x2 : tensor<8x1xf64>
  %c2_1 = stablehlo.exponential %c2_0 : tensor<8x1xf64>
  %c2_2 = stablehlo.negate %c2_1 : tensor<8x1xf64>
  %c2_3 = stablehlo.exponential %c2_2 : tensor<8x1xf64>
  %c2_4 = stablehlo.negate %c2_3 : tensor<8x1xf64>
  %c2_5 = stablehlo.exponential %c2_4 : tensor<8x1xf64>
  %c2_6 = stablehlo.negate %c2_5 : tensor<8x1xf64>
  %c2_7 = stablehlo.exponential %c2_6 : tensor<8x1xf64>
  %c2_8 = stablehlo.negate %c2_7 : tensor<8x1xf64>
  %c2_9 = stablehlo.exponential %c2_8 : tensor<8x1xf64>
  %a = stablehlo.add %c0_9, %c1_9 : tensor<8x1xf64>
  %r = stablehlo.add %a, %c2_9 : tensor<8x1xf64>
  return %r : tensor<8x1xf64>
}

// CHECK-LABEL: func.func @deep(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x3xf64>) -> tensor<8x1xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<8x3xf64>) -> tensor<8x3x1xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.transpose %[[V2]], dims = [1, 0, 2] : (tensor<8x3x1xf64>) -> tensor<3x8x1xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.negate %[[V3]] : tensor<3x8x1xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.exponential %[[V4]] : tensor<3x8x1xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.negate %[[V5]] : tensor<3x8x1xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.exponential %[[V6]] : tensor<3x8x1xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.negate %[[V7]] : tensor<3x8x1xf64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.exponential %[[V8]] : tensor<3x8x1xf64>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = stablehlo.negate %[[V9]] : tensor<3x8x1xf64>
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = stablehlo.exponential %[[V10]] : tensor<3x8x1xf64>
// CHECK-NEXT: %[[V12:[^ ,:)]+]] = stablehlo.negate %[[V11]] : tensor<3x8x1xf64>
// CHECK-NEXT: %[[V13:[^ ,:)]+]] = stablehlo.exponential %[[V12]] : tensor<3x8x1xf64>
// CHECK-NEXT: %[[V14:[^ ,:)]+]] = stablehlo.reduce(%[[V13]] init: %[[V1]]) applies stablehlo.add across dimensions = [0] : (tensor<3x8x1xf64>, tensor<f64>) -> tensor<8x1xf64>
// CHECK-NEXT: return %[[V14]] : tensor<8x1xf64>
// CHECK-NEXT: }

// One term carries an analysis attribute the others do not: they still roll
// up, and the stacked op does not keep it.
func.func @analysis_attr(%x: tensor<8x3xf64>, %y: tensor<8x3xf64>) -> tensor<8x1xf64> {
  %x0 = stablehlo.slice %x [0:8, 0:1] : (tensor<8x3xf64>) -> tensor<8x1xf64>
  %x1 = stablehlo.slice %x [0:8, 1:2] : (tensor<8x3xf64>) -> tensor<8x1xf64>
  %x2 = stablehlo.slice %x [0:8, 2:3] : (tensor<8x3xf64>) -> tensor<8x1xf64>
  %y0 = stablehlo.slice %y [0:8, 0:1] : (tensor<8x3xf64>) -> tensor<8x1xf64>
  %y1 = stablehlo.slice %y [0:8, 1:2] : (tensor<8x3xf64>) -> tensor<8x1xf64>
  %y2 = stablehlo.slice %y [0:8, 2:3] : (tensor<8x3xf64>) -> tensor<8x1xf64>
  %m0 = stablehlo.multiply %x0, %y0 {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<8x1xf64>
  %m1 = stablehlo.multiply %x1, %y1 : tensor<8x1xf64>
  %m2 = stablehlo.multiply %x2, %y2 : tensor<8x1xf64>
  %a = stablehlo.add %m0, %m1 : tensor<8x1xf64>
  %r = stablehlo.add %a, %m2 : tensor<8x1xf64>
  return %r : tensor<8x1xf64>
}

// CHECK-LABEL: func.func @analysis_attr(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x3xf64>, %[[V1:[^ ,:)]+]]: tensor<8x3xf64>) -> tensor<8x1xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<8x3xf64>) -> tensor<8x3x1xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.transpose %[[V3]], dims = [1, 0, 2] : (tensor<8x3x1xf64>) -> tensor<3x8x1xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<8x3xf64>) -> tensor<8x3x1xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.transpose %[[V5]], dims = [1, 0, 2] : (tensor<8x3x1xf64>) -> tensor<3x8x1xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.multiply %[[V4]], %[[V6]] : tensor<3x8x1xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.reduce(%[[V7]] init: %[[V2]]) applies stablehlo.add across dimensions = [0] : (tensor<3x8x1xf64>, tensor<f64>) -> tensor<8x1xf64>
// CHECK-NEXT: return %[[V8]] : tensor<8x1xf64>
// CHECK-NEXT: }

// Each term reduces its own window of %x: the windows stack, and the reduce
// runs once over the stack (one dimension further in).
func.func @through_reduce(%x: tensor<8x9xf64>, %w: tensor<3xf64>, %b: tensor<3xf64>) -> tensor<8xf64> {
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %bb = stablehlo.broadcast_in_dim %b, dims = [1] : (tensor<3xf64>) -> tensor<8x3xf64>
  %x0 = stablehlo.slice %x [0:8, 0:3] : (tensor<8x9xf64>) -> tensor<8x3xf64>
  %x1 = stablehlo.slice %x [0:8, 3:6] : (tensor<8x9xf64>) -> tensor<8x3xf64>
  %x2 = stablehlo.slice %x [0:8, 6:9] : (tensor<8x9xf64>) -> tensor<8x3xf64>
  %m0 = stablehlo.multiply %bb, %x0 : tensor<8x3xf64>
  %m1 = stablehlo.multiply %bb, %x1 : tensor<8x3xf64>
  %m2 = stablehlo.multiply %bb, %x2 : tensor<8x3xf64>
  %r0 = stablehlo.reduce(%m0 init: %z) applies stablehlo.add across dimensions = [1] : (tensor<8x3xf64>, tensor<f64>) -> tensor<8xf64>
  %r1 = stablehlo.reduce(%m1 init: %z) applies stablehlo.add across dimensions = [1] : (tensor<8x3xf64>, tensor<f64>) -> tensor<8xf64>
  %r2 = stablehlo.reduce(%m2 init: %z) applies stablehlo.add across dimensions = [1] : (tensor<8x3xf64>, tensor<f64>) -> tensor<8xf64>
  %w0 = stablehlo.slice %w [0:1] : (tensor<3xf64>) -> tensor<1xf64>
  %w1 = stablehlo.slice %w [1:2] : (tensor<3xf64>) -> tensor<1xf64>
  %w2 = stablehlo.slice %w [2:3] : (tensor<3xf64>) -> tensor<1xf64>
  %v0 = stablehlo.broadcast_in_dim %w0, dims = [0] : (tensor<1xf64>) -> tensor<8xf64>
  %v1 = stablehlo.broadcast_in_dim %w1, dims = [0] : (tensor<1xf64>) -> tensor<8xf64>
  %v2 = stablehlo.broadcast_in_dim %w2, dims = [0] : (tensor<1xf64>) -> tensor<8xf64>
  %t0 = stablehlo.multiply %v0, %r0 : tensor<8xf64>
  %t1 = stablehlo.multiply %v1, %r1 : tensor<8xf64>
  %t2 = stablehlo.multiply %v2, %r2 : tensor<8xf64>
  %a = stablehlo.add %t0, %t1 : tensor<8xf64>
  %s = stablehlo.add %a, %t2 : tensor<8xf64>
  return %s : tensor<8xf64>
}

// CHECK-LABEL: func.func @through_reduce(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x9xf64>, %[[V1:[^ ,:)]+]]: tensor<3xf64>, %[[V2:[^ ,:)]+]]: tensor<3xf64>) -> tensor<8xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V2]], dims = [1] : (tensor<3xf64>) -> tensor<8x3xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<3xf64>) -> tensor<3x1xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V5]], dims = [0, 1] : (tensor<3x1xf64>) -> tensor<3x8xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V4]], dims = [1, 2] : (tensor<8x3xf64>) -> tensor<3x8x3xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<8x9xf64>) -> tensor<8x3x3xf64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.transpose %[[V8]], dims = [1, 0, 2] : (tensor<8x3x3xf64>) -> tensor<3x8x3xf64>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = stablehlo.multiply %[[V7]], %[[V9]] : tensor<3x8x3xf64>
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = stablehlo.reduce(%[[V10]] init: %[[V3]]) applies stablehlo.add across dimensions = [2] : (tensor<3x8x3xf64>, tensor<f64>) -> tensor<3x8xf64>
// CHECK-NEXT: %[[V12:[^ ,:)]+]] = stablehlo.multiply %[[V6]], %[[V11]] : tensor<3x8xf64>
// CHECK-NEXT: %[[V13:[^ ,:)]+]] = stablehlo.reduce(%[[V12]] init: %[[V3]]) applies stablehlo.add across dimensions = [0] : (tensor<3x8xf64>, tensor<f64>) -> tensor<8xf64>
// CHECK-NEXT: return %[[V13]] : tensor<8xf64>
// CHECK-NEXT: }

// Each term contracts %a with its own window of %x: the windows stack, and
// one dot_general, batched over the stack, replaces the three.
func.func @through_dot(%x: tensor<8x12xf64>, %a: tensor<3x4xf64>) -> tensor<3x8xf64> {
  %x0 = stablehlo.slice %x [0:8, 0:4] : (tensor<8x12xf64>) -> tensor<8x4xf64>
  %x1 = stablehlo.slice %x [0:8, 4:8] : (tensor<8x12xf64>) -> tensor<8x4xf64>
  %x2 = stablehlo.slice %x [0:8, 8:12] : (tensor<8x12xf64>) -> tensor<8x4xf64>
  %d0 = stablehlo.dot_general %a, %x0, contracting_dims = [1] x [1] : (tensor<3x4xf64>, tensor<8x4xf64>) -> tensor<3x8xf64>
  %d1 = stablehlo.dot_general %a, %x1, contracting_dims = [1] x [1] : (tensor<3x4xf64>, tensor<8x4xf64>) -> tensor<3x8xf64>
  %d2 = stablehlo.dot_general %a, %x2, contracting_dims = [1] x [1] : (tensor<3x4xf64>, tensor<8x4xf64>) -> tensor<3x8xf64>
  %s0 = stablehlo.add %d0, %d1 : tensor<3x8xf64>
  %s = stablehlo.add %s0, %d2 : tensor<3x8xf64>
  return %s : tensor<3x8xf64>
}

// CHECK-LABEL: func.func @through_dot(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x12xf64>, %[[V1:[^ ,:)]+]]: tensor<3x4xf64>) -> tensor<3x8xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V1]], dims = [1, 2] : (tensor<3x4xf64>) -> tensor<3x3x4xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.reshape %[[V0]] : (tensor<8x12xf64>) -> tensor<8x3x4xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.transpose %[[V4]], dims = [1, 0, 2] : (tensor<8x3x4xf64>) -> tensor<3x8x4xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.dot_general %[[V3]], %[[V5]], batching_dims = [0] x [0], contracting_dims = [2] x [2] : (tensor<3x3x4xf64>, tensor<3x8x4xf64>) -> tensor<3x3x8xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.reduce(%[[V6]] init: %[[V2]]) applies stablehlo.add across dimensions = [0] : (tensor<3x3x8xf64>, tensor<f64>) -> tensor<3x8xf64>
// CHECK-NEXT: return %[[V7]] : tensor<3x8xf64>
// CHECK-NEXT: }

// The terms' scalar broadcasts spell the unit operand dimensions
// differently (dims = [2, 1] in the second): the same value, so the terms
// are computed alike and roll up.
func.func @broadcast_spelling(%b: tensor<3x4xf64>, %x: tensor<8x4x4x3xf64>) -> tensor<8x4x4xf64> {
  %c0 = stablehlo.slice %b [0:1, 0:1] : (tensor<3x4xf64>) -> tensor<1x1xf64>
  %c1 = stablehlo.slice %b [1:2, 0:1] : (tensor<3x4xf64>) -> tensor<1x1xf64>
  %c2 = stablehlo.slice %b [2:3, 0:1] : (tensor<3x4xf64>) -> tensor<1x1xf64>
  %x0 = stablehlo.slice %x [0:8, 0:4, 0:4, 0:1] : (tensor<8x4x4x3xf64>) -> tensor<8x4x4x1xf64>
  %x1 = stablehlo.slice %x [0:8, 0:4, 0:4, 1:2] : (tensor<8x4x4x3xf64>) -> tensor<8x4x4x1xf64>
  %x2 = stablehlo.slice %x [0:8, 0:4, 0:4, 2:3] : (tensor<8x4x4x3xf64>) -> tensor<8x4x4x1xf64>
  %k0 = stablehlo.broadcast_in_dim %c0, dims = [1, 2] : (tensor<1x1xf64>) -> tensor<8x4x4xf64>
  %k1 = stablehlo.broadcast_in_dim %c1, dims = [2, 1] : (tensor<1x1xf64>) -> tensor<8x4x4xf64>
  %k2 = stablehlo.broadcast_in_dim %c2, dims = [1, 2] : (tensor<1x1xf64>) -> tensor<8x4x4xf64>
  %r0 = stablehlo.reshape %x0 : (tensor<8x4x4x1xf64>) -> tensor<8x4x4xf64>
  %r1 = stablehlo.reshape %x1 : (tensor<8x4x4x1xf64>) -> tensor<8x4x4xf64>
  %r2 = stablehlo.reshape %x2 : (tensor<8x4x4x1xf64>) -> tensor<8x4x4xf64>
  %t0 = stablehlo.multiply %k0, %r0 : tensor<8x4x4xf64>
  %t1 = stablehlo.multiply %k1, %r1 : tensor<8x4x4xf64>
  %t2 = stablehlo.multiply %k2, %r2 : tensor<8x4x4xf64>
  %a0 = stablehlo.add %t0, %t1 : tensor<8x4x4xf64>
  %a1 = stablehlo.add %a0, %t2 : tensor<8x4x4xf64>
  return %a1 : tensor<8x4x4xf64>
}

// CHECK-LABEL: func.func @broadcast_spelling(
// CHECK-SAME: %[[B:[^ ,:)]+]]: tensor<3x4xf64>, %[[X:[^ ,:)]+]]: tensor<8x4x4x3xf64>) -> tensor<8x4x4xf64> {
// CHECK-NEXT: %[[Z:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[S:[^ ,:)]+]] = stablehlo.slice %[[B]] [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
// CHECK-NEXT: %[[SR:[^ ,:)]+]] = stablehlo.reshape %[[S]] : (tensor<3x1xf64>) -> tensor<3x1x1xf64>
// CHECK-NEXT: %[[K:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[SR]], dims = [0, 2, 3] : (tensor<3x1x1xf64>) -> tensor<3x8x4x4xf64>
// CHECK-NEXT: %[[XR:[^ ,:)]+]] = stablehlo.reshape %[[X]] : (tensor<8x4x4x3xf64>) -> tensor<8x4x4x3x1xf64>
// CHECK-NEXT: %[[XT:[^ ,:)]+]] = stablehlo.transpose %[[XR]], dims = [3, 0, 1, 2, 4] : (tensor<8x4x4x3x1xf64>) -> tensor<3x8x4x4x1xf64>
// CHECK-NEXT: %[[XS:[^ ,:)]+]] = stablehlo.reshape %[[XT]] : (tensor<3x8x4x4x1xf64>) -> tensor<3x8x4x4xf64>
// CHECK-NEXT: %[[M:[^ ,:)]+]] = stablehlo.multiply %[[K]], %[[XS]] : tensor<3x8x4x4xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.reduce(%[[M]] init: %[[Z]]) applies stablehlo.add across dimensions = [0] : (tensor<3x8x4x4xf64>, tensor<f64>) -> tensor<8x4x4xf64>
// CHECK-NEXT: return %[[R]] : tensor<8x4x4xf64>
// CHECK-NEXT: }

// Two sums sharing one term (the same multiply read by both add trees):
// an elementwise term is cheap to recompute stacked (its consumer fuses
// it), so each sum rolls up whole and the shared multiply goes away.
func.func @two_sums(%b: tensor<3x4xf64>, %g: tensor<3x4xf64>, %x: tensor<8x4x4x3xf64>) -> (tensor<8x4x4xf64>, tensor<8x4x4xf64>) {
  %c0 = stablehlo.slice %b [0:1, 0:1] : (tensor<3x4xf64>) -> tensor<1x1xf64>
  %c1 = stablehlo.slice %b [1:2, 0:1] : (tensor<3x4xf64>) -> tensor<1x1xf64>
  %c2 = stablehlo.slice %b [2:3, 0:1] : (tensor<3x4xf64>) -> tensor<1x1xf64>
  %d0 = stablehlo.slice %g [0:1, 0:1] : (tensor<3x4xf64>) -> tensor<1x1xf64>
  %d1 = stablehlo.slice %g [1:2, 0:1] : (tensor<3x4xf64>) -> tensor<1x1xf64>
  %x0 = stablehlo.slice %x [0:8, 0:4, 0:4, 0:1] : (tensor<8x4x4x3xf64>) -> tensor<8x4x4x1xf64>
  %x1 = stablehlo.slice %x [0:8, 0:4, 0:4, 1:2] : (tensor<8x4x4x3xf64>) -> tensor<8x4x4x1xf64>
  %x2 = stablehlo.slice %x [0:8, 0:4, 0:4, 2:3] : (tensor<8x4x4x3xf64>) -> tensor<8x4x4x1xf64>
  %k0 = stablehlo.broadcast_in_dim %c0, dims = [1, 2] : (tensor<1x1xf64>) -> tensor<8x4x4xf64>
  %k1 = stablehlo.broadcast_in_dim %c1, dims = [1, 2] : (tensor<1x1xf64>) -> tensor<8x4x4xf64>
  %k2 = stablehlo.broadcast_in_dim %c2, dims = [1, 2] : (tensor<1x1xf64>) -> tensor<8x4x4xf64>
  %l0 = stablehlo.broadcast_in_dim %d0, dims = [1, 2] : (tensor<1x1xf64>) -> tensor<8x4x4xf64>
  %l1 = stablehlo.broadcast_in_dim %d1, dims = [1, 2] : (tensor<1x1xf64>) -> tensor<8x4x4xf64>
  %r0 = stablehlo.reshape %x0 : (tensor<8x4x4x1xf64>) -> tensor<8x4x4xf64>
  %r1 = stablehlo.reshape %x1 : (tensor<8x4x4x1xf64>) -> tensor<8x4x4xf64>
  %r2 = stablehlo.reshape %x2 : (tensor<8x4x4x1xf64>) -> tensor<8x4x4xf64>
  %t0 = stablehlo.multiply %k0, %r0 : tensor<8x4x4xf64>
  %t1 = stablehlo.multiply %k1, %r1 : tensor<8x4x4xf64>
  %t2 = stablehlo.multiply %k2, %r2 : tensor<8x4x4xf64>
  %u0 = stablehlo.multiply %l0, %r0 : tensor<8x4x4xf64>
  %u1 = stablehlo.multiply %l1, %r1 : tensor<8x4x4xf64>
  %a0 = stablehlo.add %t0, %t1 : tensor<8x4x4xf64>
  %a1 = stablehlo.add %a0, %t2 : tensor<8x4x4xf64>
  %s0 = stablehlo.add %u0, %u1 : tensor<8x4x4xf64>
  %s1 = stablehlo.add %s0, %t2 : tensor<8x4x4xf64>
  return %a1, %s1 : tensor<8x4x4xf64>, tensor<8x4x4xf64>
}
// CHECK-LABEL: func.func @two_sums(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<3x4xf64>, %[[V1:[^ ,:)]+]]: tensor<3x4xf64>, %[[V2:[^ ,:)]+]]: tensor<8x4x4x3xf64>) -> (tensor<8x4x4xf64>, tensor<8x4x4xf64>) {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.slice %[[V0]] [2:3, 0:1] : (tensor<3x4xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.slice %[[V1]] [0:1, 0:1] : (tensor<3x4xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.slice %[[V1]] [1:2, 0:1] : (tensor<3x4xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.slice %[[V2]] [0:8, 0:4, 0:4, 0:1] : (tensor<8x4x4x3xf64>) -> tensor<8x4x4x1xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.slice %[[V2]] [0:8, 0:4, 0:4, 1:2] : (tensor<8x4x4x3xf64>) -> tensor<8x4x4x1xf64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.slice %[[V2]] [0:8, 0:4, 0:4, 2:3] : (tensor<8x4x4x3xf64>) -> tensor<8x4x4x1xf64>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V4]], dims = [1, 2] : (tensor<1x1xf64>) -> tensor<8x4x4xf64>
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V5]], dims = [1, 2] : (tensor<1x1xf64>) -> tensor<8x4x4xf64>
// CHECK-NEXT: %[[V12:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V6]], dims = [1, 2] : (tensor<1x1xf64>) -> tensor<8x4x4xf64>
// CHECK-NEXT: %[[V13:[^ ,:)]+]] = stablehlo.reshape %[[V7]] : (tensor<8x4x4x1xf64>) -> tensor<8x4x4xf64>
// CHECK-NEXT: %[[V14:[^ ,:)]+]] = stablehlo.reshape %[[V8]] : (tensor<8x4x4x1xf64>) -> tensor<8x4x4xf64>
// CHECK-NEXT: %[[V15:[^ ,:)]+]] = stablehlo.reshape %[[V9]] : (tensor<8x4x4x1xf64>) -> tensor<8x4x4xf64>
// CHECK-NEXT: %[[V16:[^ ,:)]+]] = stablehlo.multiply %[[V10]], %[[V15]] : tensor<8x4x4xf64>
// CHECK-NEXT: %[[V17:[^ ,:)]+]] = stablehlo.multiply %[[V11]], %[[V13]] : tensor<8x4x4xf64>
// CHECK-NEXT: %[[V18:[^ ,:)]+]] = stablehlo.multiply %[[V12]], %[[V14]] : tensor<8x4x4xf64>
// CHECK-NEXT: %[[V19:[^ ,:)]+]] = stablehlo.slice %[[V0]] [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
// CHECK-NEXT: %[[V20:[^ ,:)]+]] = stablehlo.reshape %[[V19]] : (tensor<3x1xf64>) -> tensor<3x1x1xf64>
// CHECK-NEXT: %[[V21:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V20]], dims = [0, 2, 3] : (tensor<3x1x1xf64>) -> tensor<3x8x4x4xf64>
// CHECK-NEXT: %[[V22:[^ ,:)]+]] = stablehlo.reshape %[[V2]] : (tensor<8x4x4x3xf64>) -> tensor<8x4x4x3x1xf64>
// CHECK-NEXT: %[[V23:[^ ,:)]+]] = stablehlo.transpose %[[V22]], dims = [3, 0, 1, 2, 4] : (tensor<8x4x4x3x1xf64>) -> tensor<3x8x4x4x1xf64>
// CHECK-NEXT: %[[V24:[^ ,:)]+]] = stablehlo.reshape %[[V23]] : (tensor<3x8x4x4x1xf64>) -> tensor<3x8x4x4xf64>
// CHECK-NEXT: %[[V25:[^ ,:)]+]] = stablehlo.multiply %[[V21]], %[[V24]] : tensor<3x8x4x4xf64>
// CHECK-NEXT: %[[V26:[^ ,:)]+]] = stablehlo.reduce(%[[V25]] init: %[[V3]]) applies stablehlo.add across dimensions = [0] : (tensor<3x8x4x4xf64>, tensor<f64>) -> tensor<8x4x4xf64>
// CHECK-NEXT: %[[V27:[^ ,:)]+]] = stablehlo.add %[[V17]], %[[V18]] : tensor<8x4x4xf64>
// CHECK-NEXT: %[[V28:[^ ,:)]+]] = stablehlo.add %[[V27]], %[[V16]] : tensor<8x4x4xf64>
// CHECK-NEXT: return %[[V26]], %[[V28]] : tensor<8x4x4xf64>, tensor<8x4x4xf64>
// CHECK-NEXT: }
