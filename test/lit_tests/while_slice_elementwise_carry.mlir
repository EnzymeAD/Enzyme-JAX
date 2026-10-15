// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=while_slice_elementwise_carry" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// A lane accumulator over lanes padded to 8 of which the loop's reader keeps
// 3x3: the loop carries only that box, adding the box of each term.
func.func @accumulate(%x: tensor<8x8xf64>, %y: tensor<8x8xf64>) -> tensor<3x3xf64> {
  %zero = stablehlo.constant dense<0.0> : tensor<8x8xf64>
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c3 = stablehlo.constant dense<3> : tensor<i64>
  %w:2 = stablehlo.while(%i = %c0, %acc = %zero) : tensor<i64>, tensor<8x8xf64>
  cond {
    %lt = stablehlo.compare LT, %i, %c3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %lt : tensor<i1>
  } do {
    %p = stablehlo.multiply %x, %y : tensor<8x8xf64>
    %s = stablehlo.add %acc, %p : tensor<8x8xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %s : tensor<i64>, tensor<8x8xf64>
  }
  %r = stablehlo.slice %w#1 [0:3, 0:3] : (tensor<8x8xf64>) -> tensor<3x3xf64>
  return %r : tensor<3x3xf64>
}

// The carried value is also read by position: unchanged.
func.func @gathered(%x: tensor<8x8xf64>, %idx: tensor<8x8x2xi64>) -> tensor<3x3xf64> {
  %zero = stablehlo.constant dense<0.0> : tensor<8x8xf64>
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c3 = stablehlo.constant dense<3> : tensor<i64>
  %w:2 = stablehlo.while(%i = %c0, %acc = %zero) : tensor<i64>, tensor<8x8xf64>
  cond {
    %lt = stablehlo.compare LT, %i, %c3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %lt : tensor<i1>
  } do {
    %g = "stablehlo.gather"(%acc, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1>}> : (tensor<8x8xf64>, tensor<8x8x2xi64>) -> tensor<8x8xf64>
    %s = stablehlo.add %g, %x : tensor<8x8xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %s : tensor<i64>, tensor<8x8xf64>
  }
  %r = stablehlo.slice %w#1 [0:3, 0:3] : (tensor<8x8xf64>) -> tensor<3x3xf64>
  return %r : tensor<3x3xf64>
}

// The result is read whole: unchanged.
func.func @whole(%x: tensor<8x8xf64>) -> (tensor<3x3xf64>, tensor<8x8xf64>) {
  %zero = stablehlo.constant dense<0.0> : tensor<8x8xf64>
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c3 = stablehlo.constant dense<3> : tensor<i64>
  %w:2 = stablehlo.while(%i = %c0, %acc = %zero) : tensor<i64>, tensor<8x8xf64>
  cond {
    %lt = stablehlo.compare LT, %i, %c3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %lt : tensor<i1>
  } do {
    %s = stablehlo.add %acc, %x : tensor<8x8xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %s : tensor<i64>, tensor<8x8xf64>
  }
  %r = stablehlo.slice %w#1 [0:3, 0:3] : (tensor<8x8xf64>) -> tensor<3x3xf64>
  return %r, %w#1 : tensor<3x3xf64>, tensor<8x8xf64>
}

// The carried value also feeds another carried value: unchanged.
func.func @feeds_other(%x: tensor<8x8xf64>) -> (tensor<3x3xf64>, tensor<8x8xf64>) {
  %zero = stablehlo.constant dense<0.0> : tensor<8x8xf64>
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c3 = stablehlo.constant dense<3> : tensor<i64>
  %w:3 = stablehlo.while(%i = %c0, %acc = %zero, %other = %zero) : tensor<i64>, tensor<8x8xf64>, tensor<8x8xf64>
  cond {
    %lt = stablehlo.compare LT, %i, %c3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %lt : tensor<i1>
  } do {
    %s = stablehlo.add %acc, %x : tensor<8x8xf64>
    %t = stablehlo.add %other, %acc : tensor<8x8xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %s, %t : tensor<i64>, tensor<8x8xf64>, tensor<8x8xf64>
  }
  %r = stablehlo.slice %w#1 [0:3, 0:3] : (tensor<8x8xf64>) -> tensor<3x3xf64>
  return %r, %w#2 : tensor<3x3xf64>, tensor<8x8xf64>
}

// CHECK-LABEL: func.func @accumulate(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x8xf64>, %[[V1:[^ ,:)]+]]: tensor<8x8xf64>) -> tensor<3x3xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<8x8xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.slice %[[V2]] [0:3, 0:3] : (tensor<8x8xf64>) -> tensor<3x3xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]]:2 = stablehlo.while(%[[V8:[^ ,:)]+]] = %[[V3]], %[[V9:[^ ,:)]+]] = %[[V6]]) : tensor<i64>, tensor<3x3xf64>
// CHECK-NEXT: cond {
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = stablehlo.compare LT, %[[V8]], %[[V5]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT: stablehlo.return %[[V10]] : tensor<i1>
// CHECK-NEXT: } do {
// CHECK-NEXT: %[[V10]] = stablehlo.multiply %[[V0]], %[[V1]] : tensor<8x8xf64>
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = stablehlo.slice %[[V10]] [0:3, 0:3] : (tensor<8x8xf64>) -> tensor<3x3xf64>
// CHECK-NEXT: %[[V12:[^ ,:)]+]] = stablehlo.add %[[V9]], %[[V11]] : tensor<3x3xf64>
// CHECK-NEXT: %[[V13:[^ ,:)]+]] = stablehlo.add %[[V8]], %[[V4]] : tensor<i64>
// CHECK-NEXT: stablehlo.return %[[V13]], %[[V12]] : tensor<i64>, tensor<3x3xf64>
// CHECK-NEXT: }
// CHECK-NEXT: %[[V14:[^ ,:)]+]] = stablehlo.slice %[[V7]]#1 [0:3, 0:3] : (tensor<3x3xf64>) -> tensor<3x3xf64>
// CHECK-NEXT: return %[[V14]] : tensor<3x3xf64>
// CHECK-NEXT: }

// CHECK-LABEL: func.func @gathered(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x8xf64>, %[[V1:[^ ,:)]+]]: tensor<8x8x2xi64>) -> tensor<3x3xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<8x8xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]]:2 = stablehlo.while(%[[V7:[^ ,:)]+]] = %[[V3]], %[[V8:[^ ,:)]+]] = %[[V2]]) : tensor<i64>, tensor<8x8xf64>
// CHECK-NEXT: cond {
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.compare LT, %[[V7]], %[[V5]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT: stablehlo.return %[[V9]] : tensor<i1>
// CHECK-NEXT: } do {
// CHECK-NEXT: %[[V9]] = "stablehlo.gather"(%[[V8]], %[[V1]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1>}> : (tensor<8x8xf64>, tensor<8x8x2xi64>) -> tensor<8x8xf64>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = stablehlo.add %[[V9]], %[[V0]] : tensor<8x8xf64>
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = stablehlo.add %[[V7]], %[[V4]] : tensor<i64>
// CHECK-NEXT: stablehlo.return %[[V11]], %[[V10]] : tensor<i64>, tensor<8x8xf64>
// CHECK-NEXT: }
// CHECK-NEXT: %[[V12:[^ ,:)]+]] = stablehlo.slice %[[V6]]#1 [0:3, 0:3] : (tensor<8x8xf64>) -> tensor<3x3xf64>
// CHECK-NEXT: return %[[V12]] : tensor<3x3xf64>
// CHECK-NEXT: }

// CHECK-LABEL: func.func @whole(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x8xf64>) -> (tensor<3x3xf64>, tensor<8x8xf64>) {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<8x8xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]]:2 = stablehlo.while(%[[V6:[^ ,:)]+]] = %[[V2]], %[[V7:[^ ,:)]+]] = %[[V1]]) : tensor<i64>, tensor<8x8xf64>
// CHECK-NEXT: cond {
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.compare LT, %[[V6]], %[[V4]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT: stablehlo.return %[[V8]] : tensor<i1>
// CHECK-NEXT: } do {
// CHECK-NEXT: %[[V8]] = stablehlo.add %[[V7]], %[[V0]] : tensor<8x8xf64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.add %[[V6]], %[[V3]] : tensor<i64>
// CHECK-NEXT: stablehlo.return %[[V9]], %[[V8]] : tensor<i64>, tensor<8x8xf64>
// CHECK-NEXT: }
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = stablehlo.slice %[[V5]]#1 [0:3, 0:3] : (tensor<8x8xf64>) -> tensor<3x3xf64>
// CHECK-NEXT: return %[[V10]], %[[V5]]#1 : tensor<3x3xf64>, tensor<8x8xf64>
// CHECK-NEXT: }

// CHECK-LABEL: func.func @feeds_other(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<8x8xf64>) -> (tensor<3x3xf64>, tensor<8x8xf64>) {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<8x8xf64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]]:3 = stablehlo.while(%[[V6:[^ ,:)]+]] = %[[V2]], %[[V7:[^ ,:)]+]] = %[[V1]], %[[V8:[^ ,:)]+]] = %[[V1]]) : tensor<i64>, tensor<8x8xf64>, tensor<8x8xf64>
// CHECK-NEXT: cond {
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.compare LT, %[[V6]], %[[V4]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT: stablehlo.return %[[V9]] : tensor<i1>
// CHECK-NEXT: } do {
// CHECK-NEXT: %[[V9]] = stablehlo.add %[[V7]], %[[V0]] : tensor<8x8xf64>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = stablehlo.add %[[V8]], %[[V7]] : tensor<8x8xf64>
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = stablehlo.add %[[V6]], %[[V3]] : tensor<i64>
// CHECK-NEXT: stablehlo.return %[[V11]], %[[V9]], %[[V10]] : tensor<i64>, tensor<8x8xf64>, tensor<8x8xf64>
// CHECK-NEXT: }
// CHECK-NEXT: %[[V12:[^ ,:)]+]] = stablehlo.slice %[[V5]]#1 [0:3, 0:3] : (tensor<8x8xf64>) -> tensor<3x3xf64>
// CHECK-NEXT: return %[[V12]], %[[V5]]#2 : tensor<3x3xf64>, tensor<8x8xf64>
// CHECK-NEXT: }
