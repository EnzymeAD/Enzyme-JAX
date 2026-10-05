// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=parallel_while_to_batched_scatter" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// A dot product: the carried sum is a reduction.
func.func @dot(%x: tensor<8xf64>, %y: tensor<8xf64>) -> tensor<f64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %init = stablehlo.constant dense<1.5> : tensor<f64>
  %r:4 = stablehlo.while(%i = %c0, %sum = %init, %a = %x, %b = %y) : tensor<i64>, tensor<f64>, tensor<8xf64>, tensor<8xf64> attributes {enzymexla.parallel}
  cond {
    %p = stablehlo.compare LT, %i, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %p : tensor<i1>
  } do {
    %xi = stablehlo.dynamic_slice %a, %i, sizes = [1] : (tensor<8xf64>, tensor<i64>) -> tensor<1xf64>
    %yi = stablehlo.dynamic_slice %b, %i, sizes = [1] : (tensor<8xf64>, tensor<i64>) -> tensor<1xf64>
    %xs = stablehlo.reshape %xi : (tensor<1xf64>) -> tensor<f64>
    %ys = stablehlo.reshape %yi : (tensor<1xf64>) -> tensor<f64>
    %m = stablehlo.multiply %xs, %ys : tensor<f64>
    %s = stablehlo.add %sum, %m : tensor<f64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %s, %a, %b : tensor<i64>, tensor<f64>, tensor<8xf64>, tensor<8xf64>
  }
  return %r#1 : tensor<f64>
}

// CHECK:  func.func @dot(%arg0: tensor<8xf64>, %arg1: tensor<8xf64>) -> tensor<f64> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<1.500000e+00> : tensor<f64>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %0 = stablehlo.reshape %arg0 : (tensor<8xf64>) -> tensor<8x1xf64>
// CHECK-NEXT:   %1 = stablehlo.reshape %arg1 : (tensor<8xf64>) -> tensor<8x1xf64>
// CHECK-NEXT:   %2 = stablehlo.reshape %0 : (tensor<8x1xf64>) -> tensor<8xf64>
// CHECK-NEXT:   %3 = stablehlo.reshape %1 : (tensor<8x1xf64>) -> tensor<8xf64>
// CHECK-NEXT:   %4 = stablehlo.multiply %2, %3 : tensor<8xf64>
// CHECK-NEXT:   %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<8xf64>
// CHECK-NEXT:   %6 = stablehlo.add %5, %4 : tensor<8xf64>
// CHECK-NEXT:   %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<8xf64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:   %8 = stablehlo.add %cst, %7 : tensor<f64>
// CHECK-NEXT:   return %8 : tensor<f64>
// CHECK-NEXT: }

// -----

// A max over the iterations, of rows of a buffer.
func.func @row_max(%x: tensor<4x8xf32>) -> tensor<8xf32> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %init = stablehlo.constant dense<0.0> : tensor<8xf32>
  %r:3 = stablehlo.while(%i = %c0, %acc = %init, %a = %x) : tensor<i64>, tensor<8xf32>, tensor<4x8xf32> attributes {enzymexla.parallel}
  cond {
    %p = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %p : tensor<i1>
  } do {
    %row = stablehlo.dynamic_slice %a, %i, %c0, sizes = [1, 8] : (tensor<4x8xf32>, tensor<i64>, tensor<i64>) -> tensor<1x8xf32>
    %rs = stablehlo.reshape %row : (tensor<1x8xf32>) -> tensor<8xf32>
    %m = stablehlo.maximum %rs, %acc : tensor<8xf32>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %m, %a : tensor<i64>, tensor<8xf32>, tensor<4x8xf32>
  }
  return %r#1 : tensor<8xf32>
}

// CHECK:  func.func @row_max(%arg0: tensor<4x8xf32>) -> tensor<8xf32> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<8xf32>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-NEXT:   %0 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
// CHECK-NEXT:   %1 = stablehlo.reshape %arg0 : (tensor<4x8xf32>) -> tensor<4x1x8xf32>
// CHECK-NEXT:   %2 = stablehlo.reshape %1 : (tensor<4x1x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %3 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %4 = stablehlo.maximum %2, %3 : tensor<4x8xf32>
// CHECK-NEXT:   %5 = stablehlo.reduce(%4 init: %cst_0) applies stablehlo.maximum across dimensions = [0] : (tensor<4x8xf32>, tensor<f32>) -> tensor<8xf32>
// CHECK-NEXT:   %6 = stablehlo.maximum %cst, %5 : tensor<8xf32>
// CHECK-NEXT:   return %6 : tensor<8xf32>
// CHECK-NEXT: }

// -----

// What every iteration adds is the same: it is added once per iteration.
func.func @invariant_term(%v: tensor<i32>) -> tensor<i32> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c5 = stablehlo.constant dense<5> : tensor<i64>
  %r:2 = stablehlo.while(%i = %c0, %acc = %v) : tensor<i64>, tensor<i32> attributes {enzymexla.parallel}
  cond {
    %p = stablehlo.compare LT, %i, %c5 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %p : tensor<i1>
  } do {
    %s = stablehlo.add %acc, %v : tensor<i32>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %s : tensor<i64>, tensor<i32>
  }
  return %r#1 : tensor<i32>
}

// CHECK:  func.func @invariant_term(%arg0: tensor<i32>) -> tensor<i32> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:   %0 = stablehlo.add %c, %arg0 : tensor<i32>
// CHECK-NEXT:   %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<5xi32>
// CHECK-NEXT:   %2 = stablehlo.reduce(%1 init: %c) applies stablehlo.add across dimensions = [0] : (tensor<5xi32>, tensor<i32>) -> tensor<i32>
// CHECK-NEXT:   %3 = stablehlo.add %arg0, %2 : tensor<i32>
// CHECK-NEXT:   return %3 : tensor<i32>
// CHECK-NEXT: }

// -----

// The partial sum is read by the iteration too: not a reduction.
func.func @partial_sum_read(%x: tensor<8xf64>) -> tensor<8xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %r:3 = stablehlo.while(%i = %c0, %sum = %z, %a = %x) : tensor<i64>, tensor<f64>, tensor<8xf64> attributes {enzymexla.parallel}
  cond {
    %p = stablehlo.compare LT, %i, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %p : tensor<i1>
  } do {
    %xi = stablehlo.dynamic_slice %a, %i, sizes = [1] : (tensor<8xf64>, tensor<i64>) -> tensor<1xf64>
    %xs = stablehlo.reshape %xi : (tensor<1xf64>) -> tensor<f64>
    %s = stablehlo.add %sum, %xs : tensor<f64>
    %u = stablehlo.reshape %sum : (tensor<f64>) -> tensor<1xf64>
    %w = stablehlo.dynamic_update_slice %a, %u, %i : (tensor<8xf64>, tensor<1xf64>, tensor<i64>) -> tensor<8xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %s, %w : tensor<i64>, tensor<f64>, tensor<8xf64>
  }
  return %r#2 : tensor<8xf64>
}

// CHECK:  func.func @partial_sum_read(%arg0: tensor<8xf64>) -> tensor<8xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %0:3 = stablehlo.while(%iterArg = %c, %iterArg_2 = %cst, %iterArg_3 = %arg0) : tensor<i64>, tensor<f64>, tensor<8xf64> attributes {enzymexla.parallel}
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.dynamic_slice %iterArg_3, %iterArg, sizes = [1] : (tensor<8xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:     %2 = stablehlo.reshape %1 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:     %3 = stablehlo.add %iterArg_2, %2 : tensor<f64>
// CHECK-NEXT:     %4 = stablehlo.reshape %iterArg_2 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:     %5 = stablehlo.dynamic_update_slice %iterArg_3, %4, %iterArg : (tensor<8xf64>, tensor<1xf64>, tensor<i64>) -> tensor<8xf64>
// CHECK-NEXT:     %6 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %6, %3, %5 : tensor<i64>, tensor<f64>, tensor<8xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#2 : tensor<8xf64>
// CHECK-NEXT: }
