// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=parallel_while_to_batched_scatter" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// A row of the loop times a matrix the same every iteration: the rows of
// every iteration are one operand, the iterations a free dimension of it, and
// the matrix is used as it is rather than broadcast to every iteration.
func.func @lhs_varies(%v: tensor<6x5xf64>, %m: tensor<5x7xf64>, %out: tensor<6x7xf64>) -> tensor<6x7xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c6 = stablehlo.constant dense<6> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %o = %out) : tensor<i64>, tensor<6x7xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %r2 = stablehlo.dynamic_slice %v, %i, %c0, sizes = [1, 5] : (tensor<6x5xf64>, tensor<i64>, tensor<i64>) -> tensor<1x5xf64>
    %r = stablehlo.reshape %r2 : (tensor<1x5xf64>) -> tensor<5xf64>
    %d = stablehlo.dot_general %r, %m, contracting_dims = [0] x [0] : (tensor<5xf64>, tensor<5x7xf64>) -> tensor<7xf64>
    %d2 = stablehlo.reshape %d : (tensor<7xf64>) -> tensor<1x7xf64>
    %o1 = stablehlo.dynamic_update_slice %o, %d2, %i, %c0 : (tensor<6x7xf64>, tensor<1x7xf64>, tensor<i64>, tensor<i64>) -> tensor<6x7xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %o1 : tensor<i64>, tensor<6x7xf64>
  }
  return %0#1 : tensor<6x7xf64>
}

// CHECK:  func.func @lhs_varies(%arg0: tensor<6x5xf64>, %arg1: tensor<5x7xf64>, %arg2: tensor<6x7xf64>) -> tensor<6x7xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<5> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<6xi64>
// CHECK-NEXT:   %1 = stablehlo.reshape %arg0 : (tensor<6x5xf64>) -> tensor<6x1x5xf64>
// CHECK-NEXT:   %2 = stablehlo.reshape %1 : (tensor<6x1x5xf64>) -> tensor<6x5xf64>
// CHECK-NEXT:   %3 = stablehlo.dot_general %2, %arg1, contracting_dims = [1] x [0] : (tensor<6x5xf64>, tensor<5x7xf64>) -> tensor<6x7xf64>
// CHECK-NEXT:   %4 = stablehlo.reshape %3 : (tensor<6x7xf64>) -> tensor<6x1x7xf64>
// CHECK-NEXT:   %5 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i64>) -> tensor<6xi64>
// CHECK-NEXT:   %6 = stablehlo.clamp %c_0, %0, %c : (tensor<i64>, tensor<6xi64>, tensor<i64>) -> tensor<6xi64>
// CHECK-NEXT:   %7 = stablehlo.reshape %6 : (tensor<6xi64>) -> tensor<6x1xi64>
// CHECK-NEXT:   %8 = stablehlo.clamp %c_0, %5, %c_0 : (tensor<i64>, tensor<6xi64>, tensor<i64>) -> tensor<6xi64>
// CHECK-NEXT:   %9 = stablehlo.reshape %8 : (tensor<6xi64>) -> tensor<6x1xi64>
// CHECK-NEXT:   %10 = stablehlo.concatenate %7, %9, dim = 1 : (tensor<6x1xi64>, tensor<6x1xi64>) -> tensor<6x2xi64>
// CHECK-NEXT:   %11 = "stablehlo.scatter"(%arg2, %10, %4) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg4 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<6x7xf64>, tensor<6x2xi64>, tensor<6x1x7xf64>) -> tensor<6x7xf64>
// CHECK-NEXT:   return %11 : tensor<6x7xf64>
// CHECK-NEXT: }


// -----

// The same with the matrix on the left: the iterations are the last free
// dimension of the result, transposed to the front.
func.func @rhs_varies(%v: tensor<6x5xf64>, %m: tensor<7x5xf64>, %out: tensor<6x7xf64>) -> tensor<6x7xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c6 = stablehlo.constant dense<6> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %o = %out) : tensor<i64>, tensor<6x7xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %r2 = stablehlo.dynamic_slice %v, %i, %c0, sizes = [1, 5] : (tensor<6x5xf64>, tensor<i64>, tensor<i64>) -> tensor<1x5xf64>
    %r = stablehlo.reshape %r2 : (tensor<1x5xf64>) -> tensor<5xf64>
    %d = stablehlo.dot_general %m, %r, contracting_dims = [1] x [0] : (tensor<7x5xf64>, tensor<5xf64>) -> tensor<7xf64>
    %d2 = stablehlo.reshape %d : (tensor<7xf64>) -> tensor<1x7xf64>
    %o1 = stablehlo.dynamic_update_slice %o, %d2, %i, %c0 : (tensor<6x7xf64>, tensor<1x7xf64>, tensor<i64>, tensor<i64>) -> tensor<6x7xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %o1 : tensor<i64>, tensor<6x7xf64>
  }
  return %0#1 : tensor<6x7xf64>
}

// CHECK:  func.func @rhs_varies(%arg0: tensor<6x5xf64>, %arg1: tensor<7x5xf64>, %arg2: tensor<6x7xf64>) -> tensor<6x7xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<5> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<6xi64>
// CHECK-NEXT:   %1 = stablehlo.reshape %arg0 : (tensor<6x5xf64>) -> tensor<6x1x5xf64>
// CHECK-NEXT:   %2 = stablehlo.reshape %1 : (tensor<6x1x5xf64>) -> tensor<6x5xf64>
// CHECK-NEXT:   %3 = stablehlo.dot_general %arg1, %2, contracting_dims = [1] x [1] : (tensor<7x5xf64>, tensor<6x5xf64>) -> tensor<7x6xf64>
// CHECK-NEXT:   %4 = stablehlo.transpose %3, dims = [1, 0] : (tensor<7x6xf64>) -> tensor<6x7xf64>
// CHECK-NEXT:   %5 = stablehlo.reshape %4 : (tensor<6x7xf64>) -> tensor<6x1x7xf64>
// CHECK-NEXT:   %6 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i64>) -> tensor<6xi64>
// CHECK-NEXT:   %7 = stablehlo.clamp %c_0, %0, %c : (tensor<i64>, tensor<6xi64>, tensor<i64>) -> tensor<6xi64>
// CHECK-NEXT:   %8 = stablehlo.reshape %7 : (tensor<6xi64>) -> tensor<6x1xi64>
// CHECK-NEXT:   %9 = stablehlo.clamp %c_0, %6, %c_0 : (tensor<i64>, tensor<6xi64>, tensor<i64>) -> tensor<6xi64>
// CHECK-NEXT:   %10 = stablehlo.reshape %9 : (tensor<6xi64>) -> tensor<6x1xi64>
// CHECK-NEXT:   %11 = stablehlo.concatenate %8, %10, dim = 1 : (tensor<6x1xi64>, tensor<6x1xi64>) -> tensor<6x2xi64>
// CHECK-NEXT:   %12 = "stablehlo.scatter"(%arg2, %11, %5) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg4 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<6x7xf64>, tensor<6x2xi64>, tensor<6x1x7xf64>) -> tensor<6x7xf64>
// CHECK-NEXT:   return %12 : tensor<6x7xf64>
// CHECK-NEXT: }


// -----

// Batching and contracting dimensions on both sides, as the mfem kernels'
// sum-factorized contractions have: the varying side's dimensions shift by
// one, and the iterations come after the shared side's free dimensions.
func.func @batched_dims(%v: tensor<4x3x5x2xf64>, %m: tensor<3x5x2x6xf64>, %out: tensor<4x3x2x6xf64>) -> tensor<4x3x2x6xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %o = %out) : tensor<i64>, tensor<4x3x2x6xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %r4 = stablehlo.dynamic_slice %v, %i, %c0, %c0, %c0, sizes = [1, 3, 5, 2] : (tensor<4x3x5x2xf64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x3x5x2xf64>
    %r = stablehlo.reshape %r4 : (tensor<1x3x5x2xf64>) -> tensor<3x5x2xf64>
    %d = stablehlo.dot_general %m, %r, batching_dims = [0, 2] x [0, 2], contracting_dims = [1] x [1] : (tensor<3x5x2x6xf64>, tensor<3x5x2xf64>) -> tensor<3x2x6xf64>
    %d2 = stablehlo.reshape %d : (tensor<3x2x6xf64>) -> tensor<1x3x2x6xf64>
    %o1 = stablehlo.dynamic_update_slice %o, %d2, %i, %c0, %c0, %c0 : (tensor<4x3x2x6xf64>, tensor<1x3x2x6xf64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<4x3x2x6xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %o1 : tensor<i64>, tensor<4x3x2x6xf64>
  }
  return %0#1 : tensor<4x3x2x6xf64>
}

// CHECK:  func.func @batched_dims(%arg0: tensor<4x3x5x2xf64>, %arg1: tensor<3x5x2x6xf64>, %arg2: tensor<4x3x2x6xf64>) -> tensor<4x3x2x6xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK-NEXT:   %1 = stablehlo.reshape %arg0 : (tensor<4x3x5x2xf64>) -> tensor<4x1x3x5x2xf64>
// CHECK-NEXT:   %2 = stablehlo.reshape %1 : (tensor<4x1x3x5x2xf64>) -> tensor<4x3x5x2xf64>
// CHECK-NEXT:   %3 = stablehlo.dot_general %arg1, %2, batching_dims = [0, 2] x [1, 3], contracting_dims = [1] x [2] : (tensor<3x5x2x6xf64>, tensor<4x3x5x2xf64>) -> tensor<3x2x6x4xf64>
// CHECK-NEXT:   %4 = stablehlo.transpose %3, dims = [3, 0, 1, 2] : (tensor<3x2x6x4xf64>) -> tensor<4x3x2x6xf64>
// CHECK-NEXT:   %5 = stablehlo.reshape %4 : (tensor<4x3x2x6xf64>) -> tensor<4x1x3x2x6xf64>
// CHECK-NEXT:   %6 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:   %7 = stablehlo.clamp %c_0, %0, %c : (tensor<i64>, tensor<4xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:   %8 = stablehlo.reshape %7 : (tensor<4xi64>) -> tensor<4x1xi64>
// CHECK-NEXT:   %9 = stablehlo.clamp %c_0, %6, %c_0 : (tensor<i64>, tensor<4xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:   %10 = stablehlo.reshape %9 : (tensor<4xi64>) -> tensor<4x1xi64>
// CHECK-NEXT:   %11 = stablehlo.clamp %c_0, %6, %c_0 : (tensor<i64>, tensor<4xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:   %12 = stablehlo.reshape %11 : (tensor<4xi64>) -> tensor<4x1xi64>
// CHECK-NEXT:   %13 = stablehlo.clamp %c_0, %6, %c_0 : (tensor<i64>, tensor<4xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:   %14 = stablehlo.reshape %13 : (tensor<4xi64>) -> tensor<4x1xi64>
// CHECK-NEXT:   %15 = stablehlo.concatenate %8, %10, %12, %14, dim = 1 : (tensor<4x1xi64>, tensor<4x1xi64>, tensor<4x1xi64>, tensor<4x1xi64>) -> tensor<4x4xi64>
// CHECK-NEXT:   %16 = "stablehlo.scatter"(%arg2, %15, %5) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [0, 1, 2, 3], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg4 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<4x3x2x6xf64>, tensor<4x4xi64>, tensor<4x1x3x2x6xf64>) -> tensor<4x3x2x6xf64>
// CHECK-NEXT:   return %16 : tensor<4x3x2x6xf64>
// CHECK-NEXT: }

