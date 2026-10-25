// RUN: enzymexlamlir-opt %s --enzyme-hlo-element-major --canonicalize | FileCheck %s

// The element dimension (8, the leading dimension of the largest argument
// view) leads every tensor: the stack of two broadcasts of a view is read
// element-major and reduced along its stack dimension, now dimension 1.
func.func @stack(%x: tensor<48xf64>, %w: tensor<2xf64>) -> tensor<8x6xf64> {
  %v = stablehlo.reshape %x : (tensor<48xf64>) -> tensor<8x6xf64>
  %a = stablehlo.broadcast_in_dim %v, dims = [1, 2] : (tensor<8x6xf64>) -> tensor<2x8x6xf64>
  %b = stablehlo.broadcast_in_dim %w, dims = [0] : (tensor<2xf64>) -> tensor<2x8x6xf64>
  %m = stablehlo.multiply %a, %b : tensor<2x8x6xf64>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %r = stablehlo.reduce(%m init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<2x8x6xf64>, tensor<f64>) -> tensor<8x6xf64>
  return %r : tensor<8x6xf64>
}

// CHECK-LABEL: func.func @stack(%arg0: tensor<48xf64>, %arg1: tensor<2xf64>) -> tensor<8x6xf64> {
// CHECK-NEXT: %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %0 = stablehlo.reshape %arg0 : (tensor<48xf64>) -> tensor<8x6xf64>
// CHECK-NEXT: %1 = stablehlo.broadcast_in_dim %0, dims = [0, 2] : (tensor<8x6xf64>) -> tensor<8x2x6xf64>
// CHECK-NEXT: %2 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2xf64>) -> tensor<8x2x6xf64>
// CHECK-NEXT: %3 = stablehlo.multiply %1, %2 : tensor<8x2x6xf64>
// CHECK-NEXT: %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x2x6xf64>, tensor<f64>) -> tensor<8x6xf64>
// CHECK-NEXT: return %4 : tensor<8x6xf64>
// CHECK-NEXT: }

// Transposes, slices, concatenates, reshapes that keep the element dimension
// a dimension of its own, splat constants and barriers are rewritten; the
// dimensions after the element dimension keep their order.
func.func @movement(%x: tensor<96xf64>) -> tensor<3x8x4xf64> {
  %v = stablehlo.reshape %x : (tensor<96xf64>) -> tensor<8x3x4xf64>
  %t = stablehlo.transpose %v, dims = [1, 0, 2] : (tensor<8x3x4xf64>) -> tensor<3x8x4xf64>
  %s = stablehlo.slice %t [0:2, 0:8, 0:4] : (tensor<3x8x4xf64>) -> tensor<2x8x4xf64>
  %cst = stablehlo.constant dense<1.000000e+00> : tensor<1x8x4xf64>
  %c = stablehlo.concatenate %s, %cst, dim = 0 : (tensor<2x8x4xf64>, tensor<1x8x4xf64>) -> tensor<3x8x4xf64>
  %r = stablehlo.reshape %c : (tensor<3x8x4xf64>) -> tensor<3x1x8x4xf64>
  %o = stablehlo.optimization_barrier %r : tensor<3x1x8x4xf64>
  %u = stablehlo.reshape %o : (tensor<3x1x8x4xf64>) -> tensor<3x8x4xf64>
  return %u : tensor<3x8x4xf64>
}

// CHECK-LABEL: func.func @movement(%arg0: tensor<96xf64>) -> tensor<3x8x4xf64> {
// CHECK-NEXT: %cst = stablehlo.constant dense<1.000000e+00> : tensor<8x1x4xf64>
// CHECK-NEXT: %0 = stablehlo.reshape %arg0 : (tensor<96xf64>) -> tensor<8x3x4xf64>
// CHECK-NEXT: %1 = stablehlo.slice %0 [0:8, 0:2, 0:4] : (tensor<8x3x4xf64>) -> tensor<8x2x4xf64>
// CHECK-NEXT: %2 = stablehlo.concatenate %1, %cst, dim = 1 : (tensor<8x2x4xf64>, tensor<8x1x4xf64>) -> tensor<8x3x4xf64>
// CHECK-NEXT: %3 = stablehlo.reshape %2 : (tensor<8x3x4xf64>) -> tensor<8x3x1x4xf64>
// CHECK-NEXT: %4 = stablehlo.optimization_barrier %3 : tensor<8x3x1x4xf64>
// CHECK-NEXT: %5 = stablehlo.reshape %4 : (tensor<8x3x1x4xf64>) -> tensor<8x3x4xf64>
// CHECK-NEXT: %6 = stablehlo.transpose %5, dims = [1, 0, 2] : (tensor<8x3x4xf64>) -> tensor<3x8x4xf64>
// CHECK-NEXT: return %6 : tensor<3x8x4xf64>
// CHECK-NEXT: }

// An op the pass does not rewrite (a reshape merging the element dimension,
// a gather) reads the rewritten values transposed back, and its result stays
// as it was; a reduce over the element dimension stays too.
func.func @boundary(%x: tensor<48xf64>, %i: tensor<2x8x1xi64>) -> (tensor<2x48xf64>, tensor<2x8xf64>, tensor<2x6xf64>) {
  %v = stablehlo.reshape %x : (tensor<48xf64>) -> tensor<8x6xf64>
  %a = stablehlo.broadcast_in_dim %v, dims = [1, 2] : (tensor<8x6xf64>) -> tensor<2x8x6xf64>
  %n = stablehlo.negate %a : tensor<2x8x6xf64>
  %m = stablehlo.reshape %n : (tensor<2x8x6xf64>) -> tensor<2x48xf64>
  %g = "stablehlo.gather"(%n, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [2], operand_batching_dims = [0, 1], start_indices_batching_dims = [0, 1], start_index_map = [2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<2x8x6xf64>, tensor<2x8x1xi64>) -> tensor<2x8xf64>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %r = stablehlo.reduce(%n init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<2x8x6xf64>, tensor<f64>) -> tensor<2x6xf64>
  return %m, %g, %r : tensor<2x48xf64>, tensor<2x8xf64>, tensor<2x6xf64>
}

// CHECK-LABEL: func.func @boundary(%arg0: tensor<48xf64>, %arg1: tensor<2x8x1xi64>) -> (tensor<2x48xf64>, tensor<2x8xf64>, tensor<2x6xf64>) {
// CHECK-NEXT: %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %0 = stablehlo.reshape %arg0 : (tensor<48xf64>) -> tensor<8x6xf64>
// CHECK-NEXT: %1 = stablehlo.broadcast_in_dim %0, dims = [0, 2] : (tensor<8x6xf64>) -> tensor<8x2x6xf64>
// CHECK-NEXT: %2 = stablehlo.negate %1 : tensor<8x2x6xf64>
// CHECK-NEXT: %3 = stablehlo.transpose %2, dims = [1, 0, 2] : (tensor<8x2x6xf64>) -> tensor<2x8x6xf64>
// CHECK-NEXT: %4 = stablehlo.reshape %3 : (tensor<2x8x6xf64>) -> tensor<2x48xf64>
// CHECK-NEXT: %5 = "stablehlo.gather"(%3, %arg1) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [2], operand_batching_dims = [0, 1], start_indices_batching_dims = [0, 1], start_index_map = [2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<2x8x6xf64>, tensor<2x8x1xi64>) -> tensor<2x8xf64>
// CHECK-NEXT: %6 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<2x8x6xf64>, tensor<f64>) -> tensor<2x6xf64>
// CHECK-NEXT: return %4, %5, %6 : tensor<2x48xf64>, tensor<2x8xf64>, tensor<2x6xf64>
// CHECK-NEXT: }

// A tensor whose element dimension is its last one is left as it is (the
// minor dimension never moves), as is a function without argument views.
func.func @minor(%x: tensor<48xf64>) -> tensor<6x8xf64> {
  %v = stablehlo.reshape %x : (tensor<48xf64>) -> tensor<8x6xf64>
  %t = stablehlo.transpose %v, dims = [1, 0] : (tensor<8x6xf64>) -> tensor<6x8xf64>
  %n = stablehlo.negate %t : tensor<6x8xf64>
  return %n : tensor<6x8xf64>
}

// CHECK-LABEL: func.func @minor(%arg0: tensor<48xf64>) -> tensor<6x8xf64> {
// CHECK-NEXT: %0 = stablehlo.reshape %arg0 : (tensor<48xf64>) -> tensor<8x6xf64>
// CHECK-NEXT: %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<8x6xf64>) -> tensor<6x8xf64>
// CHECK-NEXT: %2 = stablehlo.negate %1 : tensor<6x8xf64>
// CHECK-NEXT: return %2 : tensor<6x8xf64>
// CHECK-NEXT: }

// The layout follows the values, not the sizes: a slice along the element
// dimension no longer has its size, and still keeps it leading. The element
// dimension is, of the dimensions of the largest view, the size leading the
// views of the most arguments (two basis arguments viewed as 2x2 and 2x2x1
// do not make 2 the element dimension: 2 is no dimension of the 8x6 view).
func.func @sliced(%x: tensor<48xf64>, %b: tensor<4xf64>, %g: tensor<4xf64>) -> tensor<3x6xf64> {
  %v = stablehlo.reshape %x : (tensor<48xf64>) -> tensor<8x6xf64>
  %bb = stablehlo.reshape %b : (tensor<4xf64>) -> tensor<2x2xf64>
  %bg = stablehlo.reshape %g : (tensor<4xf64>) -> tensor<2x2xf64>
  %bb1 = stablehlo.reshape %b : (tensor<4xf64>) -> tensor<2x2x1xf64>
  %bg1 = stablehlo.reshape %g : (tensor<4xf64>) -> tensor<2x2x1xf64>
  %w = stablehlo.slice %bb [0:1, 0:1] : (tensor<2x2xf64>) -> tensor<1x1xf64>
  %a = stablehlo.broadcast_in_dim %v, dims = [1, 2] : (tensor<8x6xf64>) -> tensor<2x8x6xf64>
  %c = stablehlo.broadcast_in_dim %w, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<2x8x6xf64>
  %m = stablehlo.multiply %a, %c : tensor<2x8x6xf64>
  %s = stablehlo.slice %m [0:1, 2:5, 0:6] : (tensor<2x8x6xf64>) -> tensor<1x3x6xf64>
  %r = stablehlo.reshape %s : (tensor<1x3x6xf64>) -> tensor<3x6xf64>
  %z = stablehlo.add %r, %r : tensor<3x6xf64>
  return %z : tensor<3x6xf64>
}

// CHECK-LABEL: func.func @sliced(%arg0: tensor<48xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>) -> tensor<3x6xf64> {
// CHECK-NEXT: %0 = stablehlo.reshape %arg0 : (tensor<48xf64>) -> tensor<8x6xf64>
// CHECK-NEXT: %1 = stablehlo.reshape %arg1 : (tensor<4xf64>) -> tensor<2x2xf64>
// CHECK-NEXT: %2 = stablehlo.slice %1 [0:1, 0:1] : (tensor<2x2xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %3 = stablehlo.broadcast_in_dim %0, dims = [0, 2] : (tensor<8x6xf64>) -> tensor<8x2x6xf64>
// CHECK-NEXT: %4 = stablehlo.broadcast_in_dim %2, dims = [1, 0] : (tensor<1x1xf64>) -> tensor<8x2x6xf64>
// CHECK-NEXT: %5 = stablehlo.multiply %3, %4 : tensor<8x2x6xf64>
// CHECK-NEXT: %6 = stablehlo.slice %5 [2:5, 0:1, 0:6] : (tensor<8x2x6xf64>) -> tensor<3x1x6xf64>
// CHECK-NEXT: %7 = stablehlo.reshape %6 : (tensor<3x1x6xf64>) -> tensor<3x6xf64>
// CHECK-NEXT: %8 = stablehlo.add %7, %7 : tensor<3x6xf64>
// CHECK-NEXT: return %8 : tensor<3x6xf64>
// CHECK-NEXT: }

// A reduce whose input already has the element dimension leading (computed
// from rewritten values, here the reduce of a stack) keeps the identity
// layout of its own rank; its users read it as it is.
func.func @leading_reduce(%x: tensor<480xf64>, %w: tensor<2xf64>, %b: tensor<5x6xf64>) -> tensor<8x5x1xf64> {
  %v = stablehlo.reshape %x : (tensor<480xf64>) -> tensor<8x6x10xf64>
  %s0 = stablehlo.slice %v [0:8, 0:6, 0:5] : (tensor<8x6x10xf64>) -> tensor<8x6x5xf64>
  %a = stablehlo.broadcast_in_dim %s0, dims = [1, 2, 3] : (tensor<8x6x5xf64>) -> tensor<2x8x6x5xf64>
  %k = stablehlo.broadcast_in_dim %w, dims = [0] : (tensor<2xf64>) -> tensor<2x8x6x5xf64>
  %p = stablehlo.multiply %a, %k : tensor<2x8x6x5xf64>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %q = stablehlo.reduce(%p init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<2x8x6x5xf64>, tensor<f64>) -> tensor<8x6x5xf64>
  %e = stablehlo.broadcast_in_dim %q, dims = [0, 1, 2] : (tensor<8x6x5xf64>) -> tensor<8x6x5x6x1xf64>
  %bb = stablehlo.broadcast_in_dim %b, dims = [2, 1] : (tensor<5x6xf64>) -> tensor<8x6x5x6x1xf64>
  %m = stablehlo.multiply %bb, %e : tensor<8x6x5x6x1xf64>
  %r = stablehlo.reduce(%m init: %cst) applies stablehlo.add across dimensions = [1, 4] : (tensor<8x6x5x6x1xf64>, tensor<f64>) -> tensor<8x5x6xf64>
  %s = stablehlo.slice %r [0:8, 0:5, 0:1] : (tensor<8x5x6xf64>) -> tensor<8x5x1xf64>
  return %s : tensor<8x5x1xf64>
}

// CHECK-LABEL: func.func @leading_reduce(%arg0: tensor<480xf64>, %arg1: tensor<2xf64>, %arg2: tensor<5x6xf64>) -> tensor<8x5x1xf64> {
// CHECK-NEXT: %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %0 = stablehlo.reshape %arg0 : (tensor<480xf64>) -> tensor<8x6x10xf64>
// CHECK-NEXT: %1 = stablehlo.slice %0 [0:8, 0:6, 0:5] : (tensor<8x6x10xf64>) -> tensor<8x6x5xf64>
// CHECK-NEXT: %2 = stablehlo.broadcast_in_dim %1, dims = [0, 2, 3] : (tensor<8x6x5xf64>) -> tensor<8x2x6x5xf64>
// CHECK-NEXT: %3 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2xf64>) -> tensor<8x2x6x5xf64>
// CHECK-NEXT: %4 = stablehlo.multiply %2, %3 : tensor<8x2x6x5xf64>
// CHECK-NEXT: %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x2x6x5xf64>, tensor<f64>) -> tensor<8x6x5xf64>
// CHECK-NEXT: %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<8x6x5xf64>) -> tensor<8x6x5x6x1xf64>
// CHECK-NEXT: %7 = stablehlo.broadcast_in_dim %arg2, dims = [2, 1] : (tensor<5x6xf64>) -> tensor<8x6x5x6x1xf64>
// CHECK-NEXT: %8 = stablehlo.multiply %7, %6 : tensor<8x6x5x6x1xf64>
// CHECK-NEXT: %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [1, 4] : (tensor<8x6x5x6x1xf64>, tensor<f64>) -> tensor<8x5x6xf64>
// CHECK-NEXT: %10 = stablehlo.slice %9 [0:8, 0:5, 0:1] : (tensor<8x5x6xf64>) -> tensor<8x5x1xf64>
// CHECK-NEXT: return %10 : tensor<8x5x1xf64>
// CHECK-NEXT: }
