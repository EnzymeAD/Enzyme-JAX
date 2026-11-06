// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=concat_operand_barrier" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// The computation behind each operand of a concatenate of three or more
// operands is materialized: a sum read through a slice and a reshape, a
// product read through a broadcast; an argument, a constant and a value
// already behind a barrier are left alone, and a barrier serves every
// operand that reads the same value. A dot_general is a kernel of its own,
// not recomputed in the consumer, so its result is not barriered.
func.func @wide(%a: tensor<64x8xf64>, %b: tensor<64x8xf64>, %c: tensor<64x4xf64>, %w: tensor<8x4xf64>) -> tensor<64x28xf64> {
  %cst = stablehlo.constant dense<1.000000e+00> : tensor<64x4xf64>
  %0 = stablehlo.add %a, %b : tensor<64x8xf64>
  %1 = stablehlo.slice %0 [0:64, 0:4] : (tensor<64x8xf64>) -> tensor<64x4xf64>
  %2 = stablehlo.slice %0 [0:64, 4:8] : (tensor<64x8xf64>) -> tensor<64x4xf64>
  %3 = stablehlo.multiply %c, %c : tensor<64x4xf64>
  %4 = stablehlo.reshape %3 : (tensor<64x4xf64>) -> tensor<64x4x1xf64>
  %5 = stablehlo.reshape %4 : (tensor<64x4x1xf64>) -> tensor<64x4xf64>
  %6 = stablehlo.negate %c : tensor<64x4xf64>
  %7 = stablehlo.optimization_barrier %6 : tensor<64x4xf64>
  %8 = stablehlo.dot_general %a, %w, contracting_dims = [1] x [0] : (tensor<64x8xf64>, tensor<8x4xf64>) -> tensor<64x4xf64>
  %9 = stablehlo.concatenate %1, %2, %5, %c, %cst, %7, %8, dim = 1 : (tensor<64x4xf64>, tensor<64x4xf64>, tensor<64x4xf64>, tensor<64x4xf64>, tensor<64x4xf64>, tensor<64x4xf64>, tensor<64x4xf64>) -> tensor<64x28xf64>
  return %9 : tensor<64x28xf64>
}

// CHECK: func.func @wide(%arg0: tensor<64x8xf64>, %arg1: tensor<64x8xf64>, %arg2: tensor<64x4xf64>, %arg3: tensor<8x4xf64>) -> tensor<64x28xf64> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<1.000000e+00> : tensor<64x4xf64>
// CHECK-NEXT:   %0 = stablehlo.add %arg0, %arg1 : tensor<64x8xf64>
// CHECK-NEXT:   %1 = stablehlo.optimization_barrier %0 : tensor<64x8xf64>
// CHECK-NEXT:   %2 = stablehlo.slice %1 [0:64, 0:4] : (tensor<64x8xf64>) -> tensor<64x4xf64>
// CHECK-NEXT:   %3 = stablehlo.slice %1 [0:64, 4:8] : (tensor<64x8xf64>) -> tensor<64x4xf64>
// CHECK-NEXT:   %4 = stablehlo.multiply %arg2, %arg2 : tensor<64x4xf64>
// CHECK-NEXT:   %5 = stablehlo.optimization_barrier %4 : tensor<64x4xf64>
// CHECK-NEXT:   %6 = stablehlo.reshape %5 : (tensor<64x4xf64>) -> tensor<64x4x1xf64>
// CHECK-NEXT:   %7 = stablehlo.reshape %6 : (tensor<64x4x1xf64>) -> tensor<64x4xf64>
// CHECK-NEXT:   %8 = stablehlo.negate %arg2 : tensor<64x4xf64>
// CHECK-NEXT:   %9 = stablehlo.optimization_barrier %8 : tensor<64x4xf64>
// CHECK-NEXT:   %10 = stablehlo.dot_general %arg0, %arg3, contracting_dims = [1] x [0] : (tensor<64x8xf64>, tensor<8x4xf64>) -> tensor<64x4xf64>
// CHECK-NEXT:   %11 = stablehlo.concatenate %2, %3, %7, %arg2, %cst, %9, %10, dim = 1 : (tensor<64x4xf64>, tensor<64x4xf64>, tensor<64x4xf64>, tensor<64x4xf64>, tensor<64x4xf64>, tensor<64x4xf64>, tensor<64x4xf64>) -> tensor<64x28xf64>
// CHECK-NEXT:   return %11 : tensor<64x28xf64>
// CHECK-NEXT: }

// A concatenate of two operands is not wide.
func.func @narrow(%a: tensor<64x8xf64>, %b: tensor<64x8xf64>) -> tensor<64x16xf64> {
  %0 = stablehlo.add %a, %b : tensor<64x8xf64>
  %1 = stablehlo.multiply %a, %b : tensor<64x8xf64>
  %2 = stablehlo.concatenate %0, %1, dim = 1 : (tensor<64x8xf64>, tensor<64x8xf64>) -> tensor<64x16xf64>
  return %2 : tensor<64x16xf64>
}

// CHECK: func.func @narrow(%arg0: tensor<64x8xf64>, %arg1: tensor<64x8xf64>) -> tensor<64x16xf64> {
// CHECK-NEXT:   %0 = stablehlo.add %arg0, %arg1 : tensor<64x8xf64>
// CHECK-NEXT:   %1 = stablehlo.multiply %arg0, %arg1 : tensor<64x8xf64>
// CHECK-NEXT:   %2 = stablehlo.concatenate %0, %1, dim = 1 : (tensor<64x8xf64>, tensor<64x8xf64>) -> tensor<64x16xf64>
// CHECK-NEXT:   return %2 : tensor<64x16xf64>
// CHECK-NEXT: }

// A gather only reads its operand: the components of a vector a gather reads
// one at a time, concatenated, are not barriered (materialized, they would be
// written out and read back).
func.func @gathers(%x: tensor<300xf64>, %i: tensor<64x1xi64>) -> tensor<64x3xf64> {
  %c1 = stablehlo.constant dense<1> : tensor<64x1xi64>
  %0 = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<300xf64>, tensor<64x1xi64>) -> tensor<64x1xf64>
  %1 = stablehlo.add %i, %c1 : tensor<64x1xi64>
  %2 = "stablehlo.gather"(%x, %1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<300xf64>, tensor<64x1xi64>) -> tensor<64x1xf64>
  %3 = stablehlo.add %1, %c1 : tensor<64x1xi64>
  %4 = "stablehlo.gather"(%x, %3) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<300xf64>, tensor<64x1xi64>) -> tensor<64x1xf64>
  %5 = stablehlo.concatenate %0, %2, %4, dim = 1 : (tensor<64x1xf64>, tensor<64x1xf64>, tensor<64x1xf64>) -> tensor<64x3xf64>
  return %5 : tensor<64x3xf64>
}

// CHECK: func.func @gathers(%arg0: tensor<300xf64>, %arg1: tensor<64x1xi64>) -> tensor<64x3xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<1> : tensor<64x1xi64>
// CHECK-NEXT:   %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<300xf64>, tensor<64x1xi64>) -> tensor<64x1xf64>
// CHECK-NEXT:   %1 = stablehlo.add %arg1, %c : tensor<64x1xi64>
// CHECK-NEXT:   %2 = "stablehlo.gather"(%arg0, %1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<300xf64>, tensor<64x1xi64>) -> tensor<64x1xf64>
// CHECK-NEXT:   %3 = stablehlo.add %1, %c : tensor<64x1xi64>
// CHECK-NEXT:   %4 = "stablehlo.gather"(%arg0, %3) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<300xf64>, tensor<64x1xi64>) -> tensor<64x1xf64>
// CHECK-NEXT:   %5 = stablehlo.concatenate %0, %2, %4, dim = 1 : (tensor<64x1xf64>, tensor<64x1xf64>, tensor<64x1xf64>) -> tensor<64x3xf64>
// CHECK-NEXT:   return %5 : tensor<64x3xf64>
// CHECK-NEXT: }
