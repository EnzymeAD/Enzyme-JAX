// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt --split-input-file | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=gather_op_canon" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s --check-prefix=CANON

// A per-element gather whose index is the [0:2, 0:2] box of a 5x5 table
// `row | (col << 1)`, shifted left once more: the table is not linear as a
// whole (3 | 2 = 3), its box is (row + 2 col), so the box's values are read
// and the gather is a strided slice of every element's row.
func.func @box_gather(%x: tensor<13824x1125xf64>) -> tensor<13824x2x2xf64> {
  %c = stablehlo.constant dense<1> : tensor<5xi64>
  %c_1 = stablehlo.constant dense<[0, 1, 2, 3, 4]> : tensor<5xi64>
  %c_0 = stablehlo.constant dense<[[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]]> : tensor<5x5xi32>
  %c_3 = stablehlo.constant dense<1> : tensor<2x2x1xi32>
  %0 = stablehlo.shift_left %c_1, %c : tensor<5xi64>
  %1 = stablehlo.convert %0 : (tensor<5xi64>) -> tensor<5xi32>
  %2 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<5xi32>) -> tensor<5x5xi32>
  %3 = stablehlo.or %c_0, %2 : tensor<5x5xi32>
  %4 = stablehlo.reshape %3 : (tensor<5x5xi32>) -> tensor<5x5x1xi32>
  %5 = stablehlo.slice %4 [0:2, 0:2, 0:1] : (tensor<5x5x1xi32>) -> tensor<2x2x1xi32>
  %6 = stablehlo.shift_left %5, %c_3 : tensor<2x2x1xi32>
  %7 = stablehlo.convert %6 : (tensor<2x2x1xi32>) -> tensor<2x2x1xi64>
  %8 = stablehlo.broadcast_in_dim %7, dims = [0, 2, 3] : (tensor<2x2x1xi64>) -> tensor<2x13824x2x1xi64>
  %9 = stablehlo.transpose %8, dims = [1, 0, 2, 3] : (tensor<2x13824x2x1xi64>) -> tensor<13824x2x2x1xi64>
  %10 = "stablehlo.gather"(%x, %9) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<13824x1125xf64>, tensor<13824x2x2x1xi64>) -> tensor<13824x2x2xf64>
  return %10 : tensor<13824x2x2xf64>
}

// CHECK:      func.func @box_gather(%arg0: tensor<13824x1125xf64>) -> tensor<13824x2x2xf64> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [0:13824, 0:8] : (tensor<13824x1125xf64>) -> tensor<13824x8xf64>
// CHECK-NEXT:   %1 = stablehlo.reshape %0 : (tensor<13824x8xf64>) -> tensor<13824x2x2x2xf64>
// CHECK-NEXT:   %2 = stablehlo.slice %1 [0:13824, 0:2, 0:2, 0:1] : (tensor<13824x2x2x2xf64>) -> tensor<13824x2x2x1xf64>
// CHECK-NEXT:   %3 = stablehlo.reshape %2 : (tensor<13824x2x2x1xf64>) -> tensor<13824x2x2xf64>
// CHECK-NEXT:   %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<13824x2x2xf64>) -> tensor<13824x2x2xf64>
// CHECK-NEXT:   return %4 : tensor<13824x2x2xf64>
// CHECK-NEXT: }

// CANON:      func.func @box_gather(%arg0: tensor<13824x1125xf64>) -> tensor<13824x2x2xf64> {
// CANON-NEXT:   %0 = stablehlo.slice %arg0 [0:13824, 0:8] : (tensor<13824x1125xf64>) -> tensor<13824x8xf64>
// CANON-NEXT:   %1 = stablehlo.reshape %0 : (tensor<13824x8xf64>) -> tensor<13824x2x2x2xf64>
// CANON-NEXT:   %2 = stablehlo.slice %1 [0:13824, 0:2, 0:2, 0:1] : (tensor<13824x2x2x2xf64>) -> tensor<13824x2x2x1xf64>
// CANON-NEXT:   %3 = stablehlo.reshape %2 : (tensor<13824x2x2x1xf64>) -> tensor<13824x2x2xf64>
// CANON-NEXT:   %4 = stablehlo.broadcast_in_dim %3, dims = [0, 2, 1] : (tensor<13824x2x2xf64>) -> tensor<13824x2x2xf64>
// CANON-NEXT:   return %4 : tensor<13824x2x2xf64>
// CANON-NEXT: }
