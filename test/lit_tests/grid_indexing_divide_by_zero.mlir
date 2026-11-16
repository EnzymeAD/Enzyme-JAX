// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=scatter_op_canon;gather_op_canon" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// An index divided by a constant with a zero in it is not evaluated (the
// reference ops would trap), and the gather is left as it is.
func.func @divide_by_zero(%x: tensor<8xf64>) -> tensor<4xf64> {
  %i = stablehlo.iota dim = 0 : tensor<4x1xi64>
  %z = stablehlo.constant dense<[[1], [0], [1], [1]]> : tensor<4x1xi64>
  %d = stablehlo.divide %i, %z : tensor<4x1xi64>
  %g = "stablehlo.gather"(%x, %d) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<8xf64>, tensor<4x1xi64>) -> tensor<4xf64>
  return %g : tensor<4xf64>
}

// CHECK-LABEL: func.func @divide_by_zero(%arg0: tensor<8xf64>) -> tensor<4xf64> {
// CHECK-NEXT: %c = stablehlo.constant dense<{{\[\[}}1], [0], [1], [1]]> : tensor<4x1xi64>
// CHECK-NEXT: %0 = stablehlo.iota dim = 0 : tensor<4x1xi64>
// CHECK-NEXT: %1 = stablehlo.divide %0, %c : tensor<4x1xi64>
// CHECK-NEXT: %2 = "stablehlo.gather"(%arg0, %1) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<8xf64>, tensor<4x1xi64>) -> tensor<4xf64>
// CHECK-NEXT: return %2 : tensor<4xf64>
// CHECK-NEXT: }
