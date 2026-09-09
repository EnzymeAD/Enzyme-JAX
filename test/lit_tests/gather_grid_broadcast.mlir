// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// The iota grid varies only along the last result dimension; the other two
// are broadcast. The single-mapping rewrite must restore them by broadcast —
// a reshape of the 3-element slice to the 27-element result does not verify.
func.func @gather_grid_broadcast(%tbl: tensor<6xf64>) -> tensor<3x3x3xf64> {
  %iota = stablehlo.iota dim = 0 : tensor<3xi64>
  %b = stablehlo.broadcast_in_dim %iota, dims = [2] : (tensor<3xi64>) -> tensor<3x3x3xi64>
  %r = stablehlo.reshape %b : (tensor<3x3x3xi64>) -> tensor<3x3x3x1xi64>
  %g = "stablehlo.gather"(%tbl, %r) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<6xf64>, tensor<3x3x3x1xi64>) -> tensor<3x3x3xf64>
  return %g : tensor<3x3x3xf64>
}

// CHECK-LABEL: func.func @gather_grid_broadcast(
// CHECK-SAME: %[[TBL:[a-z0-9]+]]: tensor<6xf64>
// CHECK: %[[S:.+]] = stablehlo.slice %[[TBL]] [0:3] : (tensor<6xf64>) -> tensor<3xf64>
// CHECK: %[[B:.+]] = stablehlo.broadcast_in_dim %[[S]], dims = [2] : (tensor<3xf64>) -> tensor<3x3x3xf64>
// CHECK: return %[[B]] : tensor<3x3x3xf64>
