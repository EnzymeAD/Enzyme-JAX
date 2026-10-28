// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=gather_of_broadcast_indexed_dims" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s --check-prefix=OPT

// A raised kernel's by-value table: nine scalars, broadcast to [3,3,3,9] and
// indexed by a constant grid whose first three components (into the added
// dimensions) carry nothing; the fourth reads entries 0, 3, 6 along the last
// grid dimension.
func.func @table(%t: tensor<9xf64>) -> tensor<3x3x3xf64> {
  %idx = stablehlo.constant dense<[[[[0, 0, 0, 0], [1, 0, 0, 3], [2, 0, 0, 6]], [[0, 0, 1, 0], [1, 0, 1, 3], [2, 0, 1, 6]], [[0, 0, 2, 0], [1, 0, 2, 3], [2, 0, 2, 6]]], [[[0, 0, 0, 0], [1, 0, 0, 3], [2, 0, 0, 6]], [[0, 0, 1, 0], [1, 0, 1, 3], [2, 0, 1, 6]], [[0, 0, 2, 0], [1, 0, 2, 3], [2, 0, 2, 6]]], [[[0, 0, 0, 0], [1, 0, 0, 3], [2, 0, 0, 6]], [[0, 0, 1, 0], [1, 0, 1, 3], [2, 0, 1, 6]], [[0, 0, 2, 0], [1, 0, 2, 3], [2, 0, 2, 6]]]]> : tensor<3x3x3x4xi64>
  %b = stablehlo.broadcast_in_dim %t, dims = [3] : (tensor<9xf64>) -> tensor<3x3x3x9xf64>
  %g = "stablehlo.gather"(%b, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2, 3], start_index_map = [0, 1, 2, 3], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1, 1>}> : (tensor<3x3x3x9xf64>, tensor<3x3x3x4xi64>) -> tensor<3x3x3xf64>
  return %g : tensor<3x3x3xf64>
}

// The added dimension sits between two of the source's: the kept components
// are two runs of the index vector.
func.func @middle(%x: tensor<4x10xf64>, %idx: tensor<5x3xi64>) -> tensor<5xf64> {
  %b = stablehlo.broadcast_in_dim %x, dims = [0, 2] : (tensor<4x10xf64>) -> tensor<4x6x10xf64>
  %g = "stablehlo.gather"(%b, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<4x6x10xf64>, tensor<5x3xi64>) -> tensor<5xf64>
  return %g : tensor<5xf64>
}

// A window along the added dimension reads different elements of it: the
// gather stays.
func.func @window(%x: tensor<10xf64>, %idx: tensor<5x2xi64>) -> tensor<5x2xf64> {
  %b = stablehlo.broadcast_in_dim %x, dims = [1] : (tensor<10xf64>) -> tensor<4x10xf64>
  %g = "stablehlo.gather"(%b, %idx) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 2, 1>}> : (tensor<4x10xf64>, tensor<5x2xi64>) -> tensor<5x2xf64>
  return %g : tensor<5x2xf64>
}

// The added dimension is not indexed at all (a full window): the gather stays.
func.func @offset(%x: tensor<10xf64>, %idx: tensor<5x1xi64>) -> tensor<5x4xf64> {
  %b = stablehlo.broadcast_in_dim %x, dims = [1] : (tensor<10xf64>) -> tensor<4x10xf64>
  %g = "stablehlo.gather"(%b, %idx) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 4, 1>}> : (tensor<4x10xf64>, tensor<5x1xi64>) -> tensor<5x4xf64>
  return %g : tensor<5x4xf64>
}

// CHECK-LABEL: func.func @table(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<9xf64>) -> tensor<3x3x3xf64> {
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<{{.*}}> : tensor<3x3x3x4xi64>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.slice %[[V1]] [0:3, 0:3, 0:3, 3:4] : (tensor<3x3x3x4xi64>) -> tensor<3x3x3x1xi64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V2]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<9xf64>, tensor<3x3x3x1xi64>) -> tensor<3x3x3xf64>
// CHECK-NEXT: return %[[V3]] : tensor<3x3x3xf64>
// CHECK-NEXT: }
// CHECK-LABEL: func.func @middle(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4x10xf64>, %[[V1:[^ ,:)]+]]: tensor<5x3xi64>) -> tensor<5xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.slice %[[V1]] [0:5, 0:1] : (tensor<5x3xi64>) -> tensor<5x1xi64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.slice %[[V1]] [0:5, 2:3] : (tensor<5x3xi64>) -> tensor<5x1xi64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.concatenate %[[V2]], %[[V3]], dim = 1 : (tensor<5x1xi64>, tensor<5x1xi64>) -> tensor<5x2xi64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = "stablehlo.gather"(%[[V0]], %[[V4]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<4x10xf64>, tensor<5x2xi64>) -> tensor<5xf64>
// CHECK-NEXT: return %[[V5]] : tensor<5xf64>
// CHECK-NEXT: }
// CHECK-LABEL: func.func @window(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<10xf64>, %[[V1:[^ ,:)]+]]: tensor<5x2xi64>) -> tensor<5x2xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V0]], dims = [1] : (tensor<10xf64>) -> tensor<4x10xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = "stablehlo.gather"(%[[V2]], %[[V1]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 2, 1>}> : (tensor<4x10xf64>, tensor<5x2xi64>) -> tensor<5x2xf64>
// CHECK-NEXT: return %[[V3]] : tensor<5x2xf64>
// CHECK-NEXT: }
// CHECK-LABEL: func.func @offset(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<10xf64>, %[[V1:[^ ,:)]+]]: tensor<5x1xi64>) -> tensor<5x4xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V0]], dims = [1] : (tensor<10xf64>) -> tensor<4x10xf64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = "stablehlo.gather"(%[[V2]], %[[V1]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 4, 1>}> : (tensor<4x10xf64>, tensor<5x1xi64>) -> tensor<5x4xf64>
// CHECK-NEXT: return %[[V3]] : tensor<5x4xf64>
// CHECK-NEXT: }

// With the whole pipeline the table is a strided slice of the scalars,
// broadcast along the two grid dimensions the index does not vary with.
// OPT-LABEL: func.func @table(
// OPT-SAME: %[[V0:[^ ,:)]+]]: tensor<9xf64>) -> tensor<3x3x3xf64> {
// OPT-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.slice %[[V0]] [0:9:3] : (tensor<9xf64>) -> tensor<3xf64>
// OPT-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V1]], dims = [2] : (tensor<3xf64>) -> tensor<3x3x3xf64>
// OPT-NEXT: return %[[V2]] : tensor<3x3x3xf64>
// OPT-NEXT: }
