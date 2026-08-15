// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// Positive case: slice(gather(x, ind)) -> gather(x, slice(ind)).
// Gather has a single user (the slice), so the rewrite applies.
func.func @slice_of_gather(%arg0: tensor<10xf32>, %arg1: tensor<10x1xi32>) -> tensor<2xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<10xf32>, tensor<10x1xi32>) -> tensor<10xf32>
  %1 = stablehlo.slice %0 [0:2] : (tensor<10xf32>) -> tensor<2xf32>
  return %1 : tensor<2xf32>
}

// CHECK-LABEL: func.func @slice_of_gather
// CHECK-NEXT:   %0 = stablehlo.slice %arg1 [0:2, 0:1] : (tensor<10x1xi32>) -> tensor<2x1xi32>
// CHECK-NEXT:   %1 = "stablehlo.gather"(%arg0, %0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<10xf32>, tensor<2x1xi32>) -> tensor<2xf32>
// CHECK-NEXT:   return %1 : tensor<2xf32>
// CHECK-NEXT: }

// Negative case: gather has two users (slice + direct return).
// The rewrite must NOT apply to avoid introducing a redundant gather.
func.func @slice_of_gather_multiple_users(%arg0: tensor<10xf32>, %arg1: tensor<10x1xi32>) -> (tensor<2xf32>, tensor<10xf32>) {
  %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<10xf32>, tensor<10x1xi32>) -> tensor<10xf32>
  %1 = stablehlo.slice %0 [0:2] : (tensor<10xf32>) -> tensor<2xf32>
  return %1, %0 : tensor<2xf32>, tensor<10xf32>
}

// CHECK-LABEL: func.func @slice_of_gather_multiple_users
// CHECK: "stablehlo.gather"(%arg0, %arg1) {{.*}} -> tensor<10xf32>

// Positive case: slicing a batching dim slices both the operand's paired
// batching dim and the start_indices batching dim in lockstep.
func.func @slice_of_gather_batching(%arg0: tensor<3x4xf32>, %arg1: tensor<3x5x1xi64>) -> tensor<2x5xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<3x4xf32>, tensor<3x5x1xi64>) -> tensor<3x5xf32>
  %1 = stablehlo.slice %0 [0:2, 0:5] : (tensor<3x5xf32>) -> tensor<2x5xf32>
  return %1 : tensor<2x5xf32>
}

// CHECK-LABEL: func.func @slice_of_gather_batching
// CHECK-NEXT:   %0 = stablehlo.slice %arg1 [0:2, 0:5, 0:1] : (tensor<3x5x1xi64>) -> tensor<2x5x1xi64>
// CHECK-NEXT:   %1 = stablehlo.slice %arg0 [0:2, 0:4] : (tensor<3x4xf32>) -> tensor<2x4xf32>
// CHECK-NEXT:   %2 = "stablehlo.gather"(%1, %0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<2x4xf32>, tensor<2x5x1xi64>) -> tensor<2x5xf32>
// CHECK-NEXT:   return %2 : tensor<2x5xf32>
