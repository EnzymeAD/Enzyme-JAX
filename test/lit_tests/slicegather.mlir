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
