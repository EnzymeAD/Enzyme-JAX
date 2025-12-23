// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// Test: reverse(reverse(x)) with same dimensions -> x
// CHECK-LABEL: @reverse_reverse_same_dims
// CHECK-NOT: stablehlo.reverse
func.func @reverse_reverse_same_dims(%arg0: tensor<8x4x3xf32>) -> tensor<8x4x3xf32> {
  %0 = stablehlo.reverse %arg0, dims = [0, 1] : tensor<8x4x3xf32>
  %1 = stablehlo.reverse %0, dims = [0, 1] : tensor<8x4x3xf32>
  return %1 : tensor<8x4x3xf32>
}

// Test: reverse(reverse(x)) with single dimension -> x
// CHECK-LABEL: @reverse_reverse_single_dim
// CHECK-NOT: stablehlo.reverse
func.func @reverse_reverse_single_dim(%arg0: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %0 = stablehlo.reverse %arg0, dims = [0] : tensor<8x4xf32>
  %1 = stablehlo.reverse %0, dims = [0] : tensor<8x4xf32>
  return %1 : tensor<8x4xf32>
}

// Test: reverse(reverse(x)) with partial overlap -> single reverse
// Inner dims [0, 1], outer dims [1, 2] -> effective dims [0, 2]
// CHECK-LABEL: @reverse_reverse_partial_overlap
// CHECK: %[[R:.*]] = stablehlo.reverse %arg0, dims = [0, 2] : tensor<8x4x3xf32>
// CHECK-NEXT: return %[[R]]
func.func @reverse_reverse_partial_overlap(%arg0: tensor<8x4x3xf32>) -> tensor<8x4x3xf32> {
  %0 = stablehlo.reverse %arg0, dims = [0, 1] : tensor<8x4x3xf32>
  %1 = stablehlo.reverse %0, dims = [1, 2] : tensor<8x4x3xf32>
  return %1 : tensor<8x4x3xf32>
}

// Test: reverse(reverse(x)) with disjoint dimensions -> combined reverse
// CHECK-LABEL: @reverse_reverse_disjoint_dims
// CHECK: %[[R:.*]] = stablehlo.reverse %arg0, dims = [0, 1, 2] : tensor<8x4x3xf32>
// CHECK-NEXT: return %[[R]]
func.func @reverse_reverse_disjoint_dims(%arg0: tensor<8x4x3xf32>) -> tensor<8x4x3xf32> {
  %0 = stablehlo.reverse %arg0, dims = [0, 1] : tensor<8x4x3xf32>
  %1 = stablehlo.reverse %0, dims = [2] : tensor<8x4x3xf32>
  return %1 : tensor<8x4x3xf32>
}

// Test: reverse(reverse(x)) with subset dimensions
// Inner has more dims, outer has subset -> remaining dims from inner
// CHECK-LABEL: @reverse_reverse_subset
// CHECK: %[[R:.*]] = stablehlo.reverse %arg0, dims = [2] : tensor<8x4x3xf32>
// CHECK-NEXT: return %[[R]]
func.func @reverse_reverse_subset(%arg0: tensor<8x4x3xf32>) -> tensor<8x4x3xf32> {
  %0 = stablehlo.reverse %arg0, dims = [0, 1, 2] : tensor<8x4x3xf32>
  %1 = stablehlo.reverse %0, dims = [0, 1] : tensor<8x4x3xf32>
  return %1 : tensor<8x4x3xf32>
}

// Test: 3 reverses - should optimize first two, then optimize result with third
// CHECK-LABEL: @reverse_three_times
// CHECK: %[[R:.*]] = stablehlo.reverse %arg0, dims = [0] : tensor<8x4xf32>
// CHECK-NEXT: return %[[R]]
func.func @reverse_three_times(%arg0: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %0 = stablehlo.reverse %arg0, dims = [0] : tensor<8x4xf32>
  %1 = stablehlo.reverse %0, dims = [0] : tensor<8x4xf32>
  %2 = stablehlo.reverse %1, dims = [0] : tensor<8x4xf32>
  return %2 : tensor<8x4xf32>
}
