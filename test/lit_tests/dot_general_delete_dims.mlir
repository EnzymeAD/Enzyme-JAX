// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// Test case 1: Simple case where both operands have reshape that deletes a size-1 batching dimension
func.func @simple_delete_dims(%arg0: tensor<2x3x4x1xf32>, %arg1: tensor<2x3x4x1xf32>) -> tensor<2x3xf32> {
  %0 = stablehlo.reshape %arg0 : (tensor<2x3x4x1xf32>) -> tensor<2x3x4xf32>
  %1 = stablehlo.reshape %arg1 : (tensor<2x3x4x1xf32>) -> tensor<2x3x4xf32>
  %2 = stablehlo.dot_general %0, %1, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2] : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3xf32>
  return %2 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @simple_delete_dims
// CHECK-SAME:    %[[ARG0:.+]]: tensor<2x3x4x1xf32>
// CHECK-SAME:    %[[ARG1:.+]]: tensor<2x3x4x1xf32>
// CHECK-NEXT:    %[[RESHAPE0:.+]] = stablehlo.reshape %[[ARG0]]
// CHECK-NEXT:    %[[RESHAPE1:.+]] = stablehlo.reshape %[[ARG1]]
// CHECK-NEXT:    %[[DOT:.+]] = stablehlo.dot_general %[[RESHAPE0]], %[[RESHAPE1]]
// CHECK-SAME:      batching_dims = [0, 1] x [0, 1]
// CHECK-SAME:      contracting_dims = [2] x [2]
// CHECK-NEXT:    return %[[DOT]]

// Test case 2: Case from the issue where batching dimension is singleton
func.func @batching_singleton(%arg0: tensor<128x128x2048x1x1xf32>, %arg1: tensor<128x128x2048x1x1xf32>) -> tensor<128x128x1xf32> {
  %0 = stablehlo.reshape %arg0 : (tensor<128x128x2048x1x1xf32>) -> tensor<128x128x2048x1xf32>
  %1 = stablehlo.reshape %arg1 : (tensor<128x128x2048x1x1xf32>) -> tensor<128x128x2048x1xf32>
  %2 = stablehlo.dot_general %0, %1, batching_dims = [0, 1, 3] x [0, 1, 3], contracting_dims = [2] x [2] : (tensor<128x128x2048x1xf32>, tensor<128x128x2048x1xf32>) -> tensor<128x128x1xf32>
  return %2 : tensor<128x128x1xf32>
}

// CHECK-LABEL: func.func @batching_singleton
// CHECK-SAME:    %[[ARG0:.+]]: tensor<128x128x2048x1x1xf32>
// CHECK-SAME:    %[[ARG1:.+]]: tensor<128x128x2048x1x1xf32>
// CHECK-NEXT:    %[[RESHAPE0:.+]] = stablehlo.reshape %[[ARG0]]
// CHECK-NEXT:    %[[RESHAPE1:.+]] = stablehlo.reshape %[[ARG1]]
// CHECK-NEXT:    %[[DOT:.+]] = stablehlo.dot_general %[[RESHAPE0]], %[[RESHAPE1]]
// CHECK-SAME:      batching_dims = [0, 1] x [0, 1]
// CHECK-SAME:      contracting_dims = [2, 3] x [2, 3]
// CHECK-NEXT:    return %[[DOT]]

// Test case 3: No optimization when dimensions are not singleton
func.func @no_opt_non_singleton(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<2x3x4x5xf32>) -> tensor<2x3x5xf32> {
  %0 = stablehlo.reshape %arg0 : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
  %1 = stablehlo.reshape %arg1 : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
  %2 = stablehlo.dot_general %0, %1, batching_dims = [0, 1, 3] x [0, 1, 3], contracting_dims = [2] x [2] : (tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32>) -> tensor<2x3x5xf32>
  return %2 : tensor<2x3x5xf32>
}

// CHECK-LABEL: func.func @no_opt_non_singleton
// CHECK:         stablehlo.dot_general
// CHECK-SAME:      batching_dims = [0, 1, 3] x [0, 1, 3]
// CHECK-SAME:      contracting_dims = [2] x [2]
