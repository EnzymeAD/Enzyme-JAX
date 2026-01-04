// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// Test case 1: Simple case where both operands have reshape that deletes a size-1 dimension
func.func @simple_delete_dims(%arg0: tensor<2x3x4x1xf32>, %arg1: tensor<2x3x4x1xf32>) -> tensor<2x3xf32> {
  %0 = stablehlo.reshape %arg0 : (tensor<2x3x4x1xf32>) -> tensor<2x3x4xf32>
  %1 = stablehlo.reshape %arg1 : (tensor<2x3x4x1xf32>) -> tensor<2x3x4xf32>
  %2 = stablehlo.dot_general %0, %1, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2] : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3xf32>
  return %2 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @simple_delete_dims
// CHECK-SAME:    %[[ARG0:.+]]: tensor<2x3x4x1xf32>
// CHECK-SAME:    %[[ARG1:.+]]: tensor<2x3x4x1xf32>
// CHECK-NEXT:    %[[DOT:.+]] = stablehlo.dot_general %[[ARG0]], %[[ARG1]]
// CHECK-SAME:      batching_dims = [0, 1] x [0, 1]
// CHECK-SAME:      contracting_dims = [2, 3] x [2, 3]
// CHECK-NEXT:    return %[[DOT]]

// Test case 2: Case from the issue where deleted dimension creates opportunity
func.func @batching_singleton(%arg0: tensor<128x128x2048x1x1xf32>, %arg1: tensor<128x128x2048x1x1xf32>) -> tensor<128x128x1xf32> {
  %0 = stablehlo.reshape %arg0 : (tensor<128x128x2048x1x1xf32>) -> tensor<128x128x2048x1xf32>
  %1 = stablehlo.reshape %arg1 : (tensor<128x128x2048x1x1xf32>) -> tensor<128x128x2048x1xf32>
  %2 = stablehlo.dot_general %0, %1, batching_dims = [0, 1, 3] x [0, 1, 3], contracting_dims = [2] x [2] : (tensor<128x128x2048x1xf32>, tensor<128x128x2048x1xf32>) -> tensor<128x128x1xf32>
  return %2 : tensor<128x128x1xf32>
}

// CHECK-LABEL: func.func @batching_singleton
// CHECK-SAME:    %[[ARG0:.+]]: tensor<128x128x2048x1x1xf32>
// CHECK-SAME:    %[[ARG1:.+]]: tensor<128x128x2048x1x1xf32>
// CHECK-NEXT:    %[[DOT:.+]] = stablehlo.dot_general %[[ARG0]], %[[ARG1]]
// CHECK-SAME:      batching_dims = [0, 1, 3] x [0, 1, 3]
// CHECK-SAME:      contracting_dims = [2, 4] x [2, 4]
// CHECK-NEXT:    return %[[DOT]]

// Test case 3: No optimization when only one operand has reshape
func.func @no_opt_one_reshape(%arg0: tensor<2x3x4x1xf32>, %arg1: tensor<2x3x4xf32>) -> tensor<2x3xf32> {
  %0 = stablehlo.reshape %arg0 : (tensor<2x3x4x1xf32>) -> tensor<2x3x4xf32>
  %1 = stablehlo.dot_general %0, %arg1, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2] : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @no_opt_one_reshape
// CHECK:         stablehlo.reshape
// CHECK:         stablehlo.dot_general

