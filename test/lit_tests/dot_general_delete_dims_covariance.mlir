// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// Simplified version of the covariance example from the issue
func.func @covariance_simplified(%arg0: tensor<128x128x2048x1x1xf32>, %arg1: tensor<128x128x2048x1x1xf32>) -> tensor<128x128xf32> {
  %0 = stablehlo.reshape %arg0 : (tensor<128x128x2048x1x1xf32>) -> tensor<128x128x2048x1xf32>
  %1 = stablehlo.reshape %arg1 : (tensor<128x128x2048x1x1xf32>) -> tensor<128x128x2048x1xf32>
  %2 = stablehlo.dot_general %0, %1, batching_dims = [0, 1, 3] x [0, 1, 3], contracting_dims = [2] x [2] : (tensor<128x128x2048x1xf32>, tensor<128x128x2048x1xf32>) -> tensor<128x128x1xf32>
  %3 = stablehlo.reshape %2 : (tensor<128x128x1xf32>) -> tensor<128x128xf32>
  return %3 : tensor<128x128xf32>
}

// CHECK-LABEL: func.func @covariance_simplified
// CHECK-SAME:    %[[ARG0:.+]]: tensor<128x128x2048x1x1xf32>
// CHECK-SAME:    %[[ARG1:.+]]: tensor<128x128x2048x1x1xf32>
// The dot_general should now work on the original tensors and contract over the deleted dimension
// CHECK:         %[[DOT:.+]] = stablehlo.dot_general %[[ARG0]], %[[ARG1]]
// CHECK-SAME:      batching_dims = [0, 1, 3] x [0, 1, 3]
// CHECK-SAME:      contracting_dims = [2, 4] x [2, 4]
// The result is tensor<128x128x1xf32>, so the final reshape should still be there
// CHECK:         stablehlo.reshape
// CHECK-SAME:      tensor<128x128x1xf32>) -> tensor<128x128xf32>
