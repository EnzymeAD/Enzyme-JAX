// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// Test that dot_general CSE recognizes swapped operands as equivalent
// when the dimension numbers are also appropriately swapped

// CHECK-LABEL: func.func @test_dotgeneral_cse_symmetric
func.func @test_dotgeneral_cse_symmetric(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  // Both operations have symmetric dimension specs: batching_dims = [0] x [0], contracting_dims = [1] x [1]
  // When operands are swapped with these symmetric specs, the operations are equivalent
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4xf32>
  %1 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4xf32>
  return %0, %1 : tensor<4xf32>, tensor<4xf32>
}

// CHECK: %[[V0:.+]] = stablehlo.dot_general %arg0, %arg1
// CHECK-NEXT: return %[[V0]], %[[V0]]

// CHECK-LABEL: func.func @test_dotgeneral_cse_asymmetric
func.func @test_dotgeneral_cse_asymmetric(%arg0: tensor<4x8xf32>, %arg1: tensor<8x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  // First: contracting_dims = [1] x [0] means A[:,*] × B[*,:]
  // Second: contracting_dims = [0] x [1] means B[*,:] × A[:,*]
  // These should be equivalent when operands are swapped
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
  %1 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [0] x [1] : (tensor<8x4xf32>, tensor<4x8xf32>) -> tensor<4x4xf32>
  return %0, %1 : tensor<4x4xf32>, tensor<4x4xf32>
}

// CHECK: %[[V0:.+]] = stablehlo.dot_general %arg0, %arg1
// CHECK-NEXT: return %[[V0]], %[[V0]]

// CHECK-LABEL: func.func @test_dotgeneral_no_cse_different_dims
func.func @test_dotgeneral_no_cse_different_dims(%arg0: tensor<4x8xf32>, %arg1: tensor<8x4xf32>) -> (tensor<4x4xf32>, tensor<8x8xf32>) {
  // These have different dimension patterns and should NOT be CSE'd
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
  %1 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [0] : (tensor<8x4xf32>, tensor<4x8xf32>) -> tensor<8x8xf32>
  return %0, %1 : tensor<4x4xf32>, tensor<8x8xf32>
}

// CHECK: %[[V0:.+]] = stablehlo.dot_general %arg0, %arg1
// CHECK-NEXT: %[[V1:.+]] = stablehlo.dot_general %arg1, %arg0
// CHECK-NEXT: return %[[V0]], %[[V1]]

// CHECK-LABEL: func.func @test_dotgeneral_cse_batched
func.func @test_dotgeneral_cse_batched(%arg0: tensor<2x4x8xf32>, %arg1: tensor<2x8x4xf32>) -> (tensor<2x4x4xf32>, tensor<2x4x4xf32>) {
  // With batching dimensions, check if swapped operands are recognized
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x4x8xf32>, tensor<2x8x4xf32>) -> tensor<2x4x4xf32>
  %1 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2] : (tensor<2x8x4xf32>, tensor<2x4x8xf32>) -> tensor<2x4x4xf32>
  return %0, %1 : tensor<2x4x4xf32>, tensor<2x4x4xf32>
}

// CHECK: %[[V0:.+]] = stablehlo.dot_general %arg0, %arg1
// CHECK-NEXT: return %[[V0]], %[[V0]]
