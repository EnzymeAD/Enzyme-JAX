// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// Test that dot_general CSE recognizes swapped operands as equivalent
// when the dimension numbers are also appropriately swapped

// CHECK-LABEL: func.func @test_dotgeneral_cse_symmetric
func.func @test_dotgeneral_cse_symmetric(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  // Both operations have symmetric dimension specs: batching_dims = [0] x [0], contracting_dims = [1] x [1]
  // This computes: for each b in [0,4), result[b] = sum_k arg0[b,k] * arg1[b,k]
  // When operands are swapped with these symmetric specs, the operations are equivalent
  // because multiplication is commutative: arg0[b,k] * arg1[b,k] = arg1[b,k] * arg0[b,k]
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4xf32>
  %1 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4xf32>
  return %0, %1 : tensor<4xf32>, tensor<4xf32>
}

// CHECK: %[[V0:.+]] = stablehlo.dot_general %arg0, %arg1
// CHECK-NEXT: return %[[V0]], %[[V0]]

// CHECK-LABEL: func.func @test_dotgeneral_cse_with_batching
func.func @test_dotgeneral_cse_with_batching(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x3x4xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  // With batching_dims = [0] x [0] and contracting_dims = [1, 2] x [1, 2]
  // This computes: for each b in [0,2), result[b] = sum_{i,j} arg0[b,i,j] * arg1[b,i,j]
  // These are equivalent when operands are swapped (Frobenius inner product is commutative)
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [1, 2] x [1, 2] : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2xf32>
  %1 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1, 2] x [1, 2] : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2xf32>
  return %0, %1 : tensor<2xf32>, tensor<2xf32>
}

// CHECK: %[[V0:.+]] = stablehlo.dot_general %arg0, %arg1
// CHECK-NEXT: return %[[V0]], %[[V0]]

// CHECK-LABEL: func.func @test_dotgeneral_no_cse_different_shapes
func.func @test_dotgeneral_no_cse_different_shapes(%arg0: tensor<4x8xf32>, %arg1: tensor<8x4xf32>) -> (tensor<4x4xf32>, tensor<8x8xf32>) {
  // These have different output shapes and should NOT be CSE'd
  // First: A(4x8) × B(8x4) → (4x4)
  // Second: B(8x4) × A(4x8) with same contracting dims → (8x8)
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
  %1 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [0] : (tensor<8x4xf32>, tensor<4x8xf32>) -> tensor<8x8xf32>
  return %0, %1 : tensor<4x4xf32>, tensor<8x8xf32>
}

// CHECK: %[[V0:.+]] = stablehlo.dot_general %arg0, %arg1
// CHECK-NEXT: %[[V1:.+]] = stablehlo.dot_general %arg1, %arg0
// CHECK-NEXT: return %[[V0]], %[[V1]]

// CHECK-LABEL: func.func @test_dotgeneral_no_cse_different_computation
func.func @test_dotgeneral_no_cse_different_computation(%arg0: tensor<4x8xf32>, %arg1: tensor<8x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  // These have the same output shape but compute different things
  // First: A(4x8) × B(8x4) → C(4x4) where C[i,j] = sum_k A[i,k] * B[k,j]
  // Second: B(8x4) × A(4x8) with contracting_dims=[0]x[1] → D(4x4) where D[i,j] = sum_k B[k,i] * A[j,k]
  // These are NOT equivalent (D is not the same as C)
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
  %1 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [0] x [1] : (tensor<8x4xf32>, tensor<4x8xf32>) -> tensor<4x4xf32>
  return %0, %1 : tensor<4x4xf32>, tensor<4x4xf32>
}

// CHECK: %[[V0:.+]] = stablehlo.dot_general %arg0, %arg1
// CHECK-NEXT: %[[V1:.+]] = stablehlo.dot_general %arg1, %arg0
// CHECK-NEXT: return %[[V0]], %[[V1]]


