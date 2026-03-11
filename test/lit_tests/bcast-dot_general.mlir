// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck

// broadcast_in_dim: A_ik = A_i1
// C_ij = sum_k(A_ik * B_kj) = A_i1 * sum_k(B_kj)
func.func @contract_bcast_dim(%a: tensor<3xf32>, %b: tensor<4x4xf32>) -> tensor<3x4xf32> {
    %a_bcast = stablehlo.broadcast_in_dim %a, dims = [0] : (tensor<3xf32>) -> tensor<3x4xf32>
    %c = stablehlo.dot_general %a_bcast, %b, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<4x4xf32>) -> tensor<3x4xf32>
    return %c : tensor<3x4xf32>
}

// CHECK: func.func @contract_bcast_dim(%arg0: tensor<3xf32>, %arg1: tensor<4x4xf32>) -> tensor<3x4xf32> {
// CHECK-NEXT:     %c = stablehlo.constant dense<0> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.reduce(%arg1 init: %c) applies stablehlo.add across dimensions = [0] : (tensor<4x4xf32>, tensor<f32>) -> tensor<4xf32>
// CHECK-NEXT:     %1 = stablehlo.dot_general %arg0, %0, batching_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3xf32>, tensor<4xf32>) -> tensor<3x4xf32>
// CHECK-NEXT:     return %1 : tensor<3x4xf32>
// CHECK-NEXT: }


// broadcast_in_dim: A_ik = A_1k
// C_ij = sum_k(A_ik * B_kj) = sum_k(A_1k * B_kj) = C_1j
func.func @contract_non_bcast_dim(%a: tensor<3xf32>, %b: tensor<4x3xf32>) -> tensor<4x4xf32> {
    %a_bcast = stablehlo.broadcast_in_dim %a, dims = [0] : (tensor<3xf32>) -> tensor<3x4xf32>
    %c = stablehlo.dot_general %a_bcast, %b, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<4x3xf32>) -> tensor<4x4xf32>
    return %c : tensor<4x4xf32>
}

// CHECK: func.func @contract_non_bcast_dim(%arg0: tensor<3xf32>, %arg1: tensor<4x4xf32>) -> tensor<3x4xf32> {
// CHECK-NEXT:     %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3xf32>, tensor<4x3xf32>) -> tensor<4xf32>
// CHECK-NEXT:     %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<4xf32>) -> tensor<3x4xf32>
// CHECK-NEXT:     return %1 : tensor<3x4xf32>
// CHECK-NEXT: }


// broadcast_in_dim: A_ik = A_i1
// C = sum_ij(A_ij * B_ji) = sum_ij(A_i1 * B_ji) = sum_i(A_i1 * sum_j(B_ji))
func.func @contract_all_dims(%a: tensor<3xf32>, %b: tensor<4x3xf32>) -> tensor<f32> {
    %a_bcast = stablehlo.broadcast_in_dim %a, dims = [0] : (tensor<3xf32>) -> tensor<3x4xf32>
    %c = stablehlo.dot_general %a_bcast, %b, contracting_dims = [0, 1] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<4x4xf32>) -> tensor<3x4xf32>
    return %c : tensor<3x4xf32>
}

// CHECK: func.func @contract_all_dims(%arg0: tensor<3xf32>, %arg1: tensor<4x4xf32>) -> tensor<3x4xf32> {
// CHECK-NEXT:     %c = stablehlo.constant dense<0> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.reduce(%arg1 init: %c) applies stablehlo.add across dimensions = [0] : (tensor<4x3xf32>, tensor<f32>) -> tensor<3xf32>
// CHECK-NEXT:     %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3xf32>, tensor<3xf32>) -> tensor<f32>
// CHECK-NEXT:     return %1 : tensor<f32>
// CHECK-NEXT: }
