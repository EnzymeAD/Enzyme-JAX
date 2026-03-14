// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// broadcast_in_dim: A_ik = A_i1
// C_ij = sum_k(A_ik * B_kj) = A_i1 * sum_k(B_kj)

func.func @contract_bcast_dim_left(%a: tensor<3xf32>, %b: tensor<4x4xf32>) -> tensor<3x4xf32> {
    %a_bcast = stablehlo.broadcast_in_dim %a, dims = [0] : (tensor<3xf32>) -> tensor<3x4xf32>
    %c = stablehlo.dot_general %a_bcast, %b, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<4x4xf32>) -> tensor<3x4xf32>
    return %c : tensor<3x4xf32>
}

// CHECK: func.func @contract_bcast_dim_left(%arg0: tensor<3xf32>, %arg1: tensor<4x4xf32>) -> tensor<3x4xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.0> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.reduce(%arg1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<4x4xf32>, tensor<f32>) -> tensor<4xf32>
// CHECK-NEXT:     %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<3xf32>, tensor<4xf32>) -> tensor<3x4xf32>
// CHECK-NEXT:     return %1 : tensor<3x4xf32>
// CHECK-NEXT: }


func.func @contract_bcast_dim_right(%a: tensor<3xf32>, %b: tensor<4x4xf32>) -> tensor<4x3xf32> {
    %a_bcast = stablehlo.broadcast_in_dim %a, dims = [0] : (tensor<3xf32>) -> tensor<3x4xf32>
    %c = stablehlo.dot_general %b, %a_bcast, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x4xf32>, tensor<3x4xf32>) -> tensor<4x3xf32>
    return %c : tensor<4x3xf32>
}

// CHECK: func.func @contract_bcast_dim_right(%arg0: tensor<3xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x3xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.0> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.reduce(%arg1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<4x4xf32>, tensor<f32>) -> tensor<4xf32>
// CHECK-NEXT:     %1 = stablehlo.dot_general %0, %arg0, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<4xf32>, tensor<3xf32>) -> tensor<4x3xf32>
// CHECK-NEXT:     return %1 : tensor<4x3xf32>
// CHECK-NEXT: }


func.func @contract_bcast_dim_both(%a: tensor<3xf32>, %b: tensor<4xf32>) -> tensor<3x4xf32> {
    %a_bcast = stablehlo.broadcast_in_dim %a, dims = [0] : (tensor<3xf32>) -> tensor<3x4xf32>
    %b_bcast = stablehlo.broadcast_in_dim %b, dims = [1] : (tensor<4xf32>) -> tensor<4x4xf32>
    %c = stablehlo.dot_general %a_bcast, %b_bcast, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<4x4xf32>) -> tensor<3x4xf32>
    return %c : tensor<3x4xf32>
}

// CHECK: func.func @contract_bcast_dim_both(%arg0: tensor<3xf32>, %arg1: tensor<4xf32>) -> tensor<3x4xf32> {
// CHECK-NEXT:     %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<3xf32>, tensor<4xf32>) -> tensor<3x4xf32>
// CHECK-NEXT:     return %0 : tensor<3x4xf32>
// CHECK-NEXT: }
