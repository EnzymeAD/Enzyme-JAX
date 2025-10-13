// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main(%arg0: tensor<1024xf32>, %arg1: tensor<1024x32xf32>, %arg2: tensor<1024x32xf32>) -> (tensor<24xf32>, tensor<1024x32xf32>, tensor<24x1024xf32>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<24x1024xf32>
    %0 = stablehlo.iota dim = 0 : tensor<24x2xi64>
    %1 = stablehlo.dot_general %cst, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<24x1024xf32>, tensor<1024x32xf32>) -> tensor<24x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xf32>) -> tensor<1024x32xf32>
    %3 = "stablehlo.gather"(%1, %0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1>}> : (tensor<24x32xf32>, tensor<24x2xi64>) -> tensor<24xf32>
    return %3, %2, %cst : tensor<24xf32>, tensor<1024x32xf32>, tensor<24x1024xf32>
}

// CHECK: module {
// CHECK-NEXT:   func.func @main(%arg0: tensor<1024xf32>, %arg1: tensor<1024x32xf32>, %arg2: tensor<1024x32xf32>) -> (tensor<24xf32>, tensor<1024x32xf32>, tensor<24x1024xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.000000e+00> : tensor<24x1024xf32>
// CHECK-NEXT{LITERAL}:     %c = stablehlo.constant dense<[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12], [13, 13], [14, 14], [15, 15], [16, 16], [17, 17], [18, 18], [19, 19], [20, 20], [21, 21], [22, 22], [23, 23]]> : tensor<24x2xi64>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.reduce(%arg1 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<1024x32xf32>, tensor<f32>) -> tensor<32xf32>
// CHECK-NEXT:     %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<32xf32>) -> tensor<24x32xf32>
// CHECK-NEXT:     %2 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xf32>) -> tensor<1024x32xf32>
// CHECK-NEXT:     %3 = "stablehlo.gather"(%1, %c) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1>}> : (tensor<24x32xf32>, tensor<24x2xi64>) -> tensor<24xf32>
// CHECK-NEXT:     return %3, %2, %cst : tensor<24xf32>, tensor<1024x32xf32>, tensor<24x1024xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
