// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(enzyme-hlo-opt{passses=65536},enzyme-hlo-opt)' | FileCheck %s

func.func @main1(%arg0: tensor<5x2xf32>, %arg1: tensor<4x3x2xf32>) -> tensor<5x2xf32> {
    %c = stablehlo.constant dense<[[[0, 1, 2, 3], [3, 1, 0, 2], [2, 4, 4, 2]]]> : tensor<1x3x4xi64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<5x2xf32>) -> tensor<2x5xf32>
    %1 = stablehlo.transpose %arg1, dims = [2, 1, 0] : (tensor<4x3x2xf32>) -> tensor<2x3x4xf32>
    %2 = "stablehlo.scatter"(%0, %c, %1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %4 = stablehlo.multiply %arg2, %arg3 : tensor<f32>
      stablehlo.return %4 : tensor<f32>
    }) : (tensor<2x5xf32>, tensor<1x3x4xi64>, tensor<2x3x4xf32>) -> tensor<2x5xf32>
    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<2x5xf32>) -> tensor<5x2xf32>
    return %3 : tensor<5x2xf32>
}

// CHECK: func.func @main1(%arg0: tensor<5x2xf32>, %arg1: tensor<4x3x2xf32>) -> tensor<5x2xf32> {
// CHECK-NEXT{LITERAL}:     %c = stablehlo.constant dense<[[[0, 1, 2, 3], [3, 1, 0, 2], [2, 4, 4, 2]]]> : tensor<1x3x4xi64>
// CHECK-NEXT:     %0 = stablehlo.transpose %arg1, dims = [2, 1, 0] : (tensor<4x3x2xf32>) -> tensor<2x3x4xf32>
// CHECK-NEXT:     %1 = "stablehlo.scatter"(%arg0, %c, %0) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [0]>}> ({
// CHECK-NEXT:     ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// CHECK-NEXT:       %2 = stablehlo.multiply %arg2, %arg3 : tensor<f32>
// CHECK-NEXT:       stablehlo.return %2 : tensor<f32>
// CHECK-NEXT:     }) : (tensor<5x2xf32>, tensor<1x3x4xi64>, tensor<2x3x4xf32>) -> tensor<5x2xf32>
// CHECK-NEXT:     return %1 : tensor<5x2xf32>
// CHECK-NEXT: }

func.func @main2(%arg0: tensor<5x2xf32>, %arg1: tensor<4x3x2xf32>) -> tensor<5x2xf32> {
    %c = stablehlo.constant dense<[[[0, 1, 2, 3], [3, 1, 0, 2], [2, 4, 4, 2]]]> : tensor<1x3x4xi64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<5x2xf32>) -> tensor<2x5xf32>
    %1 = stablehlo.transpose %arg1, dims = [2, 1, 0] : (tensor<4x3x2xf32>) -> tensor<2x3x4xf32>
    %2 = "stablehlo.scatter"(%0, %c, %1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %4 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %4 : tensor<f32>
    }) : (tensor<2x5xf32>, tensor<1x3x4xi64>, tensor<2x3x4xf32>) -> tensor<2x5xf32>
    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<2x5xf32>) -> tensor<5x2xf32>
    return %3 : tensor<5x2xf32>
}

// CHECK: func.func @main2(%arg0: tensor<5x2xf32>, %arg1: tensor<4x3x2xf32>) -> tensor<5x2xf32> {
// CHECK-NEXT{LITERAL}:     %c = stablehlo.constant dense<[[[0, 1, 2, 3], [3, 1, 0, 2], [2, 4, 4, 2]]]> : tensor<1x3x4xi64>
// CHECK-NEXT:     %0 = stablehlo.transpose %arg1, dims = [2, 1, 0] : (tensor<4x3x2xf32>) -> tensor<2x3x4xf32>
// CHECK-NEXT:     %1 = "stablehlo.scatter"(%arg0, %c, %0) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [0]>}> ({
// CHECK-NEXT:     ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// CHECK-NEXT:       %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
// CHECK-NEXT:       stablehlo.return %2 : tensor<f32>
// CHECK-NEXT:     }) : (tensor<5x2xf32>, tensor<1x3x4xi64>, tensor<2x3x4xf32>) -> tensor<5x2xf32>
// CHECK-NEXT:     return %1 : tensor<5x2xf32>
// CHECK-NEXT: }

func.func @main3(%arg0: tensor<32x32xf32>, %arg1: tensor<32xf32>) -> tensor<32x32xf32> {
    %c = stablehlo.constant dense<[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12], [13, 13], [14, 14], [15, 15], [16, 16], [17, 17], [18, 18], [19, 19], [20, 20], [21, 21], [22, 22], [23, 23], [24, 24], [25, 25], [26, 26], [27, 27], [28, 28], [29, 29], [30, 30], [31, 31]]> : tensor<32x2xi64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<32x32xf32>
    %0 = "stablehlo.scatter"(%cst, %c, %arg1) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      stablehlo.return %arg3 : tensor<f32>
    }) {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<32x32xf32>, tensor<32x2xi64>, tensor<32xf32>) -> tensor<32x32xf32>
    %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<32x32xf32>) -> tensor<32x32xf32>
    %2 = stablehlo.add %arg0, %1 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<32x32xf32>
    return %2 : tensor<32x32xf32>
}

// CHECK: func.func @main3(%arg0: tensor<32x32xf32>, %arg1: tensor<32xf32>) -> tensor<32x32xf32> {
// CHECK-NEXT{LITERAL}:     %c = stablehlo.constant dense<[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12], [13, 13], [14, 14], [15, 15], [16, 16], [17, 17], [18, 18], [19, 19], [20, 20], [21, 21], [22, 22], [23, 23], [24, 24], [25, 25], [26, 26], [27, 27], [28, 28], [29, 29], [30, 30], [31, 31]]> : tensor<32x2xi64>
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<32x32xf32>
// CHECK-NEXT:     %0 = "stablehlo.scatter"(%cst, %c, %arg1) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [1, 0], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:     ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// CHECK-NEXT:       stablehlo.return %arg3 : tensor<f32>
// CHECK-NEXT:     }) : (tensor<32x32xf32>, tensor<32x2xi64>, tensor<32xf32>) -> tensor<32x32xf32>
// CHECK-NEXT:     %1 = stablehlo.add %arg0, %0 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<32x32xf32>
// CHECK-NEXT:     return %1 : tensor<32x32xf32>
// CHECK-NEXT: }

func.func @main4(%arg0: tensor<3x4x4xf64>) -> tensor<3x4x4xf64> {
  %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
  %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<3x4x3xf64>
  %c = stablehlo.constant dense<[[0], [2], [1]]> : tensor<3x1xi64>
  %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<3x4x4xf64>) -> tensor<4x4x3xf64>
  %1 = "stablehlo.scatter"(%0, %c, %cst_0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
  ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
    stablehlo.return %cst : tensor<f64>
  }) : (tensor<4x4x3xf64>, tensor<3x1xi64>, tensor<3x4x3xf64>) -> tensor<4x4x3xf64>
  %2 = stablehlo.transpose %1, dims = [2, 1, 0] : (tensor<4x4x3xf64>) -> tensor<3x4x4xf64>
  return %2 : tensor<3x4x4xf64>
}

// CHECK: func.func @main4(%arg0: tensor<3x4x4xf64>) -> tensor<3x4x4xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<3x4x3xf64>
// CHECK-NEXT{LITERAL}:     %c = stablehlo.constant dense<[[0], [2], [1]]> : tensor<3x1xi64>
// CHECK-NEXT:     %0 = "stablehlo.scatter"(%arg0, %c, %cst_0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:     ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
// CHECK-NEXT:       stablehlo.return %cst : tensor<f64>
// CHECK-NEXT:     }) : (tensor<3x4x4xf64>, tensor<3x1xi64>, tensor<3x4x3xf64>) -> tensor<3x4x4xf64>
// CHECK-NEXT:     return %0 : tensor<3x4x4xf64>
// CHECK-NEXT: }
