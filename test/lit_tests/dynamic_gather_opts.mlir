// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  // Test 1: Simple dynamic_gather(transpose(x)) with 2D transpose
  func.func @gather_transpose_2d(%arg0: tensor<6x6xf64>) -> tensor<5xf64> {
    %c = stablehlo.constant dense<[[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]> : tensor<5x2xi64>
    %c_0 = stablehlo.constant dense<1> : tensor<2xi64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<6x6xf64>) -> tensor<6x6xf64>
    %1 = "stablehlo.dynamic_gather"(%0, %c, %c_0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>}> : (tensor<6x6xf64>, tensor<5x2xi64>, tensor<2xi64>) -> tensor<5xf64>
    return %1 : tensor<5xf64>
  }

  // Test 2: Multiple gathers from the same transpose (from the issue)
  func.func @multiple_gathers_transpose(%arg0: tensor<6x6xf64>) -> (tensor<5xf64>, tensor<6xf64>, tensor<5xf64>) {
    %c_1 = stablehlo.constant dense<[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]> : tensor<5x2xi64>
    %c_2 = stablehlo.constant dense<[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]> : tensor<6x2xi64>
    %c_3 = stablehlo.constant dense<1> : tensor<2xi64>
    %c_4 = stablehlo.constant dense<[[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]> : tensor<5x2xi64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<6x6xf64>) -> tensor<6x6xf64>
    %1 = "stablehlo.dynamic_gather"(%0, %c_4, %c_3) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>}> : (tensor<6x6xf64>, tensor<5x2xi64>, tensor<2xi64>) -> tensor<5xf64>
    %2 = "stablehlo.dynamic_gather"(%0, %c_2, %c_3) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>}> : (tensor<6x6xf64>, tensor<6x2xi64>, tensor<2xi64>) -> tensor<6xf64>
    %3 = "stablehlo.dynamic_gather"(%0, %c_1, %c_3) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>}> : (tensor<6x6xf64>, tensor<5x2xi64>, tensor<2xi64>) -> tensor<5xf64>
    return %1, %2, %3 : tensor<5xf64>, tensor<6xf64>, tensor<5xf64>
  }

  // Test 3: 3D transpose case
  func.func @gather_transpose_3d(%arg0: tensor<4x5x6xf32>) -> tensor<2xf32> {
    %c = stablehlo.constant dense<[[1, 2, 3], [2, 3, 4]]> : tensor<2x3xi64>
    %c_0 = stablehlo.constant dense<1> : tensor<3xi64>
    %0 = stablehlo.transpose %arg0, dims = [2, 0, 1] : (tensor<4x5x6xf32>) -> tensor<6x4x5xf32>
    %1 = "stablehlo.dynamic_gather"(%0, %c, %c_0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 1>}> : (tensor<6x4x5xf32>, tensor<2x3xi64>, tensor<3xi64>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
  }
}

// CHECK-LABEL: func.func @gather_transpose_2d
// CHECK-SAME: (%arg0: tensor<6x6xf64>) -> tensor<5xf64>
// CHECK-NEXT: %[[C0:.+]] = stablehlo.constant dense<{{\[\[}}1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]> : tensor<5x2xi64>
// CHECK-NEXT: %[[C1:.+]] = stablehlo.constant dense<1> : tensor<2xi64>
// CHECK-NEXT: %[[RES:.+]] = "stablehlo.dynamic_gather"(%arg0, %[[C0]], %[[C1]])
// CHECK-SAME: dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1, 0], start_index_map = [1, 0], index_vector_dim = 1>
// CHECK-NEXT: return %[[RES]]

// CHECK-LABEL: func.func @multiple_gathers_transpose
// CHECK-SAME: (%arg0: tensor<6x6xf64>) -> (tensor<5xf64>, tensor<6xf64>, tensor<5xf64>)
// CHECK-NOT: stablehlo.transpose
// CHECK: %[[G1:.+]] = "stablehlo.dynamic_gather"(%arg0
// CHECK-SAME: dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1, 0], start_index_map = [1, 0]
// CHECK: %[[G2:.+]] = "stablehlo.dynamic_gather"(%arg0
// CHECK-SAME: dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1, 0], start_index_map = [1, 0]
// CHECK: %[[G3:.+]] = "stablehlo.dynamic_gather"(%arg0
// CHECK-SAME: dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1, 0], start_index_map = [1, 0]
// CHECK: return %[[G1]], %[[G2]], %[[G3]]

// CHECK-LABEL: func.func @gather_transpose_3d
// CHECK-SAME: (%arg0: tensor<4x5x6xf32>) -> tensor<2xf32>
// CHECK-NEXT: %[[C0:.+]] = stablehlo.constant dense<{{\[\[}}1, 2, 3], [2, 3, 4]]> : tensor<2x3xi64>
// CHECK-NEXT: %[[C1:.+]] = stablehlo.constant dense<1> : tensor<3xi64>
// CHECK-NEXT: %[[RES:.+]] = "stablehlo.dynamic_gather"(%arg0, %[[C0]], %[[C1]])
// CHECK-SAME: dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1, 2, 0], start_index_map = [2, 1, 0], index_vector_dim = 1>
// CHECK-NEXT: return %[[RES]]


