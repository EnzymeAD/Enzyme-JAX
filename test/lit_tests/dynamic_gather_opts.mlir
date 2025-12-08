// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  // Test 1: dynamic_gather(transpose(x)) should be rewritten to dynamic_gather(x)
  func.func @gather_transpose(%arg0: tensor<6x6xf64>) -> tensor<5xf64> {
    %c = stablehlo.constant dense<[[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]> : tensor<5x2xi64>
    %c_0 = stablehlo.constant dense<1> : tensor<2xi64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<6x6xf64>) -> tensor<6x6xf64>
    %1 = "stablehlo.dynamic_gather"(%0, %c, %c_0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>}> : (tensor<6x6xf64>, tensor<5x2xi64>, tensor<2xi64>) -> tensor<5xf64>
    return %1 : tensor<5xf64>
  }
}

// CHECK-LABEL: func.func @gather_transpose
// CHECK-SAME: (%arg0: tensor<6x6xf64>) -> tensor<5xf64>
// CHECK-NEXT: %[[C0:.+]] = stablehlo.constant dense<{{\[\[}}1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]> : tensor<5x2xi64>
// CHECK-NEXT: %[[C1:.+]] = stablehlo.constant dense<1> : tensor<2xi64>
// CHECK-NEXT: %[[RES:.+]] = "stablehlo.dynamic_gather"(%arg0, %[[C0]], %[[C1]])
// CHECK-SAME: dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [1, 0], index_vector_dim = 1>
// CHECK-NEXT: return %[[RES]]

