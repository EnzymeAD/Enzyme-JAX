// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%0: tensor<6x6xf64>) -> tensor<16xf64> {
    %c_1 = stablehlo.constant dense<[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]> : tensor<5x2xi64>
    %c_2 = stablehlo.constant dense<[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]> : tensor<6x2xi64>
    %c_3 = stablehlo.constant dense<[[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]> : tensor<5x2xi64>
    %1 = "stablehlo.gather"(%0, %c_3) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1>}> : (tensor<6x6xf64>, tensor<5x2xi64>) -> tensor<5xf64>
    %2 = "stablehlo.gather"(%0, %c_2) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1>}> : (tensor<6x6xf64>, tensor<6x2xi64>) -> tensor<6xf64>
    %3 = "stablehlo.gather"(%0, %c_1) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1>}> : (tensor<6x6xf64>, tensor<5x2xi64>) -> tensor<5xf64>
    %4 = stablehlo.concatenate %1, %2, %3, dim = 0 : (tensor<5xf64>, tensor<6xf64>, tensor<5xf64>) -> tensor<16xf64>
    return %4 : tensor<16xf64>
  }
}

// CHECK: module {
// CHECK-NEXT:   func.func @main(%arg0: tensor<6x6xf64>) -> tensor<16xf64> {
// CHECK-NEXT:     %c = stablehlo.constant dense<\[\[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [0, 1], [1, 2], [2, 3], [3, 4], [4, 5\]\]> : tensor<16x2xi64>
// CHECK-NEXT:     %0 = "stablehlo.gather"(%arg0, %c) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<6x6xf64>, tensor<16x2xi64>) -> tensor<16xf64>
// CHECK-NEXT:     return %0 : tensor<16xf64>
// CHECK-NEXT:   }
// CHECK-NEXT: }
