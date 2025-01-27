// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @simple_concat(%0: tensor<6x6xf64>) -> tensor<16xf64> {
    %c_1 = stablehlo.constant dense<[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]> : tensor<5x2xi64>
    %c_2 = stablehlo.constant dense<[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]> : tensor<6x2xi64>
    %c_3 = stablehlo.constant dense<[[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]> : tensor<5x2xi64>
    %1 = "stablehlo.gather"(%0, %c_3) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1>}> : (tensor<6x6xf64>, tensor<5x2xi64>) -> tensor<5xf64>
    %2 = "stablehlo.gather"(%0, %c_2) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1>}> : (tensor<6x6xf64>, tensor<6x2xi64>) -> tensor<6xf64>
    %3 = "stablehlo.gather"(%0, %c_1) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1>}> : (tensor<6x6xf64>, tensor<5x2xi64>) -> tensor<5xf64>
    %4 = stablehlo.concatenate %1, %2, %3, dim = 0 : (tensor<5xf64>, tensor<6xf64>, tensor<5xf64>) -> tensor<16xf64>
    return %4 : tensor<16xf64>
  }

  func.func @two_part_concat(%0: tensor<6x6xf64>) -> tensor<24xf64> {
    %c_1 = stablehlo.constant dense<[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]> : tensor<5x2xi64>
    %c_2 = stablehlo.constant dense<[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]> : tensor<6x2xi64>
    %c_3 = stablehlo.constant dense<[[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]> : tensor<5x2xi64>
    %c_4 = stablehlo.constant dense<1.000000e+01> : tensor<3xf64>
    %1 = "stablehlo.gather"(%0, %c_3) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1>}> : (tensor<6x6xf64>, tensor<5x2xi64>) -> tensor<5xf64>
    %2 = "stablehlo.gather"(%0, %c_2) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1>}> : (tensor<6x6xf64>, tensor<6x2xi64>) -> tensor<6xf64>
    %3 = "stablehlo.gather"(%0, %c_1) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1>}> : (tensor<6x6xf64>, tensor<5x2xi64>) -> tensor<5xf64>
    %4 = "stablehlo.gather"(%0, %c_1) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1>}> : (tensor<6x6xf64>, tensor<5x2xi64>) -> tensor<5xf64>
    %5 = stablehlo.concatenate %1, %2, %c_4, %3, %4, dim = 0 : (tensor<5xf64>, tensor<6xf64>, tensor<3xf64>, tensor<5xf64>, tensor<5xf64>) -> tensor<24xf64>
    return %5 : tensor<24xf64>
  }
}



// CHECK: module {
// CHECK-NEXT:   func.func @simple_concat(%arg0: tensor<6x6xf64>) -> tensor<16xf64> {
// CHECK-NEXT:     %c = stablehlo.constant dense<\[\[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [0, 1], [1, 2], [2, 3], [3, 4], [4, 5\]\]> : tensor<16x2xi64>
// CHECK-NEXT:     %0 = "stablehlo.gather"(%arg0, %c) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<6x6xf64>, tensor<16x2xi64>) -> tensor<16xf64>
// CHECK-NEXT:     return %0 : tensor<16xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:  func.func @two_part_concat(%arg0: tensor<6x6xf64>) -> tensor<24xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<\[\[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 1], [1, 2], [2, 3], [3, 4], [4, 5\]\]> : tensor<10x2xi64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<1.000000e+01> : tensor<3xf64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<\[\[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5\]\]> : tensor<11x2xi64>
// CHECK-NEXT:    %0 = "stablehlo.gather"(%arg0, %c_0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<6x6xf64>, tensor<11x2xi64>) -> tensor<11xf64>
// CHECK-NEXT:    %1 = "stablehlo.gather"(%arg0, %c) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<6x6xf64>, tensor<10x2xi64>) -> tensor<10xf64>
// CHECK-NEXT:    %2 = stablehlo.concatenate %0, %cst, %1, dim = 0 : (tensor<11xf64>, tensor<3xf64>, tensor<10xf64>) -> tensor<24xf64>
// CHECK-NEXT:    return %2 : tensor<24xf64>
// CHECK-NEXT:  }
// CHECK-NEXT: }
