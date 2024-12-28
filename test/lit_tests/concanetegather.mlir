// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
         func.func @main(%arg0: tensor<6x6xf64>) -> tensor<6x6xf64> {
           %cst = stablehlo.constant dense<1.000000e+00> : tensor<6x6xf64>
           %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<6x6xf64>
           %c = stablehlo.constant dense<[[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]> : tensor<16x2xi64>
           %c_1 = stablehlo.constant dense<[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]> : tensor<5x2xi64>
           %c_2 = stablehlo.constant dense<[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]> : tensor<6x2xi64>
           %c_3 = stablehlo.constant dense<1> : tensor<2xi64>
           %c_4 = stablehlo.constant dense<[[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]> : tensor<5x2xi64>
           %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<6x6xf64>) -> tensor<6x6xf64>
           %1 = "stablehlo.dynamic_gather"(%0, %c_4, %c_3) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>}> : (tensor<6x6xf64>, tensor<5x2xi64>, tensor<2xi64>) -> tensor<5xf64>
           %2 = "stablehlo.dynamic_gather"(%0, %c_2, %c_3) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>}> : (tensor<6x6xf64>, tensor<6x2xi64>, tensor<2xi64>) -> tensor<6xf64>
           %3 = "stablehlo.dynamic_gather"(%0, %c_1, %c_3) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>}> : (tensor<6x6xf64>, tensor<5x2xi64>, tensor<2xi64>) -> tensor<5xf64>
           %4 = stablehlo.concatenate %1, %2, %3, dim = 0 : (tensor<5xf64>, tensor<6xf64>, tensor<5xf64>) -> tensor<16xf64>
           %5 = "stablehlo.scatter"(%cst_0, %c, %4) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
           ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
             stablehlo.return %arg2 : tensor<f64>
           }) : (tensor<6x6xf64>, tensor<16x2xi64>, tensor<16xf64>) -> tensor<6x6xf64>
           %6 = stablehlo.add %5, %cst : tensor<6x6xf64>
           %7 = stablehlo.transpose %6, dims = [1, 0] : (tensor<6x6xf64>) -> tensor<6x6xf64>
           return %7 : tensor<6x6xf64>
         }
      }

// CHECK: func.func @main(%arg0: tensor<6x6xf64>) -> tensor<6x6xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<6x6xf64>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<6x6xf64>
// CHECK-NEXT:    %c = stablehlo.constant dense<[[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]> : tensor<16x2xi64>
// CHECK-NEXT:    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<6x6xf64>) -> tensor<6x6xf64>
// CHECK-NEXT:    %1 = "stablehlo.gather"(%0, %c) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<6x6xf64>, tensor<16x2xi64>) -> tensor<16xf64>
// CHECK-NEXT:    %2 = "stablehlo.scatter"(%cst_0, %c, %1) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
// CHECK-NEXT:    ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
// CHECK-NEXT:    stablehlo.return %arg2 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<6x6xf64>, tensor<16x2xi64>, tensor<16xf64>) -> tensor<6x6xf64>
// CHECK-NEXT:    %3 = stablehlo.add %2, %cst : tensor<6x6xf64>
// CHECK-NEXT:    %4 = stablehlo.transpose %3, dims = [1, 0] : (tensor<6x6xf64>) -> tensor<6x6xf64>
// CHECK-NEXT:    return %4 : tensor<6x6xf64>
// CHECK-NEXT:    }

