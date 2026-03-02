// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%arg0: tensor<7x6xf64>) -> (tensor<4x3xf64>, tensor<7x6xf64>) {
    %c = stablehlo.constant dense<1> : tensor<3x1xi64>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<3x4xf64>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %c_0 = stablehlo.constant dense<[[1], [3], [2]]> : tensor<3x1xi64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<7x6xf64>) -> tensor<6x7xf64>
    %1 = stablehlo.subtract %c_0, %c : tensor<3x1xi64>
    %2 = "stablehlo.scatter"(%0, %1, %cst) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
    ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
      stablehlo.return %cst_1 : tensor<f64>
    }) : (tensor<6x7xf64>, tensor<3x1xi64>, tensor<3x4xf64>) -> tensor<6x7xf64>
    %3 = "stablehlo.gather"(%2, %1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 4>}> : (tensor<6x7xf64>, tensor<3x1xi64>) -> tensor<3x4xf64>
    %4 = stablehlo.transpose %3, dims = [1, 0] : (tensor<3x4xf64>) -> tensor<4x3xf64>
    %5 = stablehlo.transpose %2, dims = [1, 0] : (tensor<6x7xf64>) -> tensor<7x6xf64>
    return %4, %5 : tensor<4x3xf64>, tensor<7x6xf64>
  }
}

// CHECK: func.func @main
// CHECK:     %[[SCATTER:.*]] = "stablehlo.scatter"(%0, %c, %cst_0)
// CHECK-NOT:     "stablehlo.gather"
// CHECK:     %[[TS:.*]] = stablehlo.transpose %[[SCATTER]]
// CHECK:     return %cst, %[[TS]] : tensor<4x3xf64>, tensor<7x6xf64>

module {
  func.func @main(%arg0: tensor<7x6xf64>, %arg1: tensor<4x3xf64>) -> (tensor<4x3xf64>, tensor<7x6xf64>) {
    %c = stablehlo.constant dense<1> : tensor<3x1xi64>
    %c_0 = stablehlo.constant dense<[[1], [3], [2]]> : tensor<3x1xi64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<7x6xf64>) -> tensor<6x7xf64>
    %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<4x3xf64>) -> tensor<3x4xf64>
    %2 = stablehlo.subtract %c_0, %c : tensor<3x1xi64>
    %3 = "stablehlo.scatter"(%0, %2, %1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
      stablehlo.return %arg3 : tensor<f64>
    }) : (tensor<6x7xf64>, tensor<3x1xi64>, tensor<3x4xf64>) -> tensor<6x7xf64>
    %4 = "stablehlo.gather"(%3, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 4>}> : (tensor<6x7xf64>, tensor<3x1xi64>) -> tensor<3x4xf64>
    %5 = stablehlo.transpose %4, dims = [1, 0] : (tensor<3x4xf64>) -> tensor<4x3xf64>
    %6 = stablehlo.transpose %3, dims = [1, 0] : (tensor<6x7xf64>) -> tensor<7x6xf64>
    return %5, %6 : tensor<4x3xf64>, tensor<7x6xf64>
  }
}

// CHECK: func.func @main
// CHECK:     %[[SCATTER:.*]] = "stablehlo.scatter"(%0, %c, %1)
// CHECK-NOT:     "stablehlo.gather"
// CHECK:     %[[TS:.*]] = stablehlo.transpose %[[SCATTER]]
// CHECK:     return %arg1, %[[TS]] : tensor<4x3xf64>, tensor<7x6xf64>

module {
  func.func @main(%arg0: tensor<7x6xf64>, %arg1: tensor<4x3xf64>, %arg2: tensor<3xi64>, %arg3: tensor<4xi64>) -> (tensor<4x3xf64>, tensor<7x6xf64>) {
    %c = stablehlo.constant dense<1> : tensor<12x2xi64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<7x6xf64>) -> tensor<6x7xf64>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<3xi64>) -> tensor<4x3xi64>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<4xi64>) -> tensor<4x3xi64>
    %3 = stablehlo.reshape %2 : (tensor<4x3xi64>) -> tensor<12x1xi64>
    %4 = stablehlo.reshape %1 : (tensor<4x3xi64>) -> tensor<12x1xi64>
    %5 = stablehlo.concatenate %4, %3, dim = 1 : (tensor<12x1xi64>, tensor<12x1xi64>) -> tensor<12x2xi64>
    %6 = stablehlo.reshape %arg1 : (tensor<4x3xf64>) -> tensor<12xf64>
    %7 = stablehlo.subtract %5, %c : tensor<12x2xi64>
    %8 = "stablehlo.scatter"(%0, %7, %6) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg4: tensor<f64>, %arg5: tensor<f64>):
      stablehlo.return %arg5 : tensor<f64>
    }) : (tensor<6x7xf64>, tensor<12x2xi64>, tensor<12xf64>) -> tensor<6x7xf64>
    %9 = "stablehlo.gather"(%8, %7) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<6x7xf64>, tensor<12x2xi64>) -> tensor<12xf64>
    %10 = stablehlo.reshape %9 : (tensor<12xf64>) -> tensor<4x3xf64>
    %11 = stablehlo.transpose %8, dims = [1, 0] : (tensor<6x7xf64>) -> tensor<7x6xf64>
    return %10, %11 : tensor<4x3xf64>, tensor<7x6xf64>
  }
}

// CHECK: func.func @main
// CHECK:     %[[SCATTER:.*]] = "stablehlo.scatter"(%0, %7, %6)
// CHECK-NOT:     "stablehlo.gather"
// CHECK:     %[[TS:.*]] = stablehlo.transpose %[[SCATTER]]
// CHECK:     return %arg1, %[[TS]] : tensor<4x3xf64>, tensor<7x6xf64>
