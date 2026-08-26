// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%arg0: tensor<4x4xf64>, %arg1: tensor<f32>) -> tensor<4x4xf64> {
    %c = stablehlo.constant dense<[[0, 0], [1, 1], [2, 2], [3, 3]]> : tensor<4x2xi64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf64>
    %0 = stablehlo.convert %arg1 : (tensor<f32>) -> tensor<f64>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %2 = "stablehlo.scatter"(%cst, %c, %1) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
      stablehlo.return %arg3 : tensor<f64>
    }) : (tensor<4x4xf64>, tensor<4x2xi64>, tensor<4xf64>) -> tensor<4x4xf64>
    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<4x4xf64>) -> tensor<4x4xf64>
    %4 = stablehlo.subtract %arg0, %3 : tensor<4x4xf64>
    return %4 : tensor<4x4xf64>
  }
}

// CHECK: func.func @main(%arg0: tensor<4x4xf64>, %arg1: tensor<f32>) -> tensor<4x4xf64> {
// CHECK-NEXT{LITERAL}:     %c = stablehlo.constant dense<[[0, 0], [1, 1], [2, 2], [3, 3]]> : tensor<4x2xi64>
// CHECK-NEXT:     %0 = stablehlo.convert %arg1 : (tensor<f32>) -> tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:     %2 = "stablehlo.scatter"(%arg0, %c, %1) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:     ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:       %3 = stablehlo.subtract %arg2, %arg3 : tensor<f64>
// CHECK-NEXT:       stablehlo.return %3 : tensor<f64>
// CHECK-NEXT:     }) : (tensor<4x4xf64>, tensor<4x2xi64>, tensor<4xf64>) -> tensor<4x4xf64>
// CHECK-NEXT:     return %2 : tensor<4x4xf64>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<4x4xf64>, %arg1: tensor<f32>) -> tensor<4x4xf64> {
    %c = stablehlo.constant dense<[[0, 0], [1, 1], [2, 2], [3, 3]]> : tensor<4x2xi64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf64>
    %0 = stablehlo.convert %arg1 : (tensor<f32>) -> tensor<f64>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %2 = "stablehlo.scatter"(%cst, %c, %1) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
      stablehlo.return %arg3 : tensor<f64>
    }) : (tensor<4x4xf64>, tensor<4x2xi64>, tensor<4xf64>) -> tensor<4x4xf64>
    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<4x4xf64>) -> tensor<4x4xf64>
    %4 = stablehlo.subtract %3, %arg0 : tensor<4x4xf64>
    return %4 : tensor<4x4xf64>
  }
}

// CHECK: func.func @main(%arg0: tensor<4x4xf64>, %arg1: tensor<f32>) -> tensor<4x4xf64> {
// CHECK-NEXT{LITERAL}:     %c = stablehlo.constant dense<[[0, 0], [1, 1], [2, 2], [3, 3]]> : tensor<4x2xi64>
// CHECK-NEXT:     %0 = stablehlo.convert %arg1 : (tensor<f32>) -> tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:     %2 = stablehlo.negate %arg0 : tensor<4x4xf64>
// CHECK-NEXT:     %3 = "stablehlo.scatter"(%2, %c, %1) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:     ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:       %4 = stablehlo.add %arg3, %arg2 : tensor<f64>
// CHECK-NEXT:       stablehlo.return %4 : tensor<f64>
// CHECK-NEXT:     }) : (tensor<4x4xf64>, tensor<4x2xi64>, tensor<4xf64>) -> tensor<4x4xf64>
// CHECK-NEXT:     return %3 : tensor<4x4xf64>
// CHECK-NEXT: }
