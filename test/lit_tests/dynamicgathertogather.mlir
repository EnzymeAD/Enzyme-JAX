// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<4x4xf64>) -> tensor<2xf64> {
    %c = stablehlo.constant dense<1> : tensor<2xi64>
    %c_0 = stablehlo.constant dense<[[1, -1], [2, 0]]> : tensor<2x2xi64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xf64>) -> tensor<4x4xf64>
    %1 = "stablehlo.dynamic_gather"(%0, %c_0, %c) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>}> : (tensor<4x4xf64>, tensor<2x2xi64>, tensor<2xi64>) -> tensor<2xf64>
    return %1 : tensor<2xf64>
  }
}

// CHECK: func.func @main(%arg0: tensor<4x4xf64>) -> tensor<2xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<{{\[\[}}1, -1{{\]}}, {{\[}}2, 0{{\]\]}}> : tensor<2x2xi64>
// CHECK-NEXT:    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xf64>) -> tensor<4x4xf64>
// CHECK-NEXT:    %1 = "stablehlo.gather"(%0, %c) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1>}> : (tensor<4x4xf64>, tensor<2x2xi64>) -> tensor<2xf64>
// CHECK-NEXT:    return %1 : tensor<2xf64>
// CHECK-NEXT:  }

module {
  func.func @main(%arg0: tensor<4x4xf64>) -> tensor<2xf64> {
    %c = stablehlo.constant dense<1> : tensor<2xi32>
    %c_0 = stablehlo.constant dense<[[1, -1], [2, 0]]> : tensor<2x2xi64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xf64>) -> tensor<4x4xf64>
    %1 = "stablehlo.dynamic_gather"(%0, %c_0, %c) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>}> : (tensor<4x4xf64>, tensor<2x2xi64>, tensor<2xi32>) -> tensor<2xf64>
    return %1 : tensor<2xf64>
  }
}

// CHECK: func.func @main(%arg0: tensor<4x4xf64>) -> tensor<2xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<{{\[\[}}1, -1{{\]}}, {{\[}}2, 0{{\]\]}}> : tensor<2x2xi64>
// CHECK-NEXT:    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xf64>) -> tensor<4x4xf64>
// CHECK-NEXT:    %1 = "stablehlo.gather"(%0, %c) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1>}> : (tensor<4x4xf64>, tensor<2x2xi64>) -> tensor<2xf64>
// CHECK-NEXT:    return %1 : tensor<2xf64>
// CHECK-NEXT:  }
