// RUN: enzymexlamlir-opt --pass-pipeline="any(enzyme-hlo-generate-td{patterns=scatter_const_fold(1024);slice_simplify},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<2x3xf32> {enzymexla.memory_effects = []}) -> (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<6x6xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<6xf32>
    %c = stablehlo.constant dense<[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]> : tensor<6x2xi64>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = "stablehlo.scatter"(%cst, %c, %cst_0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      stablehlo.return %cst_1 : tensor<f32>
    }) : (tensor<6x6xf32>, tensor<6x2xi64>, tensor<6xf32>) -> tensor<6x6xf32>
    %1 = stablehlo.slice %0 [0:6, 0:1] : (tensor<6x6xf32>) -> tensor<6x1xf32>
    %2 = stablehlo.reshape %1 : (tensor<6x1xf32>) -> tensor<2x3xf32>
    %3 = stablehlo.slice %0 [0:6, 1:2] : (tensor<6x6xf32>) -> tensor<6x1xf32>
    %4 = stablehlo.reshape %3 : (tensor<6x1xf32>) -> tensor<2x3xf32>
    %5 = stablehlo.slice %0 [0:6, 2:3] : (tensor<6x6xf32>) -> tensor<6x1xf32>
    %6 = stablehlo.reshape %5 : (tensor<6x1xf32>) -> tensor<2x3xf32>
    %7 = stablehlo.slice %0 [0:6, 3:4] : (tensor<6x6xf32>) -> tensor<6x1xf32>
    %8 = stablehlo.reshape %7 : (tensor<6x1xf32>) -> tensor<2x3xf32>
    %9 = stablehlo.slice %0 [0:6, 4:5] : (tensor<6x6xf32>) -> tensor<6x1xf32>
    %10 = stablehlo.reshape %9 : (tensor<6x1xf32>) -> tensor<2x3xf32>
    %11 = stablehlo.slice %0 [0:6, 5:6] : (tensor<6x6xf32>) -> tensor<6x1xf32>
    %12 = stablehlo.reshape %11 : (tensor<6x1xf32>) -> tensor<2x3xf32>
    return %2, %4, %6, %8, %10, %12 : tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>
  }
}

// CHECK: module {
// CHECK-NEXT:   func.func @main(%arg0: tensor<2x3xf32> {enzymexla.memory_effects = []}) -> (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT{LITERAL}:     %cst = stablehlo.constant dense<[[0.000000e+00], [0.000000e+00], [0.000000e+00], [0.000000e+00], [0.000000e+00], [1.000000e+00]]> : tensor<6x1xf32>
// CHECK-NEXT{LITERAL}:     %cst_0 = stablehlo.constant dense<[[0.000000e+00], [0.000000e+00], [0.000000e+00], [0.000000e+00], [1.000000e+00], [0.000000e+00]]> : tensor<6x1xf32>
// CHECK-NEXT{LITERAL}:     %cst_1 = stablehlo.constant dense<[[0.000000e+00], [0.000000e+00], [0.000000e+00], [1.000000e+00], [0.000000e+00], [0.000000e+00]]> : tensor<6x1xf32>
// CHECK-NEXT{LITERAL}:     %cst_2 = stablehlo.constant dense<[[0.000000e+00], [0.000000e+00], [1.000000e+00], [0.000000e+00], [0.000000e+00], [0.000000e+00]]> : tensor<6x1xf32>
// CHECK-NEXT{LITERAL}:     %cst_3 = stablehlo.constant dense<[[0.000000e+00], [1.000000e+00], [0.000000e+00], [0.000000e+00], [0.000000e+00], [0.000000e+00]]> : tensor<6x1xf32>
// CHECK-NEXT{LITERAL}:     %cst_4 = stablehlo.constant dense<[[1.000000e+00], [0.000000e+00], [0.000000e+00], [0.000000e+00], [0.000000e+00], [0.000000e+00]]> : tensor<6x1xf32>
// CHECK-NEXT:     %0 = stablehlo.reshape %cst_4 : (tensor<6x1xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:     %1 = stablehlo.reshape %cst_3 : (tensor<6x1xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:     %2 = stablehlo.reshape %cst_2 : (tensor<6x1xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:     %3 = stablehlo.reshape %cst_1 : (tensor<6x1xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:     %4 = stablehlo.reshape %cst_0 : (tensor<6x1xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:     %5 = stablehlo.reshape %cst : (tensor<6x1xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:     return %0, %1, %2, %3, %4, %5 : tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }