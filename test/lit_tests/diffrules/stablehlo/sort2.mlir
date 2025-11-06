// RUN: enzymexlamlir-opt --enzyme --canonicalize --remove-unnecessary-enzyme-ops --inline --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func private @"Const{TestLayer}(TestLayer([1.0, 2.0]))_autodiff"(%arg0: tensor<2xf64> {enzymexla.memory_effects = []}) -> (tensor<2xf64>, tensor<2xf64>) attributes {enzymexla.memory_effects = []} {
    %0 = "stablehlo.sort"(%arg0) <{dimension = 0 : i64, is_stable = false}> ({
    ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
      %1 = stablehlo.compare  LT, %arg1, %arg2 : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<2xf64>) -> tensor<2xf64>
    return %0, %arg0 : tensor<2xf64>, tensor<2xf64>
  }
  func.func @main(%arg0: tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<2xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<2xf64>
    %0:2 = enzyme.autodiff @"Const{TestLayer}(TestLayer([1.0, 2.0]))_autodiff"(%arg0, %cst, %cst_0) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_active>]} : (tensor<2xf64>, tensor<2xf64>, tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>)
    return %0#1, %0#0 : tensor<2xf64>, tensor<2xf64>
  }
}

// CHECK: func.func @main(%arg0: tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>) {
// CHECK-NEXT:   %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:   %cst_0 = arith.constant dense<0.000000e+00> : tensor<2xf64>
// CHECK-NEXT:   %cst_1 = arith.constant dense<1.000000e+00> : tensor<2xf64>
// CHECK-NEXT:   %c = stablehlo.constant dense<[0, 1]> : tensor<2xui32>
// CHECK-NEXT:   %0:2 = "stablehlo.sort"(%arg0, %c) <{dimension = 0 : i64, is_stable = false}> ({
// CHECK-NEXT:   ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<ui32>, %arg4: tensor<ui32>):
// CHECK-NEXT:     %3 = stablehlo.compare  LT, %arg1, %arg2 : (tensor<f64>, tensor<f64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %3 : tensor<i1>
// CHECK-NEXT:   }) : (tensor<2xf64>, tensor<2xui32>) -> (tensor<2xf64>, tensor<2xui32>)
// CHECK-NEXT:   %1 = stablehlo.reshape %0#1 : (tensor<2xui32>) -> tensor<2x1xui32>
// CHECK-NEXT:   %2 = "stablehlo.scatter"(%cst_0, %1, %cst_1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:   ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %cst : tensor<f64>
// CHECK-NEXT:   }) : (tensor<2xf64>, tensor<2x1xui32>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:   return %2, %arg0 : tensor<2xf64>, tensor<2xf64>
// CHECK-NEXT: }
