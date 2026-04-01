// RUN: enzymexlamlir-opt --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --inline --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func private @sort(%arg0: tensor<2xf64> {enzymexla.memory_effects = []}) -> (tensor<2xf64>, tensor<2xf64>) attributes {enzymexla.memory_effects = []} {
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
    %0:2 = enzyme.autodiff @sort(%arg0, %cst, %cst_0) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_active>]} : (tensor<2xf64>, tensor<2xf64>, tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>)
    return %0#1, %0#0 : tensor<2xf64>, tensor<2xf64>
  }
}

// CHECK: func.func @main(%arg0: tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>) {
// CHECK-NEXT:   %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<2xf64>
// CHECK-NEXT:   %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<2xf64>
// CHECK-NEXT:   %c = stablehlo.constant dense<[0, 1]> : tensor<2xui32>
// CHECK-NEXT:   %0:2 = "stablehlo.sort"(%arg0, %c) <{dimension = 0 : i64, is_stable = false}> ({
// CHECK-NEXT:   ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<ui32>, %arg4: tensor<ui32>):
// CHECK-NEXT:     %3 = stablehlo.compare  LT, %arg1, %arg2 : (tensor<f64>, tensor<f64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %3 : tensor<i1>
// CHECK-NEXT:   }) : (tensor<2xf64>, tensor<2xui32>) -> (tensor<2xf64>, tensor<2xui32>)
// CHECK-NEXT:   %1 = stablehlo.reshape %0#1 : (tensor<2xui32>) -> tensor<2x1xui32>
// CHECK-NEXT:   %2 = "stablehlo.scatter"(%cst_1, %1, %cst_0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:   ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %cst : tensor<f64>
// CHECK-NEXT:   }) : (tensor<2xf64>, tensor<2x1xui32>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:   return %2, %arg0 : tensor<2xf64>, tensor<2xf64>
// CHECK-NEXT: }

module {
  func.func private @sort(%arg0: tensor<5x4x3x2xf32>) -> (tensor<f32>, tensor<5x4x3x2xf32>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.transpose %arg0, dims = [3, 2, 1, 0] : (tensor<5x4x3x2xf32>) -> tensor<2x3x4x5xf32>
    %1 = "stablehlo.sort"(%0) <{dimension = 2 : i64, is_stable = false}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %4 = stablehlo.compare  LT, %arg1, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %4 : tensor<i1>
    }) : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<2x3x4x5xf32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<2x3x4x5xf32>, tensor<f32>) -> tensor<f32>
    return %3, %arg0 : tensor<f32>, tensor<5x4x3x2xf32>
  }
  func.func @main(%arg0: tensor<5x4x3x2xf32>) -> (tensor<5x4x3x2xf32>, tensor<5x4x3x2xf32>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0:2 = enzyme.autodiff @sort(%arg0, %cst) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_const>]} : (tensor<5x4x3x2xf32>, tensor<f32>) -> (tensor<5x4x3x2xf32>, tensor<5x4x3x2xf32>)
    return %0#1, %0#0 : tensor<5x4x3x2xf32>, tensor<5x4x3x2xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<5x4x3x2xf32>) -> (tensor<5x4x3x2xf32>, tensor<5x4x3x2xf32>) {
// CHECK-NEXT:   %c = stablehlo.constant dense<"0x000000000000000000000000000000000000000001000000010000000100000001000000010000000200000002000000020000000200000002000000030000000300000003000000030000000300000000000000000000000000000000000000000000000100000001000000010000000100000001000000020000000200000002000000020000000200000003000000030000000300000003000000030000000000000000000000000000000000000000000000010000000100000001000000010000000100000002000000020000000200000002000000020000000300000003000000030000000300000003000000000000000000000000000000000000000000000001000000010000000100000001000000010000000200000002000000020000000200000002000000030000000300000003000000030000000300000000000000000000000000000000000000000000000100000001000000010000000100000001000000020000000200000002000000020000000200000003000000030000000300000003000000030000000000000000000000000000000000000000000000010000000100000001000000010000000100000002000000020000000200000002000000020000000300000003000000030000000300000003000000"> : tensor<2x3x4x5xui32>
// CHECK-NEXT:   %cst = arith.constant dense<0.000000e+00> : tensor<2x3x4x5xf32>
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [3, 2, 1, 0] : (tensor<5x4x3x2xf32>) -> tensor<2x3x4x5xf32>
// CHECK-NEXT:   %1:2 = "stablehlo.sort"(%0, %c) <{dimension = 2 : i64, is_stable = false}> ({
// CHECK-NEXT:   ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<ui32>, %arg4: tensor<ui32>):
// CHECK-NEXT:     %6 = stablehlo.compare  LT, %arg1, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %6 : tensor<i1>
// CHECK-NEXT:   }) : (tensor<2x3x4x5xf32>, tensor<2x3x4x5xui32>) -> (tensor<2x3x4x5xf32>, tensor<2x3x4x5xui32>)
// CHECK-NEXT:   %2 = arith.addf %1#0, %1#0 : tensor<2x3x4x5xf32>
// CHECK-NEXT:   %3 = stablehlo.reshape %1#1 : (tensor<2x3x4x5xui32>) -> tensor<2x3x4x5x1xui32>
// CHECK-NEXT:   %4 = "stablehlo.scatter"(%cst, %3, %2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [2], input_batching_dims = [0, 1, 3], scatter_indices_batching_dims = [0, 1, 3], scatter_dims_to_operand_dims = [2], index_vector_dim = 4>, unique_indices = true}> ({
// CHECK-NEXT:   ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:     stablehlo.return %arg2 : tensor<f32>
// CHECK-NEXT:   }) : (tensor<2x3x4x5xf32>, tensor<2x3x4x5x1xui32>, tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
// CHECK-NEXT:   %5 = stablehlo.transpose %4, dims = [3, 2, 1, 0] : (tensor<2x3x4x5xf32>) -> tensor<5x4x3x2xf32>
// CHECK-NEXT:   return %5, %arg0 : tensor<5x4x3x2xf32>, tensor<5x4x3x2xf32>
// CHECK-NEXT: }
