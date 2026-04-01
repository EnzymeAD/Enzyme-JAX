// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s --check-prefix=REVERSE

func.func @main(%arg0: tensor<8x4xf64>) -> (tensor<8x4xf64>) {
  %0 = "stablehlo.sort"(%arg0) <{dimension = 0 : i64, is_stable = false}> ({
  ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
    %1 = stablehlo.compare  LT, %arg1, %arg2 : (tensor<f64>, tensor<f64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  }) : (tensor<8x4xf64>) -> tensor<8x4xf64>
  return %0 : tensor<8x4xf64>
}

// FORWARD: func.func @main(%arg0: tensor<8x4xf64>, %arg1: tensor<8x4xf64>) -> (tensor<8x4xf64>, tensor<8x4xf64>) {
// FORWARD-NEXT:   %0:2 = "stablehlo.sort"(%arg0, %arg1) <{dimension = 0 : i64, is_stable = false}> ({
// FORWARD-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<f64>, %arg5: tensor<f64>):
// FORWARD-NEXT:     %1 = stablehlo.compare  LT, %arg2, %arg3 : (tensor<f64>, tensor<f64>) -> tensor<i1>
// FORWARD-NEXT:     stablehlo.return %1 : tensor<i1>
// FORWARD-NEXT:   }) : (tensor<8x4xf64>, tensor<8x4xf64>) -> (tensor<8x4xf64>, tensor<8x4xf64>)
// FORWARD-NEXT:   return %0#0, %0#1 : tensor<8x4xf64>, tensor<8x4xf64>
// FORWARD-NEXT: }

// REVERSE: func.func @main(%arg0: tensor<8x4xf64>, %arg1: tensor<8x4xf64>) -> tensor<8x4xf64> {
// REVERSE-NEXT:   %cst = arith.constant dense<0.000000e+00> : tensor<8x4xf64>
// REVERSE-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<8x4xui32>
// REVERSE-NEXT:   %1:2 = "stablehlo.sort"(%arg0, %0) <{dimension = 0 : i64, is_stable = false}> ({
// REVERSE-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<ui32>, %arg5: tensor<ui32>):
// REVERSE-NEXT:     %4 = stablehlo.compare  LT, %arg2, %arg3 : (tensor<f64>, tensor<f64>) -> tensor<i1>
// REVERSE-NEXT:     stablehlo.return %4 : tensor<i1>
// REVERSE-NEXT:   }) : (tensor<8x4xf64>, tensor<8x4xui32>) -> (tensor<8x4xf64>, tensor<8x4xui32>)
// REVERSE-NEXT:   %2 = stablehlo.reshape %1#1 : (tensor<8x4xui32>) -> tensor<8x4x1xui32>
// REVERSE-NEXT:   %3 = "stablehlo.scatter"(%cst, %2, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], input_batching_dims = [1], scatter_indices_batching_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = true}> ({
// REVERSE-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// REVERSE-NEXT:     %4 = stablehlo.add %arg2, %arg3 : tensor<f64>
// REVERSE-NEXT:     stablehlo.return %4 : tensor<f64>
// REVERSE-NEXT:   }) : (tensor<8x4xf64>, tensor<8x4x1xui32>, tensor<8x4xf64>) -> tensor<8x4xf64>
// REVERSE-NEXT:   return %3 : tensor<8x4xf64>
// REVERSE-NEXT: }
