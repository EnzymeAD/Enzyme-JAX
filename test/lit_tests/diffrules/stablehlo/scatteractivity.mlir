// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_dup mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

module {
  func.func public @main(%arg3: tensor<3xf32>, %arg6: tensor<5xf32>) -> (tensor<5xf32>) {
    %c_0 = stablehlo.constant dense<[[3], [2], [1]]> : tensor<3x1xi32>
    %33 = "stablehlo.scatter"(%arg6, %c_0, %arg3) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg9: tensor<f32>, %arg10: tensor<f32>):
      stablehlo.return %arg10 : tensor<f32>
    }) : (tensor<5xf32>, tensor<3x1xi32>, tensor<3xf32>) -> tensor<5xf32>
    return %33 : tensor<5xf32>
  }
}

// FORWARD:  func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>, %arg2: tensor<5xf32>, %arg3: tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>) {
// FORWARD-NEXT:    %c = stablehlo.constant dense<{{\[\[}}3], [2], [1{{\]\]}}> : tensor<3x1xi32>
// FORWARD-NEXT:    %0:2 = "stablehlo.scatter"(%arg2, %arg3, %c, %arg0, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// FORWARD-NEXT:    ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>):
// FORWARD-NEXT:      stablehlo.return %arg6, %arg7 : tensor<f32>, tensor<f32>
// FORWARD-NEXT:    }) : (tensor<5xf32>, tensor<5xf32>, tensor<3x1xi32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<5xf32>, tensor<5xf32>)
// FORWARD-NEXT:    return %0#0, %0#1 : tensor<5xf32>, tensor<5xf32>
// FORWARD-NEXT:  }

// REVERSE: func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<5xf32>, %arg2: tensor<5xf32>, %arg3: tensor<5xf32>) -> tensor<3xf32> {
// REVERSE-NEXT:     %c = stablehlo.constant dense<{{\[\[}}3], [2], [1{{\]\]}}> : tensor<3x1xi32>
// REVERSE-NEXT:     %cst = arith.constant dense<0.000000e+00> : tensor<5xf32>
// REVERSE-NEXT:     %cst_0 = arith.constant dense<0.000000e+00> : tensor<3xf32>
// REVERSE-NEXT:     %0 = arith.addf %arg3, %cst : tensor<5xf32>
// REVERSE-NEXT:     %1 = "stablehlo.gather"(%0, %c) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<5xf32>, tensor<3x1xi32>) -> tensor<3xf32>
// REVERSE-NEXT:     %2 = arith.addf %1, %cst_0 : tensor<3xf32>
// REVERSE-NEXT:     return %2 : tensor<3xf32>
// REVERSE-NEXT: }
